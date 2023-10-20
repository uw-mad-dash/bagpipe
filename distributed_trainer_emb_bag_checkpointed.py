# I think this is it. All looks good for now
# training worker, accepts input from the oracle cacher
import os
import sys
import time
import copy
import queue
import logging
import argparse
import threading
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed.rpc as rpc
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import utils

try:
    import s3_utils
except:
    pass
from operator import itemgetter

from subprocess import call


class DistTrainModel(nn.Module):
    def __init__(
        self,
        emb_size=1,
        ln_top=None,
        ln_bot=None,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        feature_interaction="dot",
        interact_itself=False,
        loss_function="bce",
        worker_id=0,
        lookahead_value=200,
        cache_size=25000,
        mini_batch_size=128,
        ln_emb=[],
        training_worker_id=0,
        device="cuda:0",
        oracle_prefix=None,
        emb_prefix=None,
        trainer_prefix=None,
        trainer_world_size=None,
        cleanup_batch_proportion=0.25,
    ):
        super(DistTrainModel, self).__init__()
        """
        Args:
            emb_size: Size of for each sparse embedding
            ln_top (np.array): Structure of top MLP
            ln_bot (np.array): Structure of bottom MLP
            sigmoid_bot (int): Integer for listing the location of bottom
            sigmoid_top (int): Integer for listing the location of the top
        Returns:
            None
        """
        self.emb_size = emb_size
        self.ln_top = ln_top
        self.ln_bot = ln_bot
        self.sigmoid_bot = sigmoid_bot
        self.sigmoid_top = sigmoid_top
        self.feature_interaction = feature_interaction
        self.interact_itself = interact_itself
        self.lookahead_value = lookahead_value
        self.mini_batch_size = mini_batch_size
        self.ln_emb = ln_emb
        self.trainer_world_size = trainer_world_size
        self.training_worker_id = training_worker_id
        self.device = device

        self.bot_mlp = self.create_mlp(self.ln_bot, self.sigmoid_bot)
        self.top_mlp = self.create_mlp(self.ln_top, self.sigmoid_top)
        self.top_mlp.to(self.device)
        self.bot_mlp.to(self.device)
        if loss_function == "bce":
            self.loss_fn = torch.nn.BCELoss(reduction="mean")
        elif loss.function == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.loss_fn.to(self.device)

        # holds global to local mapping
        self.cache_idx = {
            k: torch.ones(idx_vals, dtype=torch.long).to(self.device)
            for k, idx_vals in enumerate(self.ln_emb)
        }

        for k in self.cache_idx:
            self.cache_idx[k][:] = -1

        # sparse drastically speeds it up
        self.local_cache = nn.EmbeddingBag(
            cache_size, self.emb_size, mode="sum", sparse=True
        ).to(device)

        # NOTE: I am deciding to keep these mapping on CPU.
        # TODO: Verify the tradeoffs here
        self.local_to_global_mapping = torch.ones((cache_size, 2), dtype=torch.long)
        self.local_to_global_mapping[:] = torch.tensor([-1, -1])
        # -1 represents empty
        # 1 represent filled
        with torch.no_grad():
            self.local_cache_status = torch.ones(cache_size, dtype=torch.int32)
            self.local_cache_status[:] = -1

        self.local_cache_ttl = torch.ones(cache_size, dtype=torch.long)
        self.local_cache_ttl[:] = -1

        self.train_queue = queue.Queue()
        self.prefetch_queue = queue.PriorityQueue()
        self.prefetch_futures_queue = queue.Queue()
        self.prefetch_queue_ttl = queue.Queue()
        self.original_idx_puts = queue.Queue()
        self.delete_element_queue = queue.Queue()
        self.prefetch_completed_signal = queue.Queue()

        self.later_sync_future = None
        self.sync_later_buffer = None

        self.prefetch_reorder_buffer = dict()
        self.dynamic_lookahead_val = dict()
        self.prefetch_expected_iter = 0

        self.worker_id = worker_id
        self.trainer_prefix = trainer_prefix
        self.worker_name = f"{trainer_prefix}_{training_worker_id}"
        self.emb_worker = f"{emb_prefix}"
        self.current_train_epoch = 0
        assert (
            cleanup_batch_proportion <= 1
        ), "Batch proportion should be between 0 and 1, currently greater than 1"

        assert (
            cleanup_batch_proportion >= 0
        ), "Batch proportion should be between 0 and 1, currently less than 0"

        self.cleanup_interval = max(
            1, int(cleanup_batch_proportion * self.lookahead_value)
        )
        self.iter_cleaned_up = 0
        self.eviction_number = 0  # use it to load balance

        self.cache_usage = 0
        self.max_cache_usage = 0

        # counters
        self.total_forward_time = 0
        self.total_backward_time = 0
        self.waiting_for_prefetch = 0
        self.cache_sync_time = 0
        self.total_time = 0

        self.total_evictions = 0
        self.total_additions = 0
        try:
            self.write_files = s3_utils.uploadFile("recommendation-data-bagpipe")
        except:
            pass

        self.prev_ttl_idx = None
        self.prev_ttl_val = None
        return None

    def convert_orig_to_local_by_table_id(self, table_id, global_id):
        """
        Give a table id and tensor of global ids converts it to local ids
        """
        return self.cache_idx[table_id][global_id]

    def convert_orig_to_local_idx(self, orig_idx):
        """
        Converts original indexes from original indexes to local cache indexes

        orig_idx (list(list)): List of list containing original index
        """
        local_cache_idxs = list()
        for table_id, idxs in enumerate(orig_idx):
            local_cache_idxs.append(self.cache_idx[table_id][idxs])
        return local_cache_idxs

    def update_ttl(self, ttl_update_idx, ttl_update_val):
        """
        Only update the TTL of the cache
        """
        for table_id in ttl_update_idx:
            orig_idx_to_update = ttl_update_idx[table_id]
            corresponding_local_idx = self.cache_idx[table_id][orig_idx_to_update]

            values_to_update = ttl_update_val[table_id]
            self.local_cache_ttl[corresponding_local_idx] = values_to_update
        return None

    def clean_up_caches(self, iter_cleanup, ttl_update_idx, ttl_update_val):
        """
        Based on stored TTLs evict from the caches
        """
        try:

            print("Clean up cache called {}".format(iter_cleanup))
            print("Cleanup Interval {}".format(self.cleanup_interval))

            # this is the TTL update code moved from critical path
            # NOTE: Following code mode moved to def update_ttl
            # we needed to update TTL pre eviction call.
            # ttl update
            # we got rid of it for the example based setup
            for table_id in ttl_update_idx:
                orig_idx_to_update = ttl_update_idx[table_id]
                corresponding_local_idx = self.cache_idx[table_id][orig_idx_to_update]

                values_to_update = ttl_update_val[table_id]
                self.local_cache_ttl[corresponding_local_idx] = values_to_update

            if iter_cleanup % self.cleanup_interval == 0 and iter_cleanup != 0:

                self.iter_cleaned_up = iter_cleanup - 1
                self.eviction_number += 1
                cleanup_time = time.time()
                local_idx_to_remove = (self.local_cache_ttl <= iter_cleanup - 1) & (
                    self.local_cache_ttl != -1
                )
                # print("Local cache ttl {}".format(self.local_cache_ttl))
                local_idx_to_remove = local_idx_to_remove.nonzero().squeeze()
                self.cache_usage -= len(local_idx_to_remove)
                self.total_evictions += len(local_idx_to_remove)
                global_idx_to_remove = self.local_to_global_mapping[local_idx_to_remove]
                # we get global idx to remove
                dict_to_update = dict()
                emb_to_update = dict()
                # we need to reassamble the embeddings
                for table_id in range(len(self.ln_emb)):
                    # for each table ID find the global indexes
                    relevant_idx_to_table_id = global_idx_to_remove[:, 0] == table_id
                    relevant_idx_to_table_id = relevant_idx_to_table_id.nonzero()

                    # if relevant_idx_to_table_id.shape[0] != 1:
                    relevant_idx_to_table_id = relevant_idx_to_table_id.squeeze()
                    # print("Relevant idx shape {}".format(relevant_idx_to_table_id.shape))
                    if len(relevant_idx_to_table_id.shape) == 0:
                        relevant_idx_to_table_id = torch.tensor(
                            [relevant_idx_to_table_id.item()]
                        )
                    global_idx_to_update_table_id = global_idx_to_remove[
                        relevant_idx_to_table_id
                    ]
                    global_idx_to_update_table_id = global_idx_to_update_table_id[:, 1]
                    # we have indexes to update
                    dict_to_update[table_id] = global_idx_to_update_table_id
                    # we need corresponding values
                    relevant_local_ids = self.convert_orig_to_local_by_table_id(
                        table_id, global_idx_to_update_table_id
                    )

                    with torch.no_grad():
                        # corresponding values
                        embedding_vals = self.local_cache(
                            relevant_local_ids,
                            torch.arange(len(relevant_local_ids), device=self.device),
                        )
                        emb_to_update[table_id] = embedding_vals.to("cpu")

                    # cleaning up global to local mapping
                    self.cache_idx[table_id][global_idx_to_update_table_id] = -1
                # TODO: Code for dynamic
                # delete_elements = range(
                # iter_cleanup - self.cleanup_interval, iter_cleanup
                # )
                # for dm in delete_elements:
                # del self.dynamic_lookahead_val[dm]

                # if iter_cleanup % self.trainer_world_size == self.training_worker_id:
                # # only send one workers embedding to update
                # # already synchronized
                # # print("Iter cleaned up {}".format(iter_cleanup))
                # rpc.rpc_async(
                # self.emb_worker,
                # cache_eviction_update,
                # args=(
                # (
                # dict_to_update,
                # emb_to_update,
                # )
                # ),
                # )
                # print("Cache eviction made {}".format(dict_to_update))
                # embeddings sent to update
                self.local_to_global_mapping[local_idx_to_remove] = torch.tensor(
                    [-1, -1]
                )
                self.local_cache_status[local_idx_to_remove] = -1
                self.local_cache_ttl[local_idx_to_remove] = -1

                end_cleanup_time = time.time()
                # print("Time to clean up {}".format(end_cleanup_time - cleanup_time))
            prefetch_prep_time = time.time()

            batched_embedding = list()
            enter_flag = 0
            while True:
                # batching the prefetch requests
                # only fetch when needed
                # print("In while True")
                # print("Prefetch Queue value {}".format(
                is_empty = self.prefetch_queue.empty()
                # if is_empty:
                # # keep going until there is an element in prefetch queue
                # logger.
                # continue
                # print("is empty {}".format(is_empty))
                if not is_empty:
                    # if prefetech queue is not empty
                    # print("In first if")

                    # supporting dynamic lookahead
                    # TODO: Enable this for dynamic lookahead value
                    # self.lookahead_value = self.dynamic_lookahead_val[
                    # self.iter_cleaned_up
                    # ]

                    # NOTE This is condition for prefetch queue using usual queue
                    # if (
                    # list(self.prefetch_queue.queue[0].keys())[0]
                    # < self.iter_cleaned_up - 1 + self.lookahead_value
                    # ):

                    # NOTE: this is condition for prefetch queue using priority queue
                    # logger.info(
                    # "Prefetch queue 1st element{}".format(
                    # self.prefetch_queue.queue[0][0]
                    # )
                    # )

                    # logger.info("Iter cleaned up {}".format(self.iter_cleaned_up))
                    if (
                        self.prefetch_queue.queue[0][0]
                        < self.iter_cleaned_up - 1 + self.lookahead_value
                    ):
                        # print("Iter cleaned up {}".format(self.iter_cleaned_up))
                        enter_flag = 1
                        # print("Getting prefetch queue")
                        ttl_val, val = self.prefetch_queue.get(block=True)
                        if ttl_val != self.prefetch_expected_iter:
                            while ttl_val != self.prefetch_expected_iter:
                                self.prefetch_queue.put((ttl_val, val))
                                ttl_val, val = self.prefetch_queue.get(block=True)

                        self.prefetch_expected_iter += 1

                        # print("Got from the queue")
                        batched_embedding.append(val)
                        # we are getting ttl val anyways
                        ttl_val = list(val.keys())[0]
                        # TODO: Enable for dynamic
                        # self.dynamic_lookahead_val[ttl_val] = val["lookahead_value"]
                        # print("Launched prefetch {}".format(ttl_val))
                        self.original_idx_puts.put(val)
                        self.prefetch_queue_ttl.put(ttl_val)
                    else:
                        if enter_flag == 1:
                            # print("Making a prefetch req")
                            fut = rpc.rpc_async(
                                self.emb_worker,
                                get_embedding,
                                args=(batched_embedding,),
                            )
                            self.prefetch_futures_queue.put(fut)
                            enter_flag = 0

                        break

                        # print("Launched prefetch {}".format(ttl_val))
                else:
                    if enter_flag == 1:
                        fut = rpc.rpc_async(
                            self.emb_worker,
                            get_embedding,
                            args=(batched_embedding,),
                        )

                        self.prefetch_futures_queue.put(fut)
                        enter_flag = 0
                        # print("Launched prefetch {}".format(ttl_val))
                        # print("Second else")
                        # print("prefetch queue {}".format(self.prefetch_queue))
                        break

                    if enter_flag == 0:
                        # keep checking prefetch queue
                        continue
            end_prefetch_pre_time = time.time()
            # print(
            # "Prefetch prep time {}".format(
            # end_prefetch_pre_time - prefetch_prep_time
            # )
            # )
            print("Out of prefetch prep loop")
            # adding to the cache post eviction
            while True:
                if not self.prefetch_queue_ttl.empty():
                    prefetch_add_time = time.time()

                    fut = self.prefetch_futures_queue.get(block=True)
                    time_spent_waiting = time.time()
                    # fetched_batch = fut
                    fetched_batch = fut.wait()
                    # print(
                    # "Time spend waiting on future {}".format(
                    # time.time() - time_spent_waiting
                    # )

                    # )
                    # print("Fetched iter {}".format(ttl_val))

                    empty_location = self.local_cache_status == -1
                    empty_location = empty_location.nonzero().squeeze()

                    if self.cache_usage > self.max_cache_usage:
                        self.max_cache_usage = self.cache_usage
                    # logger.info("Cache Size Remaining {}".format(len(empty_location)))
                    logger.info("Cache Size used {}".format(self.cache_usage))
                    last_use_location = 0
                    # assert (
                    # len(empty_location) >= emb_vals_length
                    # ), "Not enought Cache Available"

                    # cuda_start_movement = torch.cuda.Event(enable_timing=True)
                    # cuda_stop_movement = torch.cuda.Event(enable_timing=True)
                    for fetched_vals in fetched_batch:
                        ttl_val = self.prefetch_queue_ttl.get(block=True)
                        # print("Fetched itr {}".format(ttl_val))
                        # print("Evicted till {}".format(self.iter_cleaned_up))
                        val = self.original_idx_puts.get(block=True)
                        total_time_movement_per_batch = 0
                        for table_id, emb_vals in enumerate(fetched_vals):
                            if len(emb_vals) > 0:
                                emb_vals_length = len(emb_vals)
                                self.total_additions += emb_vals_length
                                with torch.cuda.stream(s):
                                    # cuda_start_movement.record()
                                    emb_vals = emb_vals.to(self.device)
                                    indexing_offset = torch.arange(
                                        emb_vals_length, device=self.device
                                    )
                                # cuda_stop_movement.record()

                                original_idx = val[ttl_val][table_id]

                                empty_location_to_use = empty_location[
                                    last_use_location : last_use_location
                                    + emb_vals_length
                                ]
                                # changing indexing avoid running lookups agains and again
                                last_use_location += emb_vals_length
                                self.cache_usage += emb_vals_length
                                self.local_cache_status[empty_location_to_use] = 1
                                empty_location_to_use_device = empty_location_to_use.to(
                                    self.device
                                )

                                self.local_to_global_mapping[
                                    empty_location_to_use, 0
                                ] = table_id

                                self.local_to_global_mapping[
                                    empty_location_to_use, 1
                                ] = original_idx

                                self.cache_idx[table_id][
                                    original_idx
                                ] = empty_location_to_use_device

                                self.local_cache_ttl[empty_location_to_use] = ttl_val

                                # torch.cuda.synchronize()
                                # total_time_movement_per_batch += (
                                # cuda_start_movement.elapsed_time(cuda_stop_movement)
                                # )

                                torch.cuda.current_stream().wait_stream(s)
                                with torch.no_grad():
                                    self.local_cache(
                                        empty_location_to_use_device,
                                        indexing_offset,
                                    ).data = emb_vals

                        # logger.info(
                        # "Total time data movement {}".format(
                        # total_time_movement_per_batch
                        # )
                        # )
                        self.prefetch_completed_signal.put(ttl_val)

                    # end_prefetch_add_time = time.time()
                    # logger.info(
                    # "prefetch add time (ms) {}".format(
                    # (end_prefetch_add_time - prefetch_add_time) * 1000
                    # )
                    # )
                else:
                    # no elements in the queue
                    break

            if iter_cleanup % self.cleanup_interval == 0 and iter_cleanup != 0:
                if (
                    self.eviction_number % self.trainer_world_size
                    == self.training_worker_id
                ):
                    # only send one workers embedding to update
                    # already synchronized
                    # print("Iter cleaned up {}".format(iter_cleanup))
                    print("evicted till {}".format(iter_cleanup - 1))
                    rpc.rpc_async(
                        self.emb_worker,
                        cache_eviction_update,
                        args=(
                            (
                                dict_to_update,
                                emb_to_update,
                            )
                        ),
                    )

        except Exception as e:
            sys.exit(e.__str__())
            logger.error(e)
            print(e)

    def create_mlp(self, ln, sigmoid_layer):
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]
            LL = nn.Linear(int(n), int(m), bias=True)
            # some xavier stuff the original pytorch code was doing
            mean = 0.0
            std_dev = np.sqrt(2 / (m + n))
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        return torch.nn.Sequential(*layers)

    def apply_mlp(self, dense_x, mlp_network):
        """
        Apply MLP on the features
        """
        # print("Shape Dense x {}".format(dense_x.shape))
        return mlp_network(dense_x)

    def get_elements_evicted(self, round_number):
        """
        Get element IDs which will be evicted in this round
        """
        evicted_in_this_round = self.local_cache_ttl == round_number
        evicted_in_this_round = evicted_in_this_round.nonzero().squeeze()
        num_elements_in_cache = self.local_cache_ttl > round_number
        num_elements_in_cache = num_elements_in_cache.nonzero().squeeze().shape
        print("Num Elemnts in cache {}".format(num_elements_in_cache))
        return evicted_in_this_round

    def get_intersection(self, int_a, int_b):
        """
        Calculate intersection of int_a and int_b.
        Also return the indices of intersection with respect to int_b
        """
        # NOTE: The values help in
        a_cat_b, counts = torch.cat([int_a, int_b]).unique(return_counts=True)
        intersect_a_b = a_cat_b[torch.where(counts.gt(1))]
        idx_in_b = (intersect_a_b.unsqueeze(1) == int_b).nonzero()[:, 1]
        return intersect_a_b, idx_in_b

    def get_only_in_b(self, intersect_a_b, int_b):
        """
        Return elements present only in b
        """
        # NOTE: It is very interesting that this works.
        # intersect_a_b - delay_update_elements_local_avail
        # int_b - grad_update_element_idx
        # delay_update_local_avail -> is the elments in the intersection of elements which are being evicted and elements which have been updated by gradient currently(grad_update_element_idx)
        # grad_update_element_idx -> is all the elements which are updated in current round
        # the interesting point is that all elements in delay_update_local_avail will be in the grad_update_element_idx, so we just need to concatenate and find elements which have a frequency of 1
        # print("Intersect a_b {}".format(intersect_a_b.shape))
        # print("int_b {}".format(int_b.shape))
        a_cat_b, counts = torch.cat([intersect_a_b, int_b]).unique(return_counts=True)
        # print("A cat b {}".format(a_cat_b))
        # print(" Counte {}".format(counts))
        elements_b = a_cat_b[torch.where(counts.eq(1))]
        idx_in_b = (elements_b.unsqueeze(1) == int_b).nonzero()[:, 1]
        return elements_b, idx_in_b

    def find_unique_embs(self, emb_vals):
        """
        Find unique embs
        """
        split_unique_start = torch.cuda.Event(enable_timing=True)
        split_unique_stop = torch.cuda.Event(enable_timing=True)
        time_spent_unique = 0
        with torch.no_grad():
            unique_embs = list()
            for table_id, emb_id in enumerate(emb_vals):
                local_cache_id = self.cache_idx[table_id][emb_id].tolist()
                unique_embs.extend(local_cache_id)
        return torch.tensor(unique_embs)

    def sync_only_next_round(self, round_number, next_round_emb):
        """
        Split the training based on learning
        """
        split_start_time = torch.cuda.Event(enable_timing=True)
        split_stop_time = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():

            embeddings_needed_next_iter = self.find_unique_embs(next_round_emb)
            embeddings_needed_next_iter = embeddings_needed_next_iter.to(self.device)

            self.local_cache.weight.grad = self.local_cache.weight.grad.coalesce()

            grad_update_element_idx = copy.deepcopy(
                self.local_cache.weight.grad.indices().squeeze()
            )
            grad_update_element_values = copy.deepcopy(
                self.local_cache.weight.grad.values()
            )
            grad_update_element_shape = copy.deepcopy(
                self.local_cache.weight.grad.size()
            )

            # print("Grad update element shape {}".format(grad_update_element_idx.shape))

            (
                elements_to_sync_now_local_avail,
                elements_to_sync_now_idx_original,
            ) = self.get_intersection(
                embeddings_needed_next_iter, grad_update_element_idx
            )

            (
                delay_update_elements_local_avail,
                delay_update_elements_idx_original,
            ) = self.get_only_in_b(
                elements_to_sync_now_local_avail, grad_update_element_idx
            )

            # print("Delay elements {}".format(delay_update_elements_local_avail.shape))
            # print("Elements to sync {}".format(elements_to_sync_now_local_avail.shape))

            values_to_sync_now = grad_update_element_values[
                elements_to_sync_now_idx_original
            ]

            values_to_sync_later = grad_update_element_values[
                delay_update_elements_idx_original
            ]

            elements_to_sync_now_local_avail = (
                elements_to_sync_now_local_avail.unsqueeze(0)
            )
            vector_to_sync_now = torch.sparse_coo_tensor(
                elements_to_sync_now_local_avail,
                values_to_sync_now,
                size=grad_update_element_shape,
            )

            if round_number != 0:
                self.later_sync_future.wait()

            dist.all_reduce(vector_to_sync_now, async_op=False)

            if round_number != 0:
                self.local_cache.weight.grad = (
                    vector_to_sync_now + self.sync_later_buffer
                )
            else:
                self.local_cache.weight.grad = vector_to_sync_now
            delay_update_elements_local_avail = (
                delay_update_elements_local_avail.unsqueeze(0)
            )
            self.sync_later_buffer = torch.sparse_coo_tensor(
                delay_update_elements_local_avail,
                values_to_sync_later,
                size=grad_update_element_shape,
            )

            self.later_sync_future = dist.all_reduce(
                self.sync_later_buffer, async_op=True
            )

            return None

    def sync_split(self, round_number):
        """
        Split the embedding sync path. Embeddings which will be needed in future will be synchronized immediately
        while embeddings which will be evicted in this round are synced async separately.
        """
        with torch.no_grad():
            # coealesce the gradient
            self.local_cache.weight.grad = self.local_cache.weight.grad.coalesce()
            # find the elements from the cache which will be evicted and which  will not be evicted
            # evicted in this round = sync later
            # not currently evicted =  sync now because we will need them
            evicted_in_this_round = self.get_elements_evicted(round_number)
            evicted_in_this_round = evicted_in_this_round.to(self.device)
            # print("Evicted in this round length {}".format(evicted_in_this_round.shape))
            # this is wehere we keep our dense elements
            # dense elements are where we keep up elements which are going to be evicted in future
            # NOTE: These elements can be synced using dense as well and then converted to sparse vector.
            # Especially I think that hot elements which are usually constantly in cache might benefit the most.
            # I will give it a shot later but for the first cut we will just create a sparse vector.

            # get local gradients updates indices and values
            grad_update_element_idx = copy.deepcopy(
                self.local_cache.weight.grad.indices().squeeze()
            )
            grad_update_element_values = copy.deepcopy(
                self.local_cache.weight.grad.values()
            )
            grad_update_element_shape = copy.deepcopy(
                self.local_cache.weight.grad.size()
            )

            # elements which are going to evicted and also have grad updates available locally
            # print(next(self.top_mlp.parameters()).device)
            # print("Evicted In this round {}".format(evicted_in_this_round))
            # print("Grad update element {}".format(grad_update_element_idx))
            # print("Grad update element shape {}".format(grad_update_element_idx.shape))
            (
                delay_update_elements_local_avail,
                delay_update_elements_idx_original,
            ) = self.get_intersection(evicted_in_this_round, grad_update_element_idx)

            # print(
            # "Delay update elements local avail {}".format(
            # delay_update_elements_local_avail.shape
            # )
            # )
            # print(
            # "Delay update idx original {}".format(
            # delay_update_elements_idx_original.shape
            # )
            # )

            # elements evicted in this round will be delayed sync
            # extract elements from the original vector
            (
                elements_to_sync_now,
                elements_to_sync_now_idx_original,
            ) = self.get_only_in_b(
                delay_update_elements_local_avail, grad_update_element_idx
            )

            # print("Elements to sync now {}".format(elements_to_sync_now.shape))

            # print(
            # "Elements to sync now idx original {}".format(
            # elements_to_sync_now_idx_original.shape
            # )
            # )

            values_to_sync_later = grad_update_element_values[
                delay_update_elements_idx_original
            ]

            values_to_sync_now = grad_update_element_values[
                elements_to_sync_now_idx_original
            ]

            # elements going to be evicted in future are going to synced now
            # print("Elements to sync now {}".format(elements_to_sync_now))
            # print("Values to sync {}".format(values_to_sync_now))
            vector_to_sync_now = torch.sparse_coo_tensor(
                elements_to_sync_now.unsqueeze_(0),
                values_to_sync_now,
                size=grad_update_element_shape,
            )

            if round_number != 0:
                # wait for the previous sync to finish
                self.later_sync_future.wait()

            dist.all_reduce(vector_to_sync_now, async_op=False)

            # we now have access to current gradient updates

            if round_number != 0:
                self.local_cache.weight.grad = (
                    vector_to_sync_now + self.sync_later_buffer
                )
            else:
                self.local_cache.weight.grad = vector_to_sync_now

            self.sync_later_buffer = torch.sparse_coo_tensor(
                delay_update_elements_local_avail.unsqueeze_(0),
                values_to_sync_later,
                size=grad_update_element_shape,
            )

            self.later_sync_future = dist.all_reduce(
                self.sync_later_buffer, async_op=True
            )

            return None

    def sync_id(self):
        # sync_embeddings = list()
        with torch.no_grad():
            # TODO: verify that this actually works in place
            dist.all_reduce(self.local_cache.weight.grad)

    def apply_emb(self, lS_i):
        """
        Fetch embedding
        """
        fetched_embeddings = list()
        for table_id, emb_id in enumerate(lS_i):
            # find the corresponding id to fetch
            local_cache_id = self.cache_idx[table_id][emb_id]
            # NOTE: Commented checking code
            # local_cache_invalid = local_cache_id == -1
            # local_cache_invalid = local_cache_invalid.nonzero().squeeze()
            # print(local_cache_invalid)
            # if len(local_cache_invalid.shape) == 0:
            # local_cache_invalid = torch.tensor([local_cache_invalid.item()])

            # if len(local_cache_invalid) != 0:
            # print("Table {} Emb {}".format(table_id, emb_id[local_cache_invalid]))
            # sys.exit("Invalid cache indexed")
            # print("local cache id device {}".format(local_cache_id.device))
            embs = self.local_cache(
                local_cache_id,
                torch.arange(len(local_cache_id)).to(self.device),
            )
            fetched_embeddings.append(embs)
        return fetched_embeddings

    def interact_features(self, x, ly):
        """
        Interaction between dense and embeddings
        """
        # Copied from interact features function of original code
        if self.feature_interaction == "dot":
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            _, ni, nj = Z.shape
            offset = 1 if self.interact_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.feature_interaction == "cat":
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit("Unsupported feature interaction")
        return R

    def forward(self, dense_x, lS_i, target):
        """
        Forward pass of the training
        """
        # first we perform bottom MLP
        # print("Dense x shape {}".format(dense_x.shape))
        # start_time = torch.cuda.Event(enable_timing=True)
        # stop_time = torch.cuda.Event(enable_timing=True)
        # start_time.record()
        x = self.apply_mlp(dense_x, self.bot_mlp)
        # stop_time.record()
        # torch.cuda.synchronize()
        # print("Time apply bottom mlp {}".format(start_time.elapsed_time(stop_time)))
        # need to fetch the embeddings
        # at this point we will either have embeddings in the local cache or
        # global cache
        # if handle_all_reduce is not None:
        # check if all reduce is done or not
        # handle_all_reduce.wait()
        # start_time.record()
        ly = self.apply_emb(lS_i)
        # stop_time.record()
        # torch.cuda.synchronize()
        # print("Time apply emb {}".format(start_time.elapsed_time(stop_time)))
        # print(x)
        # print(ly)
        # feature interaction
        # start_time.record()
        z = self.interact_features(x, ly)
        # stop_time.record()
        # torch.cuda.synchronize()
        # print("Time feature interat {}".format(start_time.elapsed_time(stop_time)))

        # pass through top mlp
        # start_time.record()
        p = self.apply_mlp(z, self.top_mlp)
        # stop_time.record()
        # torch.cuda.synchronize()
        # print("Time top mlp {}".format(start_time.elapsed_time(stop_time)))
        # start_time.record()
        loss = self.loss_fn(p, target)
        # stop_time.record()
        # torch.cuda.synchronize()
        # print("Time loss calculation{}".format(start_time.elapsed_time(stop_time)))
        # print(loss)
        return loss


def update_train_queue(input_dict):
    # print("Train queue status {}".format(input_dict))
    input_key = list(input_dict.keys())[0]
    sparse_vector = input_dict[input_key]["train_data"]["sparse_vector"]
    unique_list = list()
    for k in sparse_vector:
        unique_list.append(torch.unique(k))

    input_dict[input_key]["sparse_unique"] = unique_list

    comp_intensive_model.train_queue.put(input_dict)
    return 1


s = torch.cuda.Stream()


def fill_prefetch_cache():
    num_times_run = 0
    try:
        while num_times_run < comp_intensive_model.lookahead_value:
            prefetch_time_start = time.time()
            ttl_val, val = comp_intensive_model.prefetch_queue.get(block=True)
            # ttl_val = list(val.keys())[0]

            if ttl_val != comp_intensive_model.prefetch_expected_iter:
                while ttl_val != comp_intensive_model.prefetch_expected_iter:
                    # wrong fetch putting it back
                    comp_intensive_model.prefetch_queue.put((ttl_val, val))
                    # Hopefully by now we will have gotten what we needed
                    ttl_val, val = comp_intensive_model.prefetch_queue.get(block=True)
                    # ttl_val = list(val.keys())[0]

            fut = rpc.rpc_async(
                comp_intensive_model.emb_worker, get_embedding_single, args=(val,)
            )
            comp_intensive_model.prefetch_expected_iter += 1
            # NOTE:Enable for dynamic look ahead
            # comp_intensive_model.dynamic_lookahead_val[ttl_val] = val["lookahead_value"]
            comp_intensive_model.prefetch_futures_queue.put(fut)
            comp_intensive_model.prefetch_queue_ttl.put(ttl_val)
            # keep getting prefetch queue
            fut = comp_intensive_model.prefetch_futures_queue.get(block=True)
            ttl_val = comp_intensive_model.prefetch_queue_ttl.get(block=True)
            fetched_vals = fut.wait()
            # print("Fetched vals {}".format(fetched_vals))
            total_time_movement_per_batch = 0

            # cuda_start_movement = torch.cuda.Event(enable_timing=True)
            # cuda_stop_movement = torch.cuda.Event(enable_timing=True)

            empty_location = comp_intensive_model.local_cache_status == -1
            empty_location = empty_location.nonzero().squeeze()
            last_use_location = 0
            # logger.info("Cache Size Remaining {}".format(len(empty_location)))
            # logger.info("Cache Size used {}".format(comp_intensive_model.cache_usage))
            # logger.info(
            # "Sum of free and empty {}".format(
            # len(empty_location) + comp_intensive_model.cache_usage
            # )
            # )
            for table_id, emb_vals in enumerate(fetched_vals):
                # find the original idx
                if len(emb_vals) > 0:
                    # cuda_start_movement.record()

                    emb_vals_length = len(emb_vals)

                    comp_intensive_model.total_additions += emb_vals_length
                    # cuda_start_movement.record()
                    with torch.cuda.stream(s):
                        emb_vals = emb_vals.to(
                            comp_intensive_model.device,
                        )
                        indexing_offset = torch.arange(
                            emb_vals_length, device=comp_intensive_model.device
                        )
                    # cuda_stop_movement.record()

                    # torch.cuda.synchronize()
                    # total_time_movement_per_batch += cuda_start_movement.elapsed_time(
                    # cuda_stop_movement
                    # )
                    original_idx = val[ttl_val][table_id]
                    # find empty locations in the local cache
                    # print("Empty Location {}".format(empty_location))
                    # assert (
                    # len(empty_location) >= emb_vals_length
                    # ), "Not enough Cache Avaialable"
                    empty_location_to_use = empty_location[
                        last_use_location : last_use_location + emb_vals_length
                    ]

                    last_use_location += emb_vals_length
                    comp_intensive_model.cache_usage += emb_vals_length
                    # logger.info("Empty locations {}".format(empty_location_to_use))
                    empty_location_to_use_device = empty_location_to_use.to(
                        comp_intensive_model.device
                    )
                    # updating cache structure before access
                    comp_intensive_model.local_cache_status[empty_location_to_use] = 1
                    # we copy values to the embedding table

                    # update local to global

                    # this inserts the table id -1
                    # print("Table ID {}".format(table_id))
                    comp_intensive_model.local_to_global_mapping[
                        empty_location_to_use, 0
                    ] = table_id

                    # this inserts the  emb id
                    comp_intensive_model.local_to_global_mapping[
                        empty_location_to_use, 1
                    ] = original_idx
                    comp_intensive_model.cache_idx[table_id][
                        original_idx
                    ] = empty_location_to_use_device

                    comp_intensive_model.local_cache_ttl[
                        empty_location_to_use
                    ] = ttl_val

                    torch.cuda.current_stream().wait_stream(s)
                    with torch.no_grad():
                        comp_intensive_model.local_cache(
                            empty_location_to_use_device, indexing_offset
                        ).data = emb_vals

            # logger.info(
            # "Total time data movement(ms) {}".format(total_time_movement_per_batch)
            # )
            # prefetch_time_end = time.time()
            # logger.info(
            # "Fill prefetch move time(ms) {}".format(
            # (prefetch_time_end - prefetch_time_start) * 1000
            # )
            # )
            comp_intensive_model.prefetch_completed_signal.put(ttl_val)
            num_times_run += 1
        logger.info(
            "Total size after prefetch {}".format(comp_intensive_model.total_additions)
        )
    except queue.Empty:
        pass


def update_prefetch_queue(input_dict):
    input_iter = list(input_dict.keys())[0]
    comp_intensive_model.prefetch_queue.put((input_iter, input_dict), block=True)
    return 1


def launch_cache_cleanup():
    """
    Launch cache cleanup
    """
    # print("Cache cleanup launched")
    while True:
        try:
            (
                iter_to_cleanup,
                ttl_update_idx,
                ttl_update_val,
            ) = comp_intensive_model.delete_element_queue.get(block=True)
            # print("Iter to cleanup {}".format(iter_to_cleanup))
            # print("iter to cleanup {}".format(iter_to_cleanup))
            comp_intensive_model.clean_up_caches(
                iter_to_cleanup, ttl_update_idx, ttl_update_val
            )
        except queue.Empty:
            pass


def launch_cache_cleanup_no_thread():
    """
    Launch cache cleanup
    """
    # print("Cache cleanup launched")
    # while True:
    (
        iter_to_cleanup,
        ttl_update_idx,
        ttl_update_val,
    ) = comp_intensive_model.delete_element_queue.get(block=True)
    # print("iter to cleanup {}".format(iter_to_cleanup))
    comp_intensive_model.clean_up_caches(
        iter_to_cleanup, ttl_update_idx, ttl_update_val
    )


def exit_worker(input_dict):
    rpc.shutdown()
    return 1


def cache_eviction_update(dict_to_update, emb_to_update):
    # This is dummy function real one is in embedding server
    """
    update_dict- key - (table_id, emb_id): tensor to store
    """
    emb_grouped_by_table_id = defaultdict(list)
    emb_id_grouped_by_table_id = defaultdict(list)

    for key in update_dict:
        table_id, emb_id = key
        emb_grouped_by_table_id[table_id].append(update_dict[key])
        emb_id_grouped_by_table_id[table_id].append(emb_id)

    for key in emb_grouped_by_table_id:
        grouped_by_table_id[key] = torch.tensor(emb_grouped_by_table_id[key])
        emb_id_grouped_by_table_id[key] = torch.tensor(emb_id_grouped_by_table_id[key])
    embedding_object.update_embeddings(
        emb_grouped_by_table_id, emb_id_grouped_by_table_id
    )
    return 1


def get_embedding(input_list):
    # This is dummy function real one is in embedding server
    """
    These are prefetch embeddings
    Args:
        input_list (list(tuples)): List of tuples, tuples(table_id, emb_id)
    """
    emb_decompressed = defaultdict(list)
    for table_id, emb_id in emb_decompressed:
        emb_decompressed[table_id].append(emb_id)

    fetched_embeddings = embedding_object.get_embeddings(emb_decompressed)
    return fetched_embeddings


def get_embedding_single(input_list):
    # This is dummy function real one is in embedding server
    """
    These are prefetch embeddings
    Args:
        input_list (list(tuples)): List of tuples, tuples(table_id, emb_id)
    """
    emb_decompressed = defaultdict(list)
    for table_id, emb_id in emb_decompressed:
        emb_decompressed[table_id].append(emb_id)

    fetched_embeddings = embedding_object.get_embeddings(emb_decompressed)
    return fetched_embeddings


def main(args):
    expected_iter = 0
    iter_overflow = dict()
    os.environ["MASTER_ADDR"] = args.master_ip
    os.environ["MASTER_PORT"] = args.master_port
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    arch_mlp_top_adjusted = (
        str(
            utils.get_first_layer_size_top_mlp(
                args.arch_interaction_op,
                args.arch_interaction_itself,
                ln_bot,
                args.ln_emb,
            )
        )
        + "-"
        + args.arch_mlp_top
    )
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
    print("LN TOP {}".format(ln_top))
    global comp_intensive_model
    comp_intensive_model = DistTrainModel(
        emb_size=args.emb_size,
        ln_bot=ln_bot,
        ln_top=ln_top,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        loss_function=args.loss_function,
        feature_interaction=args.arch_interaction_op,
        worker_id=args.worker_id,
        lookahead_value=args.lookahead_value,
        cache_size=args.cache_size,
        ln_emb=args.ln_emb,
        mini_batch_size=args.mini_batch_size,
        training_worker_id=args.dist_worker_id,
        device=args.device,
        emb_prefix=args.emb_prefix,
        trainer_prefix=args.trainer_prefix,
        trainer_world_size=args.world_size_trainers,
        cleanup_batch_proportion=args.cleanup_batch_proportion,
    )

    # rpc fuctions

    # rpc setup
    # rpc.init_rpc(
    # comp_intensive_model.worker_name,
    # rank=args.worker_id,
    # world_size=args.world_size,
    # )
    print("Rpc init")
    # dist.init_process_group(
    # backend=args.dist_backend,
    # init_method=f"tcp://{args.dist_master_ip}:{args.master_port_ccom}",
    # world_size=args.world_size_trainers,
    # rank=args.dist_worker_id,
    # )

    # comp_intensive_model.top_mlp = DDP(comp_intensive_model.top_mlp, device_ids=[0])
    # comp_intensive_model.bot_mlp = DDP(comp_intensive_model.bot_mlp, device_ids=[0])
    # comp_intensive_model.local_cache = DDP(
    # comp_intensive_model.local_cache, device_ids=[0]
    # )
    start_time_save = time.time()
    torch.save(
        {
            "epc": 0,
            "model": comp_intensive_model.state_dict(),
            "cache": comp_intensive_model.local_to_global_mapping,
            "cache_stat": comp_intensive_model.local_cache,
            "cache_ttl": comp_intensive_model.local_cache_ttl,
        },
        "set.pt",
    )
    print("time to save model {}".format(time.time() - start_time_save))

    load_time = time.time()
    comp_intensive_model.load_state_dict(torch.load("set.pt")["model"])
    print("Load time {}".format(time.time() - load_time))
    import ipdb

    ipdb.set_trace()
    # thread for waiting on prefetching

    # prefetch_update_thread = threading.Thread(target=update_prefetch_cache)
    # prefetch_update_thread.start()

    cleanup_and_update_thread = threading.Thread(target=launch_cache_cleanup)
    cleanup_and_update_thread.start()

    optimizer = optim.SGD(
        [
            {
                "params": comp_intensive_model.local_cache.parameters(),
                "lr": 0.01,
            },
            {
                "params": comp_intensive_model.top_mlp.parameters(),
                "lr": 0.01,
            },
            {
                "params": comp_intensive_model.bot_mlp.parameters(),
                "lr": 0.01,
            },
        ]
    )

    # train loop

    # this will fill the prefetch cache

    # this is the handle we use for storing all reduce future
    fill_prefetch_cache()
    next_example = None
    current_example = None
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_stop = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_stop = torch.cuda.Event(enable_timing=True)
    sync_time = torch.cuda.Event(enable_timing=True)
    sync_time_stop = torch.cuda.Event(enable_timing=True)
    cache_update_sync_start = torch.cuda.Event(enable_timing=True)
    cache_update_sync_stop = torch.cuda.Event(enable_timing=True)

    while True:
        total_start = time.time()
        try:
            queue_fetch_time = time.time()
            next_example = comp_intensive_model.train_queue.get(block=True)
            queue_fetch_end = time.time()
            print("Queue Fetch Time {}".format(queue_fetch_end - queue_fetch_time))
            current_epoch = list(next_example.keys())[0]
            # handling rpc potential reordering
            if current_epoch != expected_iter:
                # move current train example to the dictionary
                iter_overflow[current_epoch] = copy.deepcopy(next_example)
                # check if we have the expected iter in the overflow
                if expected_iter in iter_overflow:
                    next_example = iter_overflow.pop(expected_iter)
                    current_epoch = list(next_example.keys())[0]
                    expected_iter += 1
                else:
                    # pop more and see if we find what we want
                    continue
            else:
                expected_iter += 1

            if current_epoch == 0:
                # need to fetch next example
                current_example = copy.deepcopy(next_example)
                next_example = comp_intensive_model.train_queue.get(block=True)
                current_epoch = list(next_example.keys())[0]
                if current_epoch != expected_iter:
                    # move current train example to the dictionary
                    iter_overflow[current_epoch] = copy.deepcopy(next_example)
                    # check if we have the expected iter in the overflow
                    if expected_iter in iter_overflow:
                        train_example = iter_overflow.pop(expected_iter)
                        current_epoch = list(next_example.keys())[0]
                        expected_iter += 1
                    else:
                        # pop more and see if we find what we want
                        continue
                else:
                    expected_iter += 1

            # train example is what is current example
            train_example = copy.deepcopy(current_example)
            current_epoch = list(train_example.keys())[0]
            # current example is what is next example
            current_example = copy.deepcopy(next_example)

            if current_epoch == args.stop_iter:
                # s3 out code
                logger.info("Total Time {}".format(comp_intensive_model.total_time))
                logger.info(
                    "Total Time forward {}".format(
                        comp_intensive_model.total_forward_time
                    )
                )

                logger.info(
                    "Total Time backward {}".format(
                        comp_intensive_model.total_backward_time
                    )
                )

                logger.info(
                    "Total Time prefetch {}".format(
                        comp_intensive_model.waiting_for_prefetch
                    )
                )

                logger.info(
                    "Total Time cache sync {}".format(
                        comp_intensive_model.cache_sync_time
                    )
                )

                logger.info(
                    "Total evictions {}".format(comp_intensive_model.total_evictions)
                )
                logger.info(
                    "Total additions {}".format(comp_intensive_model.total_additions)
                )

                logger.info(
                    "Max cache usage {}".format(comp_intensive_model.max_cache_usage)
                )
                file_name = (
                    f"training_worker_{args.dist_worker_id}_{args.logging_prefix}.log"
                )
                try:
                    comp_intensive_model.write_files.push_file(
                        file_name,
                        f"{comp_intensive_model.trainer_world_size}_trainers/{file_name}",
                    )
                except:
                    pass
                call("pkill -9 python", shell=True)

            # logger.info(f"Current Iter {current_epoch}")
            print(f"Current Iter {current_epoch}")
            checked_iter = time.time()
            check_iter_prefetch = comp_intensive_model.prefetch_completed_signal.get(
                block=True
            )
            checked_iter_time = time.time()
            comp_intensive_model.waiting_for_prefetch += (
                checked_iter_time - checked_iter
            ) * 1000
            # logger.info(
            # "Checked time(ms) {}".format((checked_iter_time - checked_iter) * 1000)
            # )
            print(f"Checked Iter {check_iter_prefetch}")
            logger.info(f"Iter {current_epoch}")
            comp_intensive_model.current_train_epoch = current_epoch

            forward_start.record()
            loss = comp_intensive_model.forward(
                train_example[current_epoch]["train_data"]["dense_x"].to(
                    comp_intensive_model.device
                ),
                train_example[current_epoch]["train_data"]["sparse_vector"],
                train_example[current_epoch]["train_data"]["target"].to(
                    comp_intensive_model.device
                ),
            )
            logger.info("Loss {}".format(loss.item()))
            forward_stop.record()

            backward_start.record()
            loss.backward()
            backward_stop.record()

            # backward_stop = time.time()
            # print("Time for backward {}".format(backward_stop - backward_start))

            sync_time.record()

            # ids which need sync training
            # ids which need training training
            # we are done with backward pass
            # if current_epoch != 0:

            # pass

            ttl_update_idx = train_example[current_epoch]["ttl_idx"]
            ttl_update_val = train_example[current_epoch]["ttl_val"]

            # cache_update_sync_start.record()
            # comp_intensive_model.update_ttl(ttl_update_idx, ttl_update_val)
            # cache_update_sync_stop.record()
            # cache_update_sync_stop.synchronize()
            # print(
            # "Time for cache update {}".format(
            # cache_update_sync_start.elapsed_time(cache_update_sync_stop)
            # )
            # )

            # comp_intensive_model.sync_split(current_epoch)
            comp_intensive_model.sync_only_next_round(
                current_epoch,
                next_example[current_epoch + 1]["sparse_unique"],
            )
            # comp_intensive_model.sync_id()
            sync_time_stop.record()

            # optim_time = time.time()
            optimizer.step()
            optimizer.zero_grad()
            # optim_time_end = time.time()
            # print("Optim Time end to end {}".format(optim_time_end - optim_time))

            # decide which elemenst are going to live longer
            # elements_to_cache = train_example[current_epoch]["cache_elements"]
            # data_move = time.time()

            # ttl_update_time = torch.cuda.Event(enable_timing=True)
            # ttl_update_time_end = torch.cuda.Event(enable_timing=True)

            # ttl_update_time.record()
            if current_epoch != 0:
                # we are updating one behind the training
                comp_intensive_model.delete_element_queue.put(
                    (
                        current_epoch - 1,
                        comp_intensive_model.prev_ttl_idx,
                        comp_intensive_model.prev_ttl_val,
                    )
                )
            comp_intensive_model.prev_ttl_idx = copy.deepcopy(ttl_update_idx)
            comp_intensive_model.prev_ttl_val = copy.deepcopy(ttl_update_val)
            # data_move_end = time.time()
            # print("Data move end {}".format(data_move_end - data_move))
            # launch_cache_cleanup_no_thread()
            # del loss
            torch.cuda.synchronize()
            logger.info(
                "Forward time(ms) {}".format(forward_start.elapsed_time(forward_stop))
            )

            comp_intensive_model.total_forward_time += forward_start.elapsed_time(
                forward_stop
            )

            logger.info(
                "Backward time(ms) {}".format(
                    backward_start.elapsed_time(backward_stop)
                )
            )

            comp_intensive_model.total_backward_time += backward_start.elapsed_time(
                backward_stop
            )
            logger.info(
                "Time for cache sync(ms) {}".format(
                    sync_time.elapsed_time(sync_time_stop)
                )
            )
            comp_intensive_model.cache_sync_time += sync_time.elapsed_time(
                sync_time_stop
            )
            total_end = time.time()
            logger.info(
                "Total end to end time(ms) {}".format((total_end - total_start) * 1000)
            )

            comp_intensive_model.total_time += (total_end - total_start) * 1000

        except queue.Empty:
            pass


def get_emb_length(in_file):
    with open(in_file, "r") as fin:
        data = fin.readlines()

    data = [int(d) for d in data]
    return data


def parse_args(parser):
    parser.add_argument(
        "--arch-mlp-bot",
        type=utils.dash_separated_ints,
        help="dimensions of the bottom mlp",
    )
    parser.add_argument(
        "--arch-mlp-top", type=utils.dash_separated_ints, help="dimensions of top mlp"
    )

    parser.add_argument(
        "--emb-size",
        type=int,
        default=48,
        help="size of the embedding for each sparse feature",
    )

    parser.add_argument(
        "--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot"
    )

    parser.add_argument(
        "--lookahead-value",
        type=int,
        default=200,
        help="The number of batches further to look ahead for getting cache",
    )

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    parser.add_argument("--loss-function", choices=["mse", "bce"], default="bce")
    parser.add_argument(
        "--ln-emb",
        type=utils.dash_separated_ints,
        help="embedding table sizes in the right order",
        default=[
            1460,
            583,
            10131227,
            2202608,
            305,
            24,
            12517,
            633,
            3,
            93145,
            5683,
            8351593,
            3194,
            27,
            14992,
            5461306,
            10,
            5652,
            2173,
            4,
            7046547,
            18,
            15,
            286181,
            105,
            142572,
        ],
    )
    parser.add_argument(
        "--mini-batch-size", type=int, default=128, help="The batch size to train"
    )
    parser.add_argument("--cache-size", type=int, required=True)
    parser.add_argument(
        "--worker-id",
        type=int,
        required=True,
        help="Global worker ID, i.e., rank for RPC init",
    )
    parser.add_argument(
        "--world-size", type=int, required=True, help="Global world size"
    )
    parser.add_argument("--master-ip", type=str, default="localhost")
    parser.add_argument("--master-port", type=str, default="18000")
    parser.add_argument(
        "--master-port-ccom",
        type=int,
        default=9988,
        help="Port for collective communication",
    )
    parser.add_argument("--dist-backend", type=str, default="gloo")

    parser.add_argument(
        "--dist-master-ip", type=str, required=True, help="IP of rank 0 training worker"
    )
    parser.add_argument(
        "--dist-worker-id",
        type=int,
        required=True,
        help="Distributed Worker ID, for collective collection library",
    )
    parser.add_argument(
        # "--dist-world-size",
        "--world-size-trainers",
        type=int,
        required=True,
        help="Distributed World Size for collective communication library",
    )

    parser.add_argument(
        "--oracle-prefix",
        type=str,
        default="oracle",
        help="Prefix to name oracle cacher",
    )

    parser.add_argument(
        "--trainer-prefix",
        type=str,
        default="worker",
        help="prefix to call the trainer",
    )

    parser.add_argument(
        "--emb-prefix",
        type=str,
        default="emb_worker",
        help="Name of embedding worker Currently I am assuming there is currently only one worker ID",
    )

    parser.add_argument(
        "--logging-prefix", type=str, default="test", help="Add for logging"
    )

    parser.add_argument(
        "--cleanup-batch-proportion",
        type=float,
        default=0.25,
        help="The proportion of lookahead value at which we evict and fetch",
    )
    parser.add_argument("--stop-iter", type=int, default=1000, help="Add for logging")
    parser.add_argument("--emb-info-file", type=str, default=None)
    parser.add_argument("--delay-emb-sync", action="store_true", default=False)
    args = parser.parse_args()

    if args.emb_info_file is not None:
        args.ln_emb = get_emb_length(args.emb_info_file)

    return args


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(description="Arguments for DLRM"))
    logging.basicConfig(
        filename=f"training_worker_{args.dist_worker_id}_{args.logging_prefix}.log"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(args)
    main(args)
