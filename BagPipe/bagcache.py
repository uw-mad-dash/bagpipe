import queue
import threading
import numpy as np

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.distributed as dist

class BagCache:
    prefetch_queue = queue.PriorityQueue()
    train_queue = queue.PriorityQueue()
    fetchable = queue.Queue()

    def __init__(
        self,
        lookahead_value,
        emb_size,
        ln_emb,
        cache_size,
        ragged,
        cleanup_batch_proportion,
        emb_optim,
        emb_optim_params,
        device,
        worker_id,
        world_size,
        training_worker_id,
        logger,
        trainer_world_size,
        cache_eviction_update,
        get_embedding,
    ):
        self.emb_size = emb_size
        self.lookahead_value = lookahead_value
        self.use_cache = self.lookahead_value != 1

        assert (
            cleanup_batch_proportion <= 1
        ), "Batch proportion should be between 0 and 1, currently greater than 1"

        assert (
            cleanup_batch_proportion >= 0
        ), "Batch proportion should be between 0 and 1, currently less than 0"

        self.cleanup_interval = max(
            1, int(cleanup_batch_proportion * self.lookahead_value)
        )
        self.ln_emb = ln_emb
        self.ln_emb_len = len(ln_emb)
        self.ragged = ragged
        self.worker_id = worker_id
        self.world_size = world_size
        self.training_worker_id = training_worker_id
        self.trainer_world_size = trainer_world_size
        self.trainer_prefix = "worker"
        self.emb_prefix = "emb_worker"
        self.oracle_prefix = "oracle"
        self.worker_name = f"{self.trainer_prefix}_{training_worker_id}"
        self.emb_worker = f"{self.emb_prefix}"
        self.device = device

        self.cache_idx = {
            k: torch.ones(idx_vals, dtype=torch.long, device=self.device)
            for k, idx_vals in enumerate(self.ln_emb)
        }

        for k in self.cache_idx:
            self.cache_idx[k][:] = -1

        self.local_cache = nn.EmbeddingBag(
            cache_size, self.emb_size, mode="sum", sparse=True, device=self.device
        )

        self.local_to_global_mapping = torch.ones((cache_size, 2), dtype=torch.long, device=self.device)
        self.local_to_global_mapping[:] = torch.tensor([-1, -1], device=self.device)
        # -1 represents empty
        # 1 represent filled
        self.local_cache_status = torch.ones(cache_size, dtype=torch.int32, device=self.device)
        self.local_cache_status[:] = -1

        self.local_cache_ttl = torch.ones(cache_size, dtype=torch.long, device=self.device)
        self.local_cache_ttl[:] = -1

        self.delete_element_queue = queue.Queue()
        self.prefetch_completed_signal = queue.Queue()
        self.local_no_sync_accumulator = [list() for _ in range(self.ln_emb_len)]
        
        self.s = torch.cuda.Stream(device=self.device)
        self.sync_later_s = torch.cuda.Stream(device=self.device)

        self.cached_syncs = None
        self.sync_later_future = None
        self.sync_later_size = None
        self.sync_later_indices = None
        self.sync_later_values = None

        self.iter_cleaned_up = 0
        self.eviction_number = 0  # use it to load balance

        self.iter_overflow = dict()
        self.iter_num = 0
        self.previous_example_ttl_idx = None
        self.previous_example_ttl_val = None
        self.current_example = None

        self.optimizer = emb_optim(self.local_cache.parameters(), **emb_optim_params)

        self.logger = logger

        self.cache_eviction_update = cache_eviction_update
        self.get_embedding = get_embedding

        rpc.init_rpc(
            self.worker_name,
            rank=self.worker_id,
            world_size=self.world_size,
        )
        
        if self.logger is not None:
            self.logger.info("Rpc init")
            
        if self.use_cache:
            # only if we are using bagpipe
            # in case we are running the baseline no need to perform this
            threading.Thread(target=self.launch_cache_cleanup).start()
            self.fill_prefetch_cache()
        else:
            _, self.current_example = BagCache.train_queue.get(block=True)

    def convert_orig_to_local_by_table_id(self, table_id, global_id):
        """
        Give a table id and tensor of global ids converts it to local ids
        """
        return self.cache_idx[table_id][global_id]

    def fetch_elements(self):
        """
        We only use this for baseline.
        We specifically call this to fetch elements in the training system.
        Pick elements out of prefetch queue.
        """
        empty_location = self.local_cache_status.eq(-1).nonzero(as_tuple=True)[0]
        last_use_location = 0
        _, prefetch, _ = BagCache.prefetch_queue.get(block=True)
        prefetch_list = prefetch["prefetch_list"]
        tensor_prefetch = [list() for _ in range(self.ln_emb_len)]
        for table_id in range(self.ln_emb_len):
            tensor_prefetch[table_id] = torch.from_numpy(prefetch_list[table_id])
        fut = rpc.rpc_async(
            self.emb_worker,
            self.get_embedding,
            args=(tensor_prefetch,),
        )
        fetched_vals = fut.wait()
        for table_id, emb_vals in enumerate(fetched_vals):
            if len(emb_vals) > 0:
                emb_vals_length = len(emb_vals)
                emb_vals = emb_vals.to(self.device)
                indexing_offset = torch.arange(emb_vals_length, device=self.device)
                original_idx = tensor_prefetch[table_id].to(self.device)
                empty_location_to_use = empty_location[last_use_location : last_use_location + emb_vals_length]
                last_use_location += emb_vals_length
                self.local_cache_status[empty_location_to_use] = 1
                self.local_to_global_mapping[empty_location_to_use, 0] = table_id
                self.local_to_global_mapping[empty_location_to_use, 1] = original_idx
                self.cache_idx[table_id][original_idx] = empty_location_to_use
                with torch.no_grad():
                    self.local_cache(empty_location_to_use, indexing_offset).data = emb_vals

    def delete_cache_elements(self):
        # remove all elements from the cache
        dict_to_update = dict()
        emb_to_update = dict()
        local_idx_to_remove = self.local_cache_status != -1
        global_idx_to_remove = self.local_to_global_mapping[local_idx_to_remove]
        for table_id in range(len(self.ln_emb)):
            # for each table ID find the global indexes
            global_idx_to_update_table_id = global_idx_to_remove[global_idx_to_remove[:, 0].eq(table_id)]
            global_idx_to_update_table_id = global_idx_to_update_table_id[:, 1]
            # we have indexes to update
            dict_to_update[table_id] = global_idx_to_update_table_id.to("cpu")
            # we need corresponding values
            relevant_local_ids = self.convert_orig_to_local_by_table_id(table_id, global_idx_to_update_table_id)

            with torch.no_grad():
                # corresponding values
                embedding_vals = self.local_cache(
                    relevant_local_ids,
                    torch.arange(len(relevant_local_ids), device=self.device),
                )
                emb_to_update[table_id] = embedding_vals.to("cpu")

            # cleaning up global to local mapping
            self.cache_idx[table_id][global_idx_to_update_table_id] = -1
        self.local_to_global_mapping[local_idx_to_remove] = torch.tensor([-1, -1], device=self.device)
        self.local_cache_status[local_idx_to_remove] = -1

        if self.worker_id == 0:
            rpc.rpc_sync(
                self.emb_worker,
                self.cache_eviction_update,
                args=(
                    (
                        dict_to_update,
                        emb_to_update,
                    )
                ),
            )

    def clean_up_caches(self, iter_cleanup, ttl_update_idx, ttl_update_val):
        """
        Based on stored TTLs evict from the caches
        """
        for table_id in ttl_update_idx:
            orig_idx_to_update = ttl_update_idx[table_id]
            corresponding_local_idx = self.cache_idx[table_id][orig_idx_to_update]

            values_to_update = ttl_update_val[table_id]
            self.local_cache_ttl[corresponding_local_idx] = values_to_update.to(self.device)
        
        #Evict Code
        if iter_cleanup % self.cleanup_interval == 0 and iter_cleanup != 0:
            self.iter_cleaned_up = iter_cleanup
            self.eviction_number += 1
            with torch.cuda.stream(self.s):
                local_idx_to_remove = torch.logical_and(self.local_cache_ttl.lt(iter_cleanup), self.local_cache_ttl.ne(-1))
                global_idx_to_remove = self.local_to_global_mapping[local_idx_to_remove]
                
                # we get global idx to remove
                dict_to_update = dict()
                emb_to_update = dict()
                # we need to reassamble the embeddings
                for table_id in range(self.ln_emb_len):
                    # for each table ID find the global indexes we have indexes to update
                    
                    # collect first #self.cleanup_interval of local_no_sync, remove them from accumulator
                    cat_local_no_sync = np.concatenate(self.local_no_sync_accumulator[table_id][:self.cleanup_interval])
                    if len(cat_local_no_sync) == 0:
                        local_no_sync = torch.tensor([], dtype=torch.long).to(self.device)
                    else:
                        local_no_sync = torch.from_numpy(cat_local_no_sync).to(self.device)
                    
                    self.local_no_sync_accumulator[table_id][:self.cleanup_interval] = []
                    
                    # get the global indexes, pruned with local_no_sync, and sort to do slicing
                    global_update_table_id = global_idx_to_remove[global_idx_to_remove[:, 0].eq(table_id)][:, 1]
                    global_update_table_id_pruned = global_update_table_id[torch.isin(global_update_table_id, local_no_sync, assume_unique=True, invert=True)]
                    sorted_global_idx, _ = torch.sort(global_update_table_id_pruned)
                    
                    # Slicing and then concat with local_no_sync
                    if len(sorted_global_idx) < self.trainer_world_size:
                        if self.training_worker_id == 0:
                            dict_to_update[table_id] = torch.cat([sorted_global_idx, local_no_sync])
                        else:
                            dict_to_update[table_id] = local_no_sync
                    else:
                        chunk_size = len(sorted_global_idx) // self.trainer_world_size
                        if self.training_worker_id == self.trainer_world_size - 1:
                            dict_to_update[table_id] = torch.cat([sorted_global_idx[self.training_worker_id * chunk_size : ], local_no_sync])
                        else:
                            dict_to_update[table_id] = torch.cat([sorted_global_idx[self.training_worker_id * chunk_size : (self.training_worker_id + 1) * chunk_size], local_no_sync])
                    
                    # we need corresponding values
                    relevant_local_ids = self.convert_orig_to_local_by_table_id(table_id, dict_to_update[table_id])
                    
                    dict_to_update[table_id] = dict_to_update[table_id].to("cpu", non_blocking=True)
                    
                    with torch.no_grad():
                        # corresponding values
                        embedding_vals = self.local_cache(
                            relevant_local_ids,
                            torch.arange(len(relevant_local_ids), device=self.device),
                        )
                        emb_to_update[table_id] = embedding_vals.to("cpu", non_blocking=True)

                    self.cache_idx[table_id][global_update_table_id] = -1
                # embeddings sent to update
                self.local_to_global_mapping[local_idx_to_remove] = torch.tensor([-1, -1], device=self.device)
                self.local_cache_status[local_idx_to_remove] = -1
                self.local_cache_ttl[local_idx_to_remove] = -1
                
                rpc.rpc_async(
                    self.emb_worker,
                    self.cache_eviction_update,
                    args=(
                        (
                            dict_to_update,
                            emb_to_update,
                        )
                    ),
                )
                if self.logger is not None:
                    self.logger.info("cache eviction on {}".format(self.iter_cleaned_up))
        
            # query the emb server whether we can prefetch
            BagCache.fetchable.get()
            if not BagCache.prefetch_queue.empty():
                batch_counter = 0
                batched_prefetch = [list() for _ in range(self.ln_emb_len)]
                last_ttl = -1
                while not BagCache.prefetch_queue.empty():
                    if BagCache.prefetch_queue.queue[0][0] < self.iter_cleaned_up + self.lookahead_value:
                        ttl_val, prefetch, sparse_vector = BagCache.prefetch_queue.get(block=True)
                        prefetch_list = prefetch["prefetch_list"]
                        no_sync = prefetch["no_sync"]
                        last_ttl = ttl_val
                        for table_id in range(self.ln_emb_len):
                            # Split the global no_sync into local_no_sync and others_no_sync
                            check_local_no_sync = np.in1d(no_sync[table_id], np.unique(sparse_vector[table_id].numpy()), assume_unique=True)
                            local_no_sync = no_sync[table_id][check_local_no_sync]
                            
                            # Accululate local_no_sync for clean_up_cache use
                            self.local_no_sync_accumulator[table_id].append(local_no_sync)
                            others_no_sync = no_sync[table_id][~check_local_no_sync]
                            
                            # Generate pruned prefetch list
                            prefetch_unpruned = prefetch_list[table_id]
                            prefetch_pruned = prefetch_unpruned[np.in1d(prefetch_unpruned, others_no_sync, assume_unique=True, invert=True)]
                            if len(prefetch_pruned) == 0:
                                batched_prefetch[table_id].append(torch.tensor([], dtype=torch.long))
                            else:
                                batched_prefetch[table_id].append(torch.from_numpy(prefetch_pruned))
                        batch_counter += 1
                    else:
                        break
                    
                if batch_counter > 0:
                    for table_id in range(self.ln_emb_len):
                        batched_prefetch[table_id] = torch.cat(batched_prefetch[table_id])
                    fut = rpc.rpc_async(
                        self.emb_worker,
                        self.get_embedding,
                        args=(batched_prefetch,),
                    )
                    fetched_list = fut.wait()
                    last_use_location = 0
                    with torch.cuda.stream(self.s):
                        empty_location = self.local_cache_status.eq(-1).nonzero(as_tuple=True)[0]
                        for table_id, emb_vals in enumerate(fetched_list):
                            if len(emb_vals) > 0:
                                emb_vals_length = len(emb_vals)
                                emb_vals = emb_vals.to(self.device)
                                indexing_offset = torch.arange(emb_vals_length, device=self.device)

                                original_idx = batched_prefetch[table_id].to(self.device)

                                empty_location_to_use = empty_location[
                                    last_use_location : last_use_location
                                    + emb_vals_length
                                ]
                                # changing indexing avoid running lookups agains and again
                                last_use_location += emb_vals_length
                                self.local_cache_status[empty_location_to_use] = 1

                                self.local_to_global_mapping[
                                    empty_location_to_use, 0
                                ] = table_id

                                self.local_to_global_mapping[
                                    empty_location_to_use, 1
                                ] = original_idx

                                self.cache_idx[table_id][
                                    original_idx
                                ] = empty_location_to_use

                                self.local_cache_ttl[empty_location_to_use] = last_ttl
                                
                                with torch.no_grad():
                                    self.local_cache(empty_location_to_use, indexing_offset).data = emb_vals
                                    
                    torch.cuda.current_stream().wait_stream(self.s)
                    for _ in range(batch_counter):
                        self.prefetch_completed_signal.put(last_ttl)

    def local_mapper(self, sync_now, sync_later):
        """
        Find unique embs
        """
        with torch.no_grad():
            local_sync_now = list()
            local_sync_later = list()
            for table_id in range(self.ln_emb_len):
                sever_point = len(sync_now[table_id])
                local_cache_id = self.cache_idx[table_id][
                    torch.cat([sync_now[table_id], sync_later[table_id]])
                ]
                local_sync_now.append(local_cache_id[:sever_point])
                local_sync_later.append(local_cache_id[sever_point:])
        self.cached_syncs = torch.cat(local_sync_now), torch.cat(local_sync_later)

    def cache_sync(self, round_number, next_example, counter=None):
        """
        Split the training based on learning
        """
        if self.use_cache:
            with torch.no_grad():
                sync_now, sync_later = self.cached_syncs

                self.local_cache.weight.grad = self.local_cache.weight.grad.coalesce()
                cache_shape = self.local_cache.weight.grad.size()

                values_sync_now = self.local_cache.weight.grad.index_select(dim=0, index=sync_now).to_dense()
                values_sync_later = self.local_cache.weight.grad.index_select(dim=0, index=sync_later).to_dense()
                if self.logger is not None:
                    self.logger.info("Sync Now Shape {}".format(values_sync_now.shape))
                    self.logger.info("Sync Later Shape {}".format(values_sync_later.shape))

                torch.cuda.current_stream().wait_stream(self.sync_later_s)
                if round_number != 0:
                    if counter is not None:
                        counter[2].record()
                        self.sync_later_future.wait()
                        counter[3].record()
                    else:
                        self.sync_later_future.wait()

                fut = dist.all_reduce(values_sync_now, async_op=True)
                
                self.prefetch_completed_signal.get(block=True)
                self.local_mapper(next_example["sync_now"], next_example["sync_later"])
                
                fut.wait()
                
                sparse_sync_now = torch.sparse_coo_tensor(
                    sync_now.unsqueeze(0), values_sync_now, size=cache_shape
                )
                if round_number != 0:
                    sparse_sync_later = torch.sparse_coo_tensor(
                        self.sync_later_indices.unsqueeze(0),
                        self.sync_later_values,
                        size=self.sync_later_size,
                    )
                    self.local_cache.weight.grad = sparse_sync_now + sparse_sync_later
                else:
                    self.local_cache.weight.grad = sparse_sync_now

                self.sync_later_size = cache_shape
                self.sync_later_indices = sync_later
                self.sync_later_values = values_sync_later
                with torch.cuda.stream(self.sync_later_s):
                    self.sync_later_future = dist.all_reduce(
                        self.sync_later_values, async_op=True
                    )
        else:
            with torch.no_grad():
                dist.all_reduce(self.local_cache.weight.grad)

    def next_batch(self, counter=None):
        if self.iter_num != 0:
            _, next_example = BagCache.train_queue.get(block=True)
            if counter is not None:
                counter[0].record()
                self.cache_sync(self.iter_num - 1, next_example, counter)
                counter[1].record()
                counter[4].record()
                self.optimizer.step()
                self.optimizer.zero_grad()
                counter[5].record()
            else:
                self.cache_sync(self.iter_num - 1, next_example)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if self.use_cache:
                if self.iter_num != 1:
                    self.delete_element_queue.put(
                        (
                            self.iter_num - 2,
                            self.previous_example_ttl_idx,
                            self.previous_example_ttl_val,
                        )
                    )
                self.previous_example_ttl_idx = self.current_example["ttl_idx"]
                self.previous_example_ttl_val = self.current_example["ttl_val"]
            else:
                self.delete_cache_elements()
            self.current_example = next_example
        
        self.iter_num += 1
        if self.use_cache:
            if self.logger is not None:
                self.logger.info("prefetch queue size {}".format(BagCache.prefetch_queue.qsize()))
                self.logger.info("prefetch_completed_signal queue size {}".format(self.prefetch_completed_signal.qsize()))
                self.logger.info("Iter cleanup {}".format(self.iter_cleaned_up))
        else:
            # fetch the elements for the baseline
            self.fetch_elements()
        return self.current_example["train_data"]

    def get_emb(self, ls_i):
        """
        Fetch embedding
        """
        fetched_embeddings = list()
        for table_id, emb_ids in enumerate(ls_i):
            # find the corresponding id to fetch
            local_cache_id = self.cache_idx[table_id][emb_ids.flatten() if self.ragged else emb_ids]
            embs = self.local_cache(
                local_cache_id,
                torch.arange(len(local_cache_id), device=self.device),
            )
            fetched_embeddings.append(
                embs.reshape(emb_ids.size(dim=0), -1, self.emb_size) if self.ragged else embs
            )
        return fetched_embeddings

    def fill_prefetch_cache(self):
        num_times_rum = 0
        while num_times_rum < self.lookahead_value:
            batch_counter = 0
            batched_prefetch = [list() for _ in range(self.ln_emb_len)]
            last_ttl = -1
            while batch_counter < self.cleanup_interval:
                ttl_val, prefetch, sparse_vector = BagCache.prefetch_queue.get(block=True)
                prefetch_list = prefetch["prefetch_list"]
                no_sync = prefetch["no_sync"]
                last_ttl = ttl_val
                for table_id in range(self.ln_emb_len):
                    check_local_no_sync = np.in1d(no_sync[table_id], np.unique(sparse_vector[table_id].numpy()), assume_unique=True)
                    local_no_sync = no_sync[table_id][check_local_no_sync]
                    self.local_no_sync_accumulator[table_id].append(local_no_sync)
                    others_no_sync = no_sync[table_id][~check_local_no_sync]
                    prefetch_unpruned = prefetch_list[table_id]
                    prefetch_pruned = prefetch_unpruned[np.in1d(prefetch_unpruned, others_no_sync, assume_unique=True, invert=True)]
                    if len(prefetch_pruned) == 0:
                        batched_prefetch[table_id].append(torch.tensor([], dtype=torch.long))
                    else:
                        batched_prefetch[table_id].append(torch.from_numpy(prefetch_pruned))
                batch_counter += 1
            for table_id in range(self.ln_emb_len):
                batched_prefetch[table_id] = torch.cat(batched_prefetch[table_id])
                
            num_times_rum += batch_counter
            
            fut = rpc.rpc_async(
                self.emb_worker,
                self.get_embedding,
                args=(batched_prefetch,),
            )
            
            fetched_list = fut.wait()
            last_use_location = 0
            empty_location = self.local_cache_status.eq(-1).nonzero(as_tuple=True)[0]
            for table_id, emb_vals in enumerate(fetched_list):
                if len(emb_vals) > 0:
                    emb_vals_length = len(emb_vals)
                    emb_vals = emb_vals.to(self.device)
                    indexing_offset = torch.arange(emb_vals_length, device=self.device)

                    original_idx = batched_prefetch[table_id].to(self.device)

                    empty_location_to_use = empty_location[
                        last_use_location : last_use_location
                        + emb_vals_length
                    ]
                    # changing indexing avoid running lookups agains and again
                    last_use_location += emb_vals_length
                    self.local_cache_status[empty_location_to_use] = 1

                    self.local_to_global_mapping[
                        empty_location_to_use, 0
                    ] = table_id

                    self.local_to_global_mapping[
                        empty_location_to_use, 1
                    ] = original_idx

                    self.cache_idx[table_id][
                        original_idx
                    ] = empty_location_to_use

                    self.local_cache_ttl[empty_location_to_use] = last_ttl
                    
                    with torch.no_grad():
                        self.local_cache(empty_location_to_use, indexing_offset).data = emb_vals
                        
            for _ in range(batch_counter):
                self.prefetch_completed_signal.put(last_ttl)
        
        _, self.current_example = BagCache.train_queue.get(block=True)
        self.prefetch_completed_signal.get(block=True)
        self.local_mapper(self.current_example["sync_now"], self.current_example["sync_later"])
        
        
    def launch_cache_cleanup(self):
        """
        Launch cache cleanup
        """
        while True:
            (
                iter_to_cleanup,
                ttl_update_idx,
                ttl_update_val,
            ) = self.delete_element_queue.get(block=True)
            self.clean_up_caches(iter_to_cleanup, ttl_update_idx, ttl_update_val)

    def exit_worker(self):
        rpc.shutdown()
        return 1
