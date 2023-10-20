# decides what to cache and does not stop
import os
import queue
import torch
import logging
import statistics
import numpy as np
import torch.distributed.rpc as rpc

from time import time

from .load_csv import CSVLoader


class OracleCacher(object):
    """
    Oracle Cacher
    """

    def __init__(self, args, update_queue=None):
        """
        Args:
            lookahead_value: The number of batches of the place
            worker_addresses: Worker addresses in the
        """
        self.lookahead_value = args.lookahead_value
        self.stop_iter = args.stop_iter + args.lookahead_value
        self.ragged = args.ragged
        # changing lookahead value based on increasing number of elements
        self.additive_increase_param = args.additive_increase_param
        self.multiplicative_decrease_param = args.multiplicative_decrease_param
        self.worker_address = args.worker_addresses
        self.cache_size = args.cache_size

        self.train_ld = CSVLoader(args.processed_csv, args.mini_batch_size, len(args.ln_emb), self.stop_iter, args.dataset_multi_num, args.ragged)
        print("Data Loaded")

        self.ln_emb = args.ln_emb
        self.ln_emb_len = len(args.ln_emb)
        self.verbose = args.verbose
        if self.verbose:
            logging.basicConfig(filename=args.log_file_path)
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)
            self.logger.info(args)

        assert (
            args.mini_batch_size % args.world_size_trainers == 0
        ), "Batch size should be multiple of number of trainers"
        self.batch_size_on_each_machine = int(
            args.mini_batch_size / args.world_size_trainers
        )

        self.cleanup_interval = max(1, int(args.cleanup_proportion * self.lookahead_value))        

        self.update_queue = update_queue

        self.batch_queue_dense = list()
        self.batch_queue_sparse = list()
        self.batch_queue_target = list()
        self.batch_number_queue = list()
        self.batch_emb_union = list()
        self.batch_sync_now = list()
        self.batch_sync_later = list()
        self.batch_once = list()
        
        self.local_cache_ttl = {
            k: np.ones(idx_vals, dtype=np.int64)
            for k, idx_vals in enumerate(self.ln_emb)
        }
        self.prefetch_cache = dict()

        # indicates the last time we have seen element
        self.latest_tracker = {
            k: np.ones(idx_vals, dtype=np.int64)
            for k, idx_vals in enumerate(self.ln_emb)
        }
        # setting values -1 as needed
        for k in self.local_cache_ttl:
            self.local_cache_ttl[k][:] = -1

        for k in self.latest_tracker:
            self.latest_tracker[k][:] = -1
            
        # sync_now & sync later
        self.current_example = None
        self.next_example = None
        
        # counters for metric collection
        self.worker_id = args.worker_id
        self.worker_name = f"{args.oracle_prefix}"
        self.num_trainers = args.world_size_trainers
        self.trainer_prefix = args.trainer_prefix
        self.world_size = args.world_size
        
        os.environ["MASTER_ADDR"] = args.master_ip
        os.environ["MASTER_PORT"] = args.master_port
        self.cleanup_interval = max(
            1, int(args.cleanup_proportion * self.lookahead_value)
        )
        
        return None

    def train_addition(self):
        """
        Initialize the data partitioning process.
        """
        # rpc setup
        rpc.init_rpc(self.worker_name, 
                     rank=self.worker_id, 
                     world_size=self.world_size,
                    )

        iteration_recorder = list()
        batch_counter = 0
        current_batch_number = 0
        self.current_example = next(self.train_ld)
        while True:
            # fetching batches from train queue
            iteration_start = time()
            try:
                while len(self.batch_queue_dense) < self.lookahead_value:
                    batch_fetch_time = time()
                    self.next_example = next(self.train_ld)
                    batch_fetch_time_end = time()
                    print("Batch time taken fetch {}ms".format((batch_fetch_time_end - batch_fetch_time) * 1000))
                    X, lS_i, lS_i_u, T = self.current_example
                    
                    # For sync_now & sync_later
                    _, _, lS_in_u, _ = self.next_example
                    
                    self.batch_queue_dense.append(torch.from_numpy(X))
                    if self.ragged:
                        self.batch_queue_sparse.append(lS_i)
                    else:
                        self.batch_queue_sparse.append(torch.from_numpy(lS_i))
                    self.batch_queue_target.append(torch.from_numpy(T))
                    self.batch_number_queue.append(batch_counter)
                    # emb_union = set()  # holds unique embeddings in a batch
                    # can be mobed to multiprocess
                    emb_union = dict()
                    sync_now = list()
                    sync_later = list()
                    once = list()
                    start_time_union = time()
                    for table_id in range(self.ln_emb_len):
                        emb_ids, emb_ids_once = lS_i_u[table_id]
                        emb_ids_next, _ = lS_in_u[table_id]
                        emb_union[table_id] = emb_ids
                        self.latest_tracker[table_id][emb_ids] = batch_counter
                        
                        # Do intersect & xor
                        current_in_next = np.in1d(emb_ids, emb_ids_next, assume_unique=True)
                        emb_ids_sync_now = emb_ids[current_in_next]
                        emb_ids_sync_later = emb_ids[~current_in_next]
                        
                        sync_now.append(torch.from_numpy(emb_ids_sync_now))
                        sync_later.append(emb_ids_sync_later)
                        once.append(emb_ids_once)
                    self.batch_emb_union.append(emb_union)
                    self.batch_sync_now.append(sync_now)
                    self.batch_sync_later.append(sync_later)
                    self.batch_once.append(once)
                    batch_counter += 1
                    end_time_union = time()
                    self.current_example = self.next_example
                    print("Total time for emb union {}ms".format((end_time_union - start_time_union) * 1000))
            except StopIteration:
                    break
            # self.batch_queue will have all the training data
            # our latest tracker seems to be working
            # fetch the current batch for analysis
            current_batch_dense = self.batch_queue_dense.pop(0)
            current_batch_sparse = self.batch_queue_sparse.pop(0)
            current_batch_target = self.batch_queue_target.pop(0)
            current_batch_number = self.batch_number_queue.pop(0)
            emb_union_current_batch = self.batch_emb_union.pop(0)
            sync_now_current_batch = self.batch_sync_now.pop(0)
            sync_later_current_batch = self.batch_sync_later.pop(0)
            once_current_batch = self.batch_once.pop(0)

            if self.verbose:
                self.logger.info("Current Batch Number {}".format(current_batch_number))
                
            prefetch_list = list()
            ttl_idx = dict()
            ttl_val = dict()
            no_sync = dict()
            
            # decide elements to prefetch
            for table_id in emb_union_current_batch:
                potential_embs_to_fetch = emb_union_current_batch[table_id]
                cache_indicator = self.local_cache_ttl[table_id][potential_embs_to_fetch]
                embs_to_fetch = potential_embs_to_fetch[cache_indicator.__lt__(current_batch_number)]
                prefetch_list.append(embs_to_fetch)
                
                unique_embs = emb_union_current_batch[table_id]
                check_subset = self.latest_tracker[table_id][unique_embs]
                embs_to_cache = np.logical_and(check_subset.__le__(current_batch_number + self.lookahead_value), check_subset.__gt__(current_batch_number))
                embs_ids_to_cache = unique_embs[embs_to_cache]
                updated_ttls = self.latest_tracker[table_id][embs_ids_to_cache]
                self.local_cache_ttl[table_id][embs_ids_to_cache] = updated_ttls
                check_once = self.latest_tracker[table_id][once_current_batch[table_id]]
                no_sync[table_id] = once_current_batch[table_id][check_once.__eq__(current_batch_number)]
                sync_later_unpruned = sync_later_current_batch[table_id]
                sync_later_current_batch[table_id] = torch.from_numpy(sync_later_unpruned[np.in1d(sync_later_unpruned, no_sync[table_id], assume_unique=True, invert=True)])
                
                ttl_idx[table_id] = torch.from_numpy(embs_ids_to_cache)
                ttl_val[table_id] = torch.from_numpy(updated_ttls)
                    
            for tid in range(self.num_trainers):
                # partition training dict
                prefetch_train = dict()
                prefetch_train["batch_number"] = current_batch_number
                prefetch_train["prefetch"] = {"prefetch_list": prefetch_list, "no_sync": no_sync}
                prefetch_train["train"] = {
                    "ttl_idx": ttl_idx,
                    "ttl_val": ttl_val,
                    "sync_now": sync_now_current_batch,
                    "sync_later": sync_later_current_batch,
                    "train_data": {
                        "dense_x": current_batch_dense[
                            tid
                            * self.batch_size_on_each_machine : (tid + 1)
                            * self.batch_size_on_each_machine,
                            :,
                        ],
                        "sparse_vector": [
                            torch.from_numpy(current_batch_sparse[i][tid * self.batch_size_on_each_machine : (tid + 1) * self.batch_size_on_each_machine]) for i in range(self.ln_emb_len)
                        ] if self.ragged else current_batch_sparse[
                            :,
                            tid
                            * self.batch_size_on_each_machine : (tid + 1)
                            * self.batch_size_on_each_machine,
                        ],
                        "target": current_batch_target[
                            tid
                            * self.batch_size_on_each_machine : (tid + 1)
                            * self.batch_size_on_each_machine
                        ],
                    },
                }
                # sending training data
                rpc.rpc_async(
                    f"{self.trainer_prefix}_{tid}",
                    self.update_queue,
                    args=(prefetch_train,),
                )
                
            iteration_end = time()
            iteration_recorder.append((iteration_end - iteration_start) * 1000)
        if self.verbose:
            self.logger.info(f"Iteration time median {statistics.median(iteration_recorder)}")


def oracle_cacher(args, update_queue):
    cacher_instance = OracleCacher(args, update_queue)
    cacher_instance.train_addition()
    print("Finished Sending")
    rpc.shutdown()


def exit_oracle_cacher(input_dict):
    rpc.shutdown()
    return 1