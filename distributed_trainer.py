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

import utils
from operator import itemgetter


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
        device="cuda:0",
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
        # this will hold the cache
        self.local_cache = nn.ParameterDict({})
        self.local_cache_ttl = dict()
        # this will hold the prefetch values
        self.prefetch_cache = nn.ParameterDict({})
        self.prefetch_cache_ttl = dict()

        self.relevant_local_cache = nn.ParameterDict({})
        self.relevant_prefetch_cache = nn.ParameterDict({})

        self.train_queue = queue.Queue()
        self.prefetch_queue = queue.Queue()
        self.prefetch_futures_queue = queue.Queue()
        self.prefetch_queue_ttl = queue.Queue()

        self.delete_element_queue = queue.Queue()

        self.worker_id = worker_id
        self.worker_name = f"worker_{worker_id}"
        self.current_train_epoch = 0
        return None

    def deepcopy_relevant_caches(self, prefetch_cache_elements, local_cache_elements):
        """
        Make copy of relevant params, helps in optimizer and synchronization.
        We are not making a copy for now, lets see how this works.
        Args:
            prefetch_cache_elements(list(table_id, emb_id))): Elements to get from
            prefetch cache
            local_cache_elements(list(table_id, emb_id)): Elements to get from
            local cache
        """
        relevant_prefetch_cache = dict()
        relevant_local_cache = dict()
        with torch.no_grad():
            for elem in prefetch_cache_elements:
                elem = elem.__str__()
                relevant_prefetch_cache[elem] = self.prefetch_cache[elem]
            for elem in local_cache_elements:
                elem = elem.__str__()
                relevant_local_cache[elem] = self.local_cache[elem]
            self.relevant_local_cache = nn.ParameterDict({})
            self.relevant_prefetch_cache = nn.ParameterDict({})

            self.relevant_local_cache.update(relevant_local_cache)
            self.relevant_prefetch_cache.update(relevant_prefetch_cache)
        return None

    def clean_up_caches(self, iter_cleanup):
        """
        Based on stored TTLs evict from the caches
        """
        try:
            # print("Clean up cache called {}".format(iter_cleanup))
            dict_to_update = dict()
            elems_to_eliminate_prefetch_cache = list()
            elems_to_eliminate_local_cache = list()
            prefetch_cache_ttl_keys = list(self.prefetch_cache_ttl.keys())
            for elem in prefetch_cache_ttl_keys:
                if self.prefetch_cache_ttl[elem] <= iter_cleanup:
                    elems_to_eliminate_prefetch_cache.append(elem)
            # logger.info(
            # f"Elements to Remove Prefetch Cache {elems_to_eliminate_prefetch_cache}"
            # )
            for del_object in elems_to_eliminate_prefetch_cache:
                dict_to_update[del_object] = self.prefetch_cache[
                    del_object.__str__()
                ].cpu()
                del self.prefetch_cache[del_object.__str__()]
                del self.prefetch_cache_ttl[del_object]

            local_cache_ttl_keys = list(self.local_cache_ttl.keys())
            for elem in local_cache_ttl_keys:
                if self.local_cache_ttl[elem] < iter_cleanup:
                    elems_to_eliminate_local_cache.append(elem)

            # logger.info(f"Elements to Remove Local Cache {elems_to_eliminate_local_cache}")
            for del_object in elems_to_eliminate_local_cache:
                dict_to_update[del_object] = self.local_cache[
                    del_object.__str__()
                ].cpu()
                del self.local_cache[del_object.__str__()]
                del self.local_cache_ttl[del_object]
            # print("Len local cache {}".format(len(self.local_cache)))
            # print("Len prefetch cache {}".format(len(self.prefetch_cache)))
            rpc.rpc_async("worker_2", cache_eviction_update, args=((dict_to_update,)))

            # launching prefetch requests as well

            while True:
                # only fetch when needed
                if not self.prefetch_queue.empty():
                    # if prefetech queue is not empty
                    # print(self.prefetch_queue.queue[0].keys())
                    if (
                        list(self.prefetch_queue.queue[0].keys())[0]
                        <= iter_cleanup + self.lookahead_value
                    ):
                        val = self.prefetch_queue.get(block=True)
                        fut = rpc.rpc_async("worker_2", get_embedding, args=(val,))
                        ttl_val = list(val.keys())[0]
                        # print("Launched prefetch {}".format(ttl_val))
                        self.prefetch_futures_queue.put(fut)
                        self.prefetch_queue_ttl.put(ttl_val)
                    else:
                        break
                else:
                    break

            # adding to the cache post eviction
            # while (
            # self.prefetch_queue_ttl.queue[0] <= iter_cleanup + self.lookahead_value
            # ):
            while True:
                if not self.prefetch_queue_ttl.empty():

                    if (
                        self.prefetch_queue_ttl.queue[0]
                        <= iter_cleanup + self.lookahead_value
                    ):

                        fut = self.prefetch_futures_queue.get(block=True)
                        ttl_val = self.prefetch_queue_ttl.get(block=True)
                        val = fut.wait()
                        val_str = {
                            k.__str__(): nn.Parameter(val[k].to(self.device))
                            for k in val
                        }
                        val_ttl = {k: ttl_val for k in val}
                        # print("Added prefetch cache {}".format(ttl_val))
                        # print(val_str)
                        self.prefetch_cache.update(val_str)
                        self.prefetch_cache_ttl.update(val_ttl)
                    else:
                        # pulled enough elements
                        break
                else:
                    # no elements in the queue
                    break

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
        return mlp_network(dense_x)

    def apply_emb(self, lS_i):
        """
        Fetch embedding
        """
        fetched_embeddings = list()
        for table_id, emb_id in enumerate(lS_i):
            # this outer for loop can be parallelized
            # we will optimize this for loop some other day
            # this has branching and if checks, I really don't like it
            # but I am commited to this at this point. So not doing anything
            # about this
            emb_by_id = list()
            emb_found = False
            for embs in emb_id:
                lookup_id = (table_id, embs.item()).__str__()
                # first look it up in the cache
                try:
                    emb_fetched = self.relevant_local_cache.get_parameter(lookup_id)
                    emb_found = True
                except AttributeError:
                    # if embedding is not found
                    emb_found = False

                if not emb_found:
                    try:
                        emb_fetched = self.relevant_prefetch_cache.get_parameter(
                            lookup_id
                        )
                        emb_found = True
                    except AttributeError:
                        # if embedding is not found
                        emb_found = False

                # if not found look it up the prefetech
                if not emb_found:
                    # element not found
                    logger.info(f"Lookup ID not found {lookup_id}")
                    sys.exit("Embedding not found in prefetch nor in local cache")
                emb_by_id.append(emb_fetched)
            concatenated_emb = torch.cat(emb_by_id)
            concatenated_emb = concatenated_emb.reshape(len(emb_id), -1)
            fetched_embeddings.append(concatenated_emb)
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

        x = self.apply_mlp(dense_x, self.bot_mlp)

        # need to fetch the embeddings
        # at this point we will either have embeddings in the local cache or
        # global cache

        # TODO: In future include more complicated processing
        ly = self.apply_emb(lS_i)
        # print(x)
        # print(ly)
        # feature interaction
        z = self.interact_features(x, ly)

        # pass through top mlp
        p = self.apply_mlp(z, self.top_mlp)
        loss = self.loss_fn(p, target)
        # print(loss)
        return loss


def update_train_queue(input_dict):
    comp_intensive_model.train_queue.put(input_dict)
    return 1


def fill_prefetch_cache():
    num_times_run = 0
    try:
        while num_times_run <= comp_intensive_model.lookahead_value:
            val = comp_intensive_model.prefetch_queue.get(block=True)
            fut = rpc.rpc_async("worker_2", get_embedding, args=(val,))
            ttl_val = list(val.keys())[0]
            comp_intensive_model.prefetch_futures_queue.put(fut)
            comp_intensive_model.prefetch_queue_ttl.put(ttl_val)
            # keep getting prefetch queue
            fut = comp_intensive_model.prefetch_futures_queue.get(block=True)
            ttl_val = comp_intensive_model.prefetch_queue_ttl.get(block=True)
            val = fut.wait()
            val_str = {
                k.__str__(): nn.Parameter(val[k].to(comp_intensive_model.device))
                for k in val
            }
            val_ttl = {k: ttl_val for k in val}
            print("Added prefetch cache {}".format(ttl_val))
            comp_intensive_model.prefetch_cache.update(val_str)
            comp_intensive_model.prefetch_cache_ttl.update(val_ttl)
            num_times_run += 1
    except queue.Empty:
        pass


def update_prefetch_queue(input_dict):
    comp_intensive_model.prefetch_queue.put(input_dict, block=True)
    size_of_queue = comp_intensive_model.prefetch_queue.qsize()
    # print("Prefetch queue size at insertion {}".format(size_of_queue))
    return 1


def launch_cache_cleanup():
    """
    Launch cache cleanup
    """
    print("Cache cleanup launched")
    while True:
        try:
            iter_to_cleanup = comp_intensive_model.delete_element_queue.get(block=True)
            # print("iter to cleanup {}".format(iter_to_cleanup))
            comp_intensive_model.clean_up_caches(iter_to_cleanup)
        except queue.Empty:
            pass


def launch_cache_cleanup_no_thread():
    """
    Launch cache cleanup
    """
    # print("Cache cleanup launched")
    # while True:
    try:
        iter_to_cleanup = comp_intensive_model.delete_element_queue.get(block=True)
        # print("iter to cleanup {}".format(iter_to_cleanup))
        comp_intensive_model.clean_up_caches(iter_to_cleanup)
    except queue.Empty:
        pass


def exit_worker(input_dict):
    rpc.shutdown()
    return 1


def cache_eviction_update(update_dict):
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
        device=args.device,
    )

    # rpc fuctions

    # rpc setup
    rpc.init_rpc(
        comp_intensive_model.worker_name,
        rank=args.worker_id,
        world_size=args.world_size,
    )

    # thread for waiting on prefetching

    # prefetch_update_thread = threading.Thread(target=update_prefetch_cache)
    # prefetch_update_thread.start()

    cleanup_and_update_thread = threading.Thread(target=launch_cache_cleanup)
    cleanup_and_update_thread.start()

    # train loop

    # this will fill the prefetch cache
    fill_prefetch_cache()
    while True:
        try:
            total_start = time.time()
            train_example = comp_intensive_model.train_queue.get(block=True)
            current_epoch = list(train_example.keys())[0]
            # handling rpc potential reordering
            if current_epoch != expected_iter:
                # move current train example to the dictionary
                iter_overflow[current_epoch] = copy.deepcopy(train_example)
                # check if we have the expected iter in the overflow
                if expected_iter in iter_overflow:
                    train_example = iter_overflow.pop(expected_iter)
                    current_epoch = list(train_example.keys())[0]
                    expected_iter += 1
                else:
                    # pop more and see if we find what we want
                    continue
            else:
                expected_iter += 1

            print("Current Iter {}".format(current_epoch))
            # logger.info(f"Current Iter {current_epoch}")
            comp_intensive_model.current_train_epoch = current_epoch

            # logger.info(f"Size local cache {len(comp_intensive_model.local_cache)}")
            # logger.info(
            # f"Size prefetch cache {len(comp_intensive_model.prefetch_cache)}"
            # )
            # logger.info(
            # f"State of Prefetch Cache {comp_intensive_model.prefetch_cache}"
            # )
            # TODO: I think this should deeply simplify a lot of things
            # logger.info(
            # "Elements from Prefetch Cache {}".format(
            # train_example[current_epoch]["train_data"][
            # "elements_from_prefetch"
            # ]
            # )
            # )

            # logger.info(
            # "Elements from Local Cache {}".format(
            # train_example[current_epoch]["train_data"][
            # "elements_from_cache"
            # ]
            # )
            # )

            # logger.info(
            # "Prefetch Cache {}".format(comp_intensive_model.prefetch_cache.keys())
            # )
            # )
            # logger.info(
            # "Prefetch Cache TTL {}".format(
            # comp_intensive_model.prefetch_cache_ttl
            # )
            # )
            # logger.info(
            # "Local Cache {}".format(comp_intensive_model.local_cache.keys())
            # )

            # logger.info(
            # "Local Cache TTL {}".format(comp_intensive_model.local_cache_ttl)
            # )
            copy_rlevant = time.time()
            comp_intensive_model.deepcopy_relevant_caches(
                train_example[current_epoch]["train_data"]["elements_from_prefetch"],
                train_example[current_epoch]["train_data"]["elements_from_cache"],
            )

            deepcopy_relevant_end = time.time()

            print(
                "Relevant cache copy time {}".format(
                    deepcopy_relevant_end - copy_rlevant
                )
            )

            forward_start = time.time()

            loss = comp_intensive_model.forward(
                train_example[current_epoch]["train_data"]["dense_x"].to(
                    comp_intensive_model.device
                ),
                train_example[current_epoch]["train_data"]["sparse_vector"],
                train_example[current_epoch]["train_data"]["target"].to(
                    comp_intensive_model.device
                ),
            )

            forward_stop = time.time()
            print("Time for forward {}".format(forward_stop - forward_start))

            optimizer_construct = time.time()
            optimizer = optim.SGD(
                [
                    {
                        "params": comp_intensive_model.relevant_local_cache.parameters(),
                        "lr": 0.01,
                    },
                    {
                        "params": comp_intensive_model.relevant_prefetch_cache.parameters(),
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

            optimizer_construct_end = time.time()
            print(
                "Time for optim construction {}".format(
                    optimizer_construct_end - optimizer_construct
                )
            )

            backward_start = time.time()

            loss.backward()

            backward_stop = time.time()
            print("Time for backward {}".format(backward_stop - backward_start))

            optim_time = time.time()
            optimizer.step()
            optimizer.zero_grad()
            optim_time_end = time.time()
            print("Optim Time end to end {}".format(optim_time_end - optim_time))
            # moving elements from prefetch cache to local cache
            elements_to_cache = train_example[current_epoch]["cache_elements"]

            data_move = time.time()
            with torch.no_grad():
                temp_dict = dict()
                for elem in elements_to_cache:
                    # moving the tensor
                    comp_intensive_model.local_cache[
                        elem[0].__str__()
                    ] = comp_intensive_model.relevant_prefetch_cache[elem[0].__str__()]
                    # moving the ttl
                    comp_intensive_model.local_cache_ttl[elem[0]] = elem[1]

            data_move_end = time.time()
            print("Data movement across cache{}".format(data_move_end - data_move))
            # TODO: Move to a thread
            # update TTL
            lease_extensions_upt = time.time()
            lease_extensions = train_example[current_epoch]["lease_extensions"]
            for elem, new_ttl in lease_extensions:
                comp_intensive_model.local_cache_ttl[elem] = new_ttl
            lease_extensions_end = time.time()
            print(
                "Lease extension time {}".format(
                    lease_extensions_end - lease_extensions_upt
                )
            )

            # evict from caches
            comp_intensive_model.delete_element_queue.put(current_epoch)
            # cleanup_and_fetch = time.time()
            # launch_cache_cleanup_no_thread()
            # cleanup_and_fetch_end = time.time()
            # print(
            # "Clean up and fetch time {}".format(
            # cleanup_and_fetch_end - cleanup_and_fetch
            # )
            # )
            del loss
            total_end = time.time()
            print("Total end to end time {}".format(total_end - total_start))
            logger.info("Total end to end time {}".format(total_end - total_start))
            # copy elements from prefetch cache to local cache
            # need to perform cache evictions

            # comp_intensive_model.zero_grad()

            # for name, params in comp_intensive_model.named_parameters():
            # print(name)
            # print(params.grad)
            # embeddings to check grad
            # print(
            # "Elements to cache {}".format(
            # train_example[current_epoch]["cache_elements"]
            # )
            # )
            # for update_emb in train_example[current_epoch]["train_data"][
            # "list_tuple_embedding"
            # ]:
            # print(update_emb)
            # print(
            # comp_intensive_model.prefetch_cache[update_emb.__str__()].grad
            # )

            # got an example to train

        except queue.Empty:
            pass


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
        default=16,
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
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-ip", type=str, default="localhost")
    parser.add_argument("--master-port", type=str, default="18000")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(description="Arguments for DLRM"))
    logging.basicConfig(filename="distributed_trainer_cache_prefetch_threaded.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main(args)
