# this is where we hold the embedding parameter
import os
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc

class embeddingServer(object):
    def __init__(self, emb_size, ln_emb, worker_id, emb_prefix, trainer_size, fetch_signal, logger):
        print("Called init")
        self.emb_l = self.create_emb(emb_size, ln_emb, weighted_pooling=False)
        self.worker_name = f"{emb_prefix}"
        self.trainer_size = trainer_size
        self.accessed_updated = 0
        self.fetch_signal = fetch_signal
        self.trainer_prefix = "worker"
        self.logger = logger

    def create_emb(self, m, ln, weighted_pooling=None):
        """
        Create embedding tables. I am still confilicted whether to use
        dictionary or embeddingbag. For now I am going with embedding bag but
        can modify later.

        Args:
            m(int) : Number of sparse features
            ln(list(int)): The size of embedding bag to use for training
        Returns:
            Module list of embedding
        """
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in ln:
            # print(i)
            n = i
            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            # approach 1
            EE.weight.data = torch.tensor(W, requires_grad=False)
            EE.requires_grad = False
            emb_l.append(EE)
        return emb_l
    
    def get_embeddings(self, emb_decompressed):
        """
        Fetch embeddings from the actual drinks
        """
        embedding_fetched = list()
        with torch.no_grad():
            for table_id, indexes_to_fetch in enumerate(emb_decompressed):
                if len(indexes_to_fetch) > 0:
                    offset_val = torch.arange(len(indexes_to_fetch))
                    emb_table = self.emb_l[table_id]
                    embedding_val = emb_table(
                        indexes_to_fetch, torch.tensor(offset_val)
                    )
                    embedding_fetched.append(embedding_val)
                else:
                    embedding_fetched.append([])

        return embedding_fetched

    def update_embeddings(self, dict_to_update, emb_to_update):
        """
        Update evicted embeddings
        """
        self.accessed_updated += 1
        with torch.no_grad():
            for table_id in dict_to_update:
                emb_table = self.emb_l[table_id]
                emb_table.weight.data[dict_to_update[table_id]] = emb_to_update[
                    table_id
                ]
        self.logger.info("Cache eviction")
        if self.accessed_updated == self.trainer_size:
            self.accessed_updated = 0
            for tid in range(self.trainer_size):
                rpc.rpc_async(
                    f"{self.trainer_prefix}_{tid}",
                    self.fetch_signal
                )
        return None


def get_emb_length(in_file):
    with open(in_file, "r") as fin:
        data = fin.readlines()

    data = [int(d) for d in data]
    return data


def emb_server(args, fetch_signal):
    os.environ["MASTER_ADDR"] = args.master_ip
    os.environ["MASTER_PORT"] = args.master_port
    if args.emb_info_file is not None:
        args.ln_emb = get_emb_length(args.emb_info_file)
        
    logging.basicConfig(
        filename="emb_server.log"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    embedding_object = embeddingServer(
        args.emb_size, args.ln_emb, args.worker_id, args.emb_prefix, args.world_size - 2, fetch_signal, logger
    )

    rpc.init_rpc(
        embedding_object.worker_name, 
        rank=args.worker_id, 
        world_size=args.world_size,
    )

    print("RPC started")
    return embedding_object
    

def exit_emb_server():
    rpc.shutdown()