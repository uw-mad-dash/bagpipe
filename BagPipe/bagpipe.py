import time

from .bagcache import BagCache
from .embserver import emb_server, exit_emb_server
from .oraclecacher import oracle_cacher

# Init


def init_cache(lookahead_value=200, emb_size=1, ln_emb=[], cache_size=25000, ragged=False,
               cleanup_batch_proportion=0.25, emb_optim=None, emb_optim_params=None, device="cuda:0",
               worker_id=0, world_size=None, training_worker_id=0,
               trainer_world_size=None, logger=None):
    return BagCache(lookahead_value=lookahead_value,
                    emb_size=emb_size,
                    ln_emb=ln_emb,
                    cache_size=cache_size,
                    ragged=ragged,
                    cleanup_batch_proportion=cleanup_batch_proportion,
                    emb_optim=emb_optim,
                    emb_optim_params=emb_optim_params,
                    device=device,
                    worker_id=worker_id,
                    world_size=world_size,
                    training_worker_id=training_worker_id,
                    trainer_world_size=trainer_world_size,
                    logger=logger,
                    cache_eviction_update=cache_eviction_update,
                    get_embedding=get_embedding)


def init_emb_server(args):
    global embedding_object
    embedding_object = emb_server(args, fetch_signal=fetch_signal)
    exit_emb_server()


def init_oracle_cacher(args):
    oracle_cacher(args, update_queue=update_queue)


# RPC Functions for BagCache


def update_queue(input_dict):
    BagCache.prefetch_queue.put((input_dict["batch_number"], input_dict["prefetch"], input_dict["train"]["train_data"]["sparse_vector"]))
    BagCache.train_queue.put((input_dict["batch_number"], input_dict["train"]))
    return 1


def fetch_signal():
    BagCache.fetchable.put(1)


# RPC Functions for Emb Server


def get_embedding(input_list):
    """
    These are prefetch embeddings
    Args:
        input_list list(tuples): tuples(table_id, emb_id)
    """
    fetched_time = time.time()
    fetched_embeddings = embedding_object.get_embeddings(input_list)
    fetched_time_end = time.time()
    print("Fetch Time {}".format(fetched_time_end - fetched_time))
    return fetched_embeddings


def cache_eviction_update(dict_to_update, emb_to_update):
    """
    update_dict- key - (table_id, emb_id): tensor to store
    """
    print("Cache eviction")
    embedding_object.update_embeddings(dict_to_update, emb_to_update)
    return 1
