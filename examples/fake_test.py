from BagPipe import init_cache

bp = init_cache(
    lookahead_value=args.lookahead_value,
    emb_size=args.emb_size,
    ln_emb=args.ln_emb,
    cache_size=args.cache_size,
    device=args.device,
    trainer_world_size=args.world_size_trainers,
    cleanup_batch_proportion=args.cleanup_batch_proportion,
    worker_id=args.worker_id,
    world_size=args.world_size,
    training_worker_id=args.dist_worker_id,
    emb_optim=optim.SGD,
    emb_optim_params={"lr": 0.01},
    rpc_num_worker_threads=mp.cpu_count() - 2,
    logger=logger,
)

next_train_example = bp.next_batch()
next_embs = bp.get_emb(next_train_example["sparse"])
