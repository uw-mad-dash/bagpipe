# here is the high level algorithm
# we do 1% embeddings are used by 90% of samples
# we sample 1% embedding values and make sure they are repeated x% of times and rest are randomly chosen

import csv
import json
import math
import random
import numpy as np

embedding_table_sizes = [
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
]
dataset_lines = 39291958


def main(topk_perc, use_sample):
    """
    Generarate embedding
    topk_perc(int) : Percentage of embeddings used
    use_sample(int): Percentage of samples using topk_perc

    Read this as "topk_perc" embeddings used in use_sample_percentage embeddings
    1% embeddings used by 90% of samples
    """

    top_perc_embeddings = []
    for emb_table_size in embedding_table_sizes:
        # calculate number of top k% embeddings
        num_embs = math.ceil((topk_perc * emb_table_size) / 100)
        # choose those embeddings
        top_perc_embeddings.append(random.sample(range(emb_table_size), num_embs))
    # top_perc_embeddings contain the embeddings which need to be reeated topk_perc times
    open_file = open(
        f"kaggle_artifiial_top_{topk_perc}_use_{use_sample}_fraction.csv", "w"
    )
    for i in range(dataset_lines):
        # we run run  a binomial sample to choose if we have are going to have a hot entry or not
        dense_line = list()
        for emb_table_id, emb_table_size in enumerate(embedding_table_sizes):
            is_hot = np.random.binomial(1, use_sample / 100)
            if is_hot:
                emb_id = random.sample(top_perc_embeddings[emb_table_id], 1)[0]
            else:
                emb_id = random.sample(range(embedding_table_sizes[emb_table_id]), 1)[0]
            dense_line.append(emb_id)

        data_dict = {
            "label": 1.0,
            "dense": [
                2.5649492740631104,
                3.044522523880005,
                1.3862943649291992,
                1.3862943649291992,
                1.0986123085021973,
                1.3862943649291992,
                2.70805025100708,
                3.7841897010803223,
                3.8712010383605957,
                1.0986123085021973,
                1.3862943649291992,
                0.0,
                1.0986123085021973,
            ],
            "sparse": dense_line,
        }
        open_file.write(json.dumps(data_dict))
        open_file.write("\n")


if __name__ == "__main__":
    main(1, 10)
