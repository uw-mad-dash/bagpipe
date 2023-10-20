import sys
import argparse


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


def get_first_layer_size_top_mlp(interaction_type, self_interaction, ln_bot, ln_emb):
    """
    Return first layer size of top MLP
    Args:
        interaction_type(str) : Interaction between operators
        self_interaction(boolean): Are features interecting among themselves
        ln_bot (np.array): Array of structure of bottom mlp
        ln_emb (list): The list of number of rows in
    Returns:
        int : Size of first layer in top mlp
    """
    # mostly copied from line around 1134 in original dlrm code
    num_fea = len(ln_emb) + 1
    m_den_out = ln_bot[ln_bot.size - 1]
    if interaction_type == "dot":
        if self_interaction:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif interaction_type == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit("Architecture interaction type not supported")
    return num_int
