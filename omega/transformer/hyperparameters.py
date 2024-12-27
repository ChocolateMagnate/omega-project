from os import environ

# The common size of vectors in the model. This hyperparameter is called "hidden" because virtually
# everything inside model uses it spanning from token embeddings to attention activations and hidden states.
# Generally speaking, research shows that larger hidden size and shallow networks outperform those that have
# smaller hidden size but deeper networks. Increasing hidden size pays GPU memory penalty but makes linear
# attention approximation more accurate and allows the model to handle longer context windows. Up-to-date
# the largest hidden size was set by Google PaLM model that used 18K hidden size.
HIDDEN_SIZE = int(environ.get("OMEGA_HIDDEN_SIZE", 2048))

# Represents the size of blocks images are broken down into and processed as tokens. This hyperparameter takes
# its roots in vision transformers where AI is able to see and process images by breaking it into small blocks
# called patches, where each patch is transformed into a hidden state vector that can be conceptually treated
# as a token and be attended to. Patch size 16 was tested to be the most reliable and trustworthy.
PATCH_SIZE = 16

# Tile size determines how many tokens the key, query and value matrices will be divided into and stored in GPU SRAM.
TILE_SIZE = int(environ.get("OMEGA_TILE_SIZE", 256))

# The size of vectors in the thought vector space. Larger thought sizes allow for more detailed and diverse thought
# representations at the cost of significantly more parameters.
THOUGHT_SIZE = int(environ.get("OMEGA_THOUGHT_SIZE", 50000))

THOUGHT_RANK_SIZE = int(environ.get("OMEGA_THOUGHT_RANK_SIZE", 64))

TOPIC_CLUSTER_SIZE = int(environ.get("OMEGA_THOUGHT_CLUSTER_SIZE", 64))

HORIZON = int(environ.get("OMEGA_HORIZON"), 5)

NUMBER_OF_EXPERTS = int(environ.get("OMEGA_NUMBER_OF_EXPERTS", 200))

