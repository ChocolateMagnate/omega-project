from os import environ

# The common size of vectors in the model. This hyperparameter is called "hidden" because virtually
# everything inside model uses it spanning from token embeddings to attention activations and hidden states.
# Generally speaking, research shows that larger hidden size and shallow networks outperform those that have
# smaller hidden size but deeper networks. Increasing hidden size pays GPU memory penalty but makes linear
# attention approximation more accurate and allows the model to handle longer context windows. Up-to-date
# the largest hidden size before Omega was set by Google PaLM model that used 18K hidden size.
HIDDEN_SIZE = int(environ.get("OMEGA_HIDDEN_SIZE", 20000))

# Intermediate size used to create embeddings for tokens in a factorized embedding layer. Since it would
# take too many parameters to store [vocabulary_size, hidden_size] matrix, we use a simple model that starts with
# bottleneck size embeddings that are expanded into hidden size that significantly saves us memory and parameters.
BOTTLENECK_SIZE = int(environ.get("OMEGA_BOTTLENECK_SIZE", 1000))

# Determines how many highway layers would compose the linear attention approximation (also known as the φ function)
# in the attention block. Research has shown that wider networks perform better than deeper, therefore it is a
# worthwhile tradeoff to consider more shallow depth since hidden size, by default, already captures many nuances.
HIGHWAY_DEPTH = int(environ.get("OMEGA_HIGHWAY_DEPTH", 2))