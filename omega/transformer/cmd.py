from os import environ

# The path to the tokenizer model. It should always be specified and if the file exists at this
# location, the model will be loaded and used, and otherwise it will be trained and stored there.
OMEGA_TOKENIZER_PATH = environ.get("OMEGA_TOKENIZER_PATH")

# The dataset to use to train the tokenizer. It should be representative of the kind of input the model
# will be trained on, such as multiple natural and programming languages, and hosted on HuggingFace for
# convenience. Multiple datasets can be enumerated with colon (:). The default dataset configuration is
# the mix of WikiText2 and CodeSearchNet that gives model strong English and programming tokenization.
# The number of chunks to train on is configured with OMEGA_HUGGINGFACE_CHUNKS.
OMEGA_HUGGINGFACE_DATASETS = environ.get("OMEGA_HUGGINGFACE_DATASET", "mindchain/wikitext2:code-search-net/code_search_net")

# The number of chunks to use to train the model from the OMEGA_HUGGINGFACE_DATASET dataset.
OMEGA_HUGGINGFACE_CHUNKS = int(environ.get("OMEGA_HUGGINGFACE_CHUNKS", 100000))

# The corpus to train the sentencepiece tokenizer on. If it is specified and no file exists at
#VOMEGA_TOKENIZER_PATH, it will train the tokenizer on said corpus.
OMEGA_TOKENIZER_DATA = environ.get("OMEGA_TOKENIZER_DATA")

# The algorithm to use for identifying tokens. The default in most LLMs is byte-pair encoding (BPE)
# that is great for languages with simple morphology (like English and French), and the other one
# is unigram that is more complex but suits better languages with complex writing systems like
# Japanese or Korean and it better handles unseen languages. It's recommended to use BPE for most
# applications and switch to unigram for 40B multilingual variant.
OMEGA_TOKENIZER_MODEL_TYPE = environ.get("OMEGA_TOKENIZER_MODEL_TYPE", "bpe")

# The number of different tokens to permit the tokenizer to learn. This number should be large enough to
# capture all tokens within the languages and scope intended by the model. Some common sizes are:
# GPT models use around 50K
# BERT uses 30K
# T5 uses 32K
OMEGA_TOKENIZER_VOCABULARY_SIZE = int(environ.get("OMEGA_TOKENIZER_VOCABULARY_SIZE", 40000))
