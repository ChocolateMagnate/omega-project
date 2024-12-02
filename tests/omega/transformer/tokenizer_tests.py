import omega.transformer.tokenizer as tk
from omega.transformer.tokenizer import OmegaTokenizer

def test_tokenizer_instantiation():
    tokenizer = OmegaTokenizer()

def test_tokenizer_consistency():
    tokenizer = OmegaTokenizer()
    input_string = "Recently, pytest has added a new core plugin that supports sys.path modifications via the pythonpath configuration value. The solution is thus much simpler now and doesn't require any workarounds anymore:"
    encoded = tokenizer.encode(input_string)
    decoded = tokenizer.decode(encoded)
    assert decoded == input_string

def test_tokenizer_bos_token():
    tokenizer = OmegaTokenizer()
    encoded = tokenizer.encode("Hello? Anybody home? Well you won't up then...")
    first_token = encoded[0]
    assert first_token == tk.BOS_TOKEN_ID

def test_always_bos_token():
    tokenizer = OmegaTokenizer()
    encoded = tokenizer.encode("")
    first_token = encoded[0]
    assert first_token == tk.BOS_TOKEN_ID

def test_tokenizer_no_eos_token():
    tokenizer = OmegaTokenizer()
    encoded = tokenizer.encode("Hey, come on!")
    last_token = encoded[-1]
    assert last_token != tk.EOS_TOKEN_ID
