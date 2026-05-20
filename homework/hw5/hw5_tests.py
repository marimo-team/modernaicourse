import copy
import io
import os
import tempfile
from contextlib import redirect_stdout

import mugrade
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tiktoken


def _causal_mask(length):
    return torch.triu(torch.full((length, length), float("-inf")), diagonal=1)


def _copy_mha_weights(layer, ref_layer):
    with torch.no_grad():
        ref_layer.in_proj_weight.copy_(torch.cat([layer.wq.weight, layer.wk.weight, layer.wv.weight], dim=0))
        ref_layer.out_proj.weight.copy_(layer.wp.weight)


def _ref_mlp(mlp, X):
    hidden = F.linear(X, mlp.w1.weight)
    return F.linear(F.silu(hidden), mlp.w2.weight)


def _ref_block(block, X, mask=None):
    dim = X.shape[-1]
    X_norm = F.rms_norm(X, (dim,))

    ref_attn = nn.MultiheadAttention(dim, block.attn.n_heads, bias=False, batch_first=True)
    _copy_mha_weights(block.attn, ref_attn)

    Z = X + ref_attn(X_norm, X_norm, X_norm, attn_mask=mask, need_weights=False)[0]
    return Z + _ref_mlp(block.mlp, F.rms_norm(Z, (dim,)))


def _ref_llm(model, tokens, seq_pos=0):
    X = F.embedding(tokens, model.embedding.weight)
    X = X + model.pos_embeddings[seq_pos : seq_pos + tokens.shape[1]]
    mask = model.mask[seq_pos : seq_pos + tokens.shape[1], : seq_pos + tokens.shape[1]]

    for layer in model.layers:
        X = _ref_block(layer, X, mask=mask)

    X = F.rms_norm(X, (model.output.weight.shape[1],))
    return F.linear(X, model.output.weight)


class _ChunkTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, allowed_special="all"):
        self.calls.append((text, allowed_special))
        return [ord(c) for c in text]


class _ToyGenerationTokenizer:
    def __init__(self, eot_token):
        self.eot_token = eot_token
        self._vocab = {
            3: "A",
            4: "!",
            5: "B",
        }

    def decode(self, tokens):
        if isinstance(tokens, int):
            tokens = [tokens]
        return "".join(self._vocab[t] for t in tokens)


class _ToyGenerationModel(nn.Module):
    def __init__(self, next_tokens, vocab_size=6):
        super().__init__()
        self.next_tokens = list(next_tokens)
        self.vocab_size = vocab_size
        self.calls = []
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, seq_pos=0, use_kv_cache=False):
        self.calls.append((tokens.detach().clone(), seq_pos, use_kv_cache, tokens.dtype, tokens.device.type))
        logits = torch.full((1, tokens.shape[1], self.vocab_size), -1e9, device=tokens.device)
        next_token = self.next_tokens[len(self.calls) - 1]
        logits[0, -1, next_token] = 0.0
        return logits


class _TinyTrainModel(nn.Module):
    def __init__(self, vocab_size=5, dim=4, output_dtype=torch.bfloat16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, vocab_size, bias=False)
        self.output_dtype = output_dtype

    def forward(self, tokens):
        return self.proj(self.embed(tokens)).to(self.output_dtype)


def _checked_cross_entropy(logits, y):
    assert(logits.dtype == torch.float32)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def _write_token_file(path, tokens):
    with open(path, "wb") as f:
        f.write(np.asarray(tokens, dtype=np.uint16).tobytes())


def _tiny_stories_eval_tokens(start_token=0, num_tokens=48):
    path = os.path.join(os.path.dirname(__file__), "TinyStoriesV2-GPT4-train.txt")
    tokenizer = tiktoken.get_encoding("gpt2")
    with open(path, "rt") as f:
        text = f.read(4096)
    all_tokens = tokenizer.encode(text, allowed_special="all")
    tokens = all_tokens[start_token : start_token + num_tokens]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


def _sequence_loss(model, tokens):
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    with torch.inference_mode():
        logits = model(tokens[:, :-1]).float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)).item()


def test_text_to_corpus(text_to_corpus):
    corpus, counts = text_to_corpus("a b b")
    assert(corpus == [["a"], [" ", "b"]])
    assert(counts == [1, 2])

    corpus, counts = text_to_corpus("hi there\nthere")
    assert(corpus == [["h", "i"], [" ", "t", "h", "e", "r", "e"], ["\n", "t", "h", "e", "r", "e"]])
    assert(counts == [1, 1, 1])


def submit_text_to_corpus(text_to_corpus):
    corpus, counts = text_to_corpus("x y y z")
    mugrade.submit(corpus)
    mugrade.submit(counts)

    corpus, counts = text_to_corpus("go\nstop\nstop")
    mugrade.submit(corpus)
    mugrade.submit(counts)


def test_most_common_pair(most_common_pair):
    corpus = [["a", "b", "a"], ["a", "b"], ["b", "c"]]
    counts = [2, 1, 3]
    assert(most_common_pair(corpus, counts) == ("a", "b"))

    corpus = [[" ", "x"], [" ", "x", "y"], ["x", "y"]]
    counts = [4, 1, 1]
    assert(most_common_pair(corpus, counts) == (" ", "x"))


def submit_most_common_pair(most_common_pair):
    corpus = [["a", "a"], ["a", "b", "a"], ["b", "a"]]
    counts = [1, 2, 3]
    mugrade.submit(most_common_pair(corpus, counts))

    corpus = [["x", "y", "z"], ["y", "z"], ["x", "y"]]
    counts = [1, 4, 2]
    mugrade.submit(most_common_pair(corpus, counts))


def test_merge_pair(merge_pair):
    corpus = [["a", "b", "c"], ["c", "a", "b"]]
    out = merge_pair(corpus, ("a", "b"))
    assert(out is None)
    assert(corpus == [["ab", "c"], ["c", "ab"]])

    corpus = [["x", "y", "z"], ["x", "y"], ["y", "z"]]
    merge_pair(corpus, ("y", "z"))
    assert(corpus == [["x", "yz"], ["x", "y"], ["yz"]])


def submit_merge_pair(merge_pair):
    corpus = [["a", "b", "c", "d"], ["a", "b"], ["c", "d"]]
    merge_pair(corpus, ("a", "b"))
    mugrade.submit(corpus)

    corpus = [["x", "y", "z"], ["y", "z", "x"]]
    merge_pair(corpus, ("y", "z"))
    mugrade.submit(corpus)


def test_train_bpe(train_bpe):
    tokens, merges = train_bpe("aa aa aa", 258)
    assert(isinstance(tokens, dict))
    assert(tokens["a"] == ord("a"))
    assert(tokens[" "] == ord(" "))
    assert(tokens["aa"] == 256)
    assert(tokens[" aa"] == 257)
    assert(merges == [("a", "a"), (" ", "aa")])


def submit_train_bpe(train_bpe):
    tokens, merges = train_bpe("aa aa aa", 258)
    mugrade.submit(merges)
    mugrade.submit(tokens["aa"])
    mugrade.submit(tokens[" aa"])


def test_bpe_encode(bpe_encode):
    tokens = {"a": 0, " ": 1, "aa": 2, " aa": 3}
    merges = [("a", "a"), (" ", "aa")]
    assert(bpe_encode("aa aa", merges, tokens) == [2, 3])
    assert(bpe_encode("aa", merges, tokens) == [2])


def submit_bpe_encode(bpe_encode):
    tokens = {"a": 0, " ": 1, "aa": 2, " aa": 3}
    merges = [("a", "a"), (" ", "aa")]
    mugrade.submit(bpe_encode("aa aa aa", merges, tokens))
    mugrade.submit(bpe_encode("aa", merges, tokens))


def test_bpe_decode(bpe_decode):
    tokens = {"a": 0, " ": 1, "aa": 2, " aa": 3}
    assert(bpe_decode([2, 3], tokens) == "aa aa")
    assert(bpe_decode([0, 1, 0], tokens) == "a a")


def submit_bpe_decode(bpe_decode):
    tokens = {"a": 0, " ": 1, "aa": 2, " aa": 3}
    mugrade.submit(bpe_decode([2, 3, 3], tokens))
    mugrade.submit(bpe_decode([0, 1, 0], tokens))


def test_Linear(Linear):
    torch.manual_seed(0)
    layer = Linear(10, 20)
    assert(hasattr(layer, "weight"))
    assert(isinstance(layer.weight, nn.Parameter))
    assert(layer.weight.shape == (20, 10))

    ref_layer = nn.Linear(10, 20, bias=False)
    ref_layer.weight.data = layer.weight.data.clone()

    X = torch.randn(50, 10)
    assert(layer(X).shape == (50, 20))
    assert(torch.allclose(ref_layer(X), layer(X), atol=1e-6))

    X = torch.randn(7, 9, 10)
    assert(layer(X).shape == (7, 9, 20))
    assert(torch.allclose(ref_layer(X), layer(X), atol=1e-6))

    layer = Linear(100, 1000)
    assert(torch.allclose(layer.weight.std(), torch.tensor((2 / 100) ** 0.5), atol=3e-3))


def submit_Linear(Linear):
    layer = Linear(4, 3)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, -1.0, 0.5, 2.0],
                                         [-0.5, 0.0, 1.0, -1.5],
                                         [2.0, 1.5, -0.5, 0.25]]))
    X = torch.tensor([[1.0, 2.0, -1.0, 0.5],
                      [0.0, -1.0, 2.0, 3.0]])
    mugrade.submit(layer(X).detach().numpy())
    mugrade.submit(layer(torch.stack([X, -X])).detach().numpy())
    mugrade.submit(type(layer.weight))


def test_Embedding(Embedding):
    torch.manual_seed(1)
    layer = Embedding(200, 20)
    assert(hasattr(layer, "weight"))
    assert(isinstance(layer.weight, nn.Parameter))
    assert(layer.weight.shape == (200, 20))

    ref_layer = nn.Embedding(200, 20)
    ref_layer.weight.data = layer.weight.data.clone()

    Y = torch.randint(0, 200, size=(13,))
    assert(torch.allclose(ref_layer(Y), layer(Y), atol=1e-6))

    Y = torch.randint(0, 200, size=(7, 11))
    assert(layer(Y).shape == (7, 11, 20))
    assert(torch.allclose(ref_layer(Y), layer(Y), atol=1e-6))

    layer = Embedding(1000, 100)
    assert(torch.allclose(layer.weight.std(), torch.tensor(1.0), atol=3e-2))


def submit_Embedding(Embedding):
    layer = Embedding(8, 3)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.0, 0.5, 1.0],
                                         [1.0, -1.0, 0.0],
                                         [0.25, 0.5, 0.75],
                                         [-0.5, 1.5, -1.0],
                                         [2.0, 0.0, 1.0],
                                         [1.25, -0.25, 0.5],
                                         [0.75, 0.25, -0.75],
                                         [-1.0, -0.5, 0.5]]))
    Y = torch.tensor([[0, 3, 5], [6, 1, 4]])
    mugrade.submit(layer(Y).detach().numpy())
    mugrade.submit(layer(torch.tensor([7, 2, 0])).detach().numpy())
    mugrade.submit(type(layer.weight))


def test_silu(silu):
    torch.manual_seed(2)
    X = torch.randn(10, 20)
    assert(torch.allclose(silu(X), F.silu(X), atol=1e-6))

    X = torch.randn(3, 4, 5, 6)
    assert(torch.allclose(silu(X), F.silu(X), atol=1e-6))


def submit_silu(silu):
    X = torch.tensor([[-2.0, -0.5, 0.0, 1.0],
                      [2.0, 3.0, -1.0, 0.25]])
    mugrade.submit(silu(X).detach().numpy())
    mugrade.submit(silu(torch.tensor([-3.0, 0.0, 3.0])).detach().numpy())


def test_rms_norm(rms_norm):
    torch.manual_seed(3)
    X = torch.randn(100, 20)
    assert(torch.allclose(rms_norm(X), F.rms_norm(X, (20,)), atol=1e-6))

    X = torch.randn(10, 7, 20)
    assert(torch.allclose(rms_norm(X, eps=1e-3), F.rms_norm(X, (20,), eps=1e-3), atol=1e-6))


def submit_rms_norm(rms_norm):
    X = torch.tensor([[1.0, -1.0, 0.5, 0.5],
                      [2.0, 0.0, -2.0, 1.0]])
    mugrade.submit(rms_norm(X, eps=1e-4).detach().numpy())
    mugrade.submit(rms_norm(X.unsqueeze(0)).detach().numpy())


def test_self_attention(self_attention):
    torch.manual_seed(4)
    Q = torch.randn(5, 8)
    K = torch.randn(5, 8)
    V = torch.randn(5, 6)
    mask = _causal_mask(5)

    ref = F.scaled_dot_product_attention(Q.unsqueeze(0).unsqueeze(0),
                                         K.unsqueeze(0).unsqueeze(0),
                                         V.unsqueeze(0).unsqueeze(0),
                                         attn_mask=mask,
                                         dropout_p=0.0)[0, 0]
    out = self_attention(Q, K, V, mask)
    assert(out.shape == (5, 6))
    assert(torch.allclose(out, ref, atol=1e-6))

    Q = torch.randn(2, 3, 5, 8)
    K = torch.randn(2, 3, 5, 8)
    V = torch.randn(2, 3, 5, 4)
    ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=0.0)
    out = self_attention(Q, K, V, mask)
    assert(out.shape == (2, 3, 5, 4))
    assert(torch.allclose(out, ref, atol=1e-6))


def submit_self_attention(self_attention):
    Q = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0]])
    K = torch.tensor([[1.0, 0.5],
                      [0.0, 1.0],
                      [1.0, -1.0]])
    V = torch.tensor([[0.5, 1.0],
                      [1.5, -0.5],
                      [-1.0, 0.25]])
    mask = _causal_mask(3)
    mugrade.submit(self_attention(Q, K, V, mask).detach().numpy())
    mugrade.submit(self_attention(Q, K, V).detach().numpy())


def test_MultiHeadAttentionKVCache(MultiHeadAttentionKVCache):
    torch.manual_seed(5)
    attn = MultiHeadAttentionKVCache(12, 3, max_cache_size=8)
    ref_attn = nn.MultiheadAttention(12, 3, bias=False, batch_first=True)

    with torch.no_grad():
        attn.wq.weight.copy_(torch.randn_like(attn.wq.weight))
        attn.wk.weight.copy_(torch.randn_like(attn.wk.weight))
        attn.wv.weight.copy_(torch.randn_like(attn.wv.weight))
        attn.wp.weight.copy_(torch.randn_like(attn.wp.weight))
    _copy_mha_weights(attn, ref_attn)

    buffers = dict(attn.named_buffers())
    assert("k_cache" in buffers and "v_cache" in buffers)
    assert(attn.k_cache.shape == (1, 8, 12))
    assert(attn.v_cache.shape == (1, 8, 12))

    X = torch.randn(1, 5, 12)
    mask = _causal_mask(5)

    ref = ref_attn(X, X, X, attn_mask=mask, need_weights=False)[0]
    out = attn(X, mask=mask, use_kv_cache=False)
    assert(out.shape == (1, 5, 12))
    assert(torch.allclose(out, ref, atol=1e-6))

    prefix = attn(X[:, :3], mask=mask[:3, :3], seq_pos=0, use_kv_cache=True)
    tail = attn(X[:, 3:], mask=mask[3:, :], seq_pos=3, use_kv_cache=True)
    assert(torch.allclose(prefix, ref[:, :3], atol=1e-6))
    assert(torch.allclose(tail, ref[:, 3:], atol=1e-6))

    with torch.no_grad():
        K = attn.wk(X)
        V = attn.wv(X)
    assert(torch.allclose(attn.k_cache[:, :5], K, atol=1e-6))
    assert(torch.allclose(attn.v_cache[:, :5], V, atol=1e-6))


def submit_MultiHeadAttentionKVCache(MultiHeadAttentionKVCache):
    attn = MultiHeadAttentionKVCache(4, 2, max_cache_size=6)
    with torch.no_grad():
        attn.wq.weight.copy_(torch.eye(4))
        attn.wk.weight.copy_(torch.tensor([[1.0, 0.5, 0.0, 0.0],
                                           [0.0, 1.0, 0.5, 0.0],
                                           [0.0, 0.0, 1.0, 0.5],
                                           [0.5, 0.0, 0.0, 1.0]]))
        attn.wv.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                           [0.0, -1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0, -1.0]]))
        attn.wp.weight.copy_(torch.eye(4))
    X = torch.tensor([[[1.0, 0.0, -1.0, 0.5],
                       [0.0, 1.0, 0.5, -0.5],
                       [1.0, 1.0, 0.0, 1.0],
                       [-1.0, 0.5, 1.0, 0.0]]])
    mask = _causal_mask(4)

    full = attn(X, mask=mask, use_kv_cache=False)
    prefix = attn(X[:, :2], mask=mask[:2, :2], seq_pos=0, use_kv_cache=True)
    tail = attn(X[:, 2:], mask=mask[2:, :], seq_pos=2, use_kv_cache=True)

    mugrade.submit(full.detach().numpy())
    mugrade.submit(prefix.detach().numpy())
    mugrade.submit(tail.detach().numpy())
    mugrade.submit(attn.k_cache[:, :4].detach().numpy())


def test_MLP(MLP):
    torch.manual_seed(6)
    mlp = MLP(5, 7)
    assert(hasattr(mlp, "w1"))
    assert(hasattr(mlp, "w2"))
    assert(mlp.w1.weight.shape == (7, 5))
    assert(mlp.w2.weight.shape == (5, 7))

    with torch.no_grad():
        mlp.w1.weight.copy_(torch.randn_like(mlp.w1.weight))
        mlp.w2.weight.copy_(torch.randn_like(mlp.w2.weight))

    X = torch.randn(4, 3, 5)
    ref = _ref_mlp(mlp, X)
    out = mlp(X)
    assert(out.shape == (4, 3, 5))
    assert(torch.allclose(out, ref, atol=1e-6))


def submit_MLP(MLP):
    mlp = MLP(2, 3)
    with torch.no_grad():
        mlp.w1.weight.copy_(torch.tensor([[1.0, 0.0],
                                          [0.0, 1.0],
                                          [1.0, -1.0]]))
        mlp.w2.weight.copy_(torch.tensor([[1.0, 0.0, -1.0],
                                          [0.0, 1.0, 1.0]]))
    X = torch.tensor([[1.0, -1.0],
                      [2.0, 0.0],
                      [0.5, 0.5]])
    mugrade.submit(mlp(X).detach().numpy())
    mugrade.submit(type(mlp.w1))


def test_TransformerBlock(TransformerBlock):
    torch.manual_seed(7)
    block = TransformerBlock(12, 3, 16, 8)
    X = torch.randn(1, 5, 12)
    mask = _causal_mask(5)

    ref = _ref_block(block, X, mask=mask)
    out = block(X, mask=mask)
    assert(out.shape == (1, 5, 12))
    assert(torch.allclose(out, ref, atol=3e-5, rtol=3e-5))

    full = block(X, mask=mask, use_kv_cache=False)
    block(X[:, :3], mask=mask[:3, :3], seq_pos=0, use_kv_cache=True)
    tail = block(X[:, 3:], mask=mask[3:, :], seq_pos=3, use_kv_cache=True)
    assert(torch.allclose(full[:, 3:], tail, atol=3e-5, rtol=3e-5))


def submit_TransformerBlock(TransformerBlock):
    torch.manual_seed(8)
    block = TransformerBlock(4, 2, 6, 5)
    with torch.no_grad():
        block.attn.wq.weight.copy_(torch.eye(4))
        block.attn.wk.weight.copy_(torch.eye(4))
        block.attn.wv.weight.copy_(torch.eye(4))
        block.attn.wp.weight.copy_(torch.eye(4))
        block.mlp.w1.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0],
                                                [1.0, -1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, -1.0]]))
        block.mlp.w2.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                                                [0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
                                                [0.0, 0.0, 1.0, 0.0, -0.5, 0.0],
                                                [0.0, 0.0, 0.0, 1.0, 0.0, -0.5]]))
    X = torch.tensor([[[1.0, 0.0, -1.0, 0.5],
                       [0.0, 1.0, 0.5, -0.5],
                       [1.0, 1.0, 0.0, 1.0]]])
    mask = _causal_mask(3)
    mugrade.submit(block(X, mask=mask).detach().numpy())
    block(X[:, :2], mask=mask[:2, :2], seq_pos=0, use_kv_cache=True)
    mugrade.submit(block(X[:, 2:], mask=mask[2:, :], seq_pos=2, use_kv_cache=True).detach().numpy())


def test_LLM(LLM):
    torch.manual_seed(9)
    model = LLM(num_tokens=11, dim=12, n_heads=3, max_seq_len=8, ffn_dim=16, num_layers=2)
    assert(model.embedding.weight.shape == (11, 12))
    assert(model.pos_embeddings.shape == (8, 12))
    assert(len(model.layers) == 2)
    assert(model.output.weight.shape == (11, 12))
    assert(model.mask.shape == (8, 8))
    assert(model.mask[0, 0].item() == 0.0)
    assert(torch.isneginf(model.mask[0, 1]))
    assert(model.mask[1, 0].item() == 0.0)

    tokens = torch.tensor([[0, 1, 2, 3, 4]])
    ref = _ref_llm(model, tokens)
    out = model(tokens)
    assert(out.shape == (1, 5, 11))
    assert(torch.allclose(out, ref, atol=6e-5, rtol=6e-5))

    model(tokens[:, :3], seq_pos=0, use_kv_cache=True)
    tail = model(tokens[:, 3:], seq_pos=3, use_kv_cache=True)
    assert(torch.allclose(out[:, 3:], tail, atol=6e-5, rtol=6e-5))


def submit_LLM(LLM):
    torch.manual_seed(10)
    model = LLM(num_tokens=7, dim=4, n_heads=2, max_seq_len=6, ffn_dim=6, num_layers=1)
    tokens = torch.tensor([[2, 3, 4]])
    mugrade.submit(model(tokens).detach().numpy())
    mugrade.submit(torch.isneginf(model.mask[:4, :4]).detach().numpy())
    mugrade.submit(type(model.layers))


def test_cross_entropy_loss(cross_entropy_loss):
    torch.manual_seed(11)
    logits = torch.randn(100, 50)
    y = torch.randint(0, 50, (100,))
    ref = nn.CrossEntropyLoss()(logits, y)
    mine = cross_entropy_loss(logits, y)
    assert(mine.ndim == 0)
    assert(torch.allclose(ref, mine, atol=1e-6))

    logits = torch.randn(4, 5, 7)
    y = torch.randint(0, 7, (4, 5))
    ref = F.cross_entropy(logits.reshape(-1, 7), y.reshape(-1))
    mine = cross_entropy_loss(logits, y)
    assert(torch.allclose(ref, mine, atol=1e-6))


def submit_cross_entropy_loss(cross_entropy_loss):
    logits = torch.tensor([[2.0, 1.0, 0.0],
                           [0.0, 2.0, 1.0],
                           [1.5, -0.5, 0.25]])
    y = torch.tensor([0, 2, 1])
    loss = cross_entropy_loss(logits, y)
    mugrade.submit(loss.item())
    mugrade.submit(type(loss))

    logits = torch.tensor([[[1.0, 0.0], [0.5, -0.5]],
                           [[-1.0, 2.0], [3.0, 1.0]]])
    y = torch.tensor([[0, 1], [1, 0]])
    mugrade.submit(cross_entropy_loss(logits, y).item())


def test_pretokenize_data(pretokenize_data):
    tokenizer = _ChunkTokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        in_filename = os.path.join(tmpdir, "sample.txt")
        out_filename = os.path.join(tmpdir, "sample.bin")

        with open(in_filename, "wt") as f:
            f.write("abcdefg")

        pretokenize_data(tokenizer, in_filename, out_filename, chunk_size=3, max_chunks=2)
        tokens = np.frombuffer(open(out_filename, "rb").read(), dtype=np.uint16).tolist()

    assert(tokenizer.calls == [("abc", "all"), ("def", "all")])
    assert(tokens == [97, 98, 99, 100, 101, 102])


def submit_pretokenize_data(pretokenize_data):
    tokenizer = _ChunkTokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        in_filename = os.path.join(tmpdir, "sample.txt")
        out_filename = os.path.join(tmpdir, "sample.bin")

        with open(in_filename, "wt") as f:
            f.write("hello!")

        pretokenize_data(tokenizer, in_filename, out_filename, chunk_size=2, max_chunks=3)
        tokens = np.frombuffer(open(out_filename, "rb").read(), dtype=np.uint16)

    mugrade.submit(tokens)
    mugrade.submit(tokenizer.calls)


def test_DataLoader(DataLoader):
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "tokens.bin")
        _write_token_file(filename, list(range(20)))

        loader = DataLoader(filename, seq_len=3, batch_size=2)
        assert(iter(loader) is loader)

        batches = list(loader)
        assert(len(batches) == 3)

        expected = [
            (torch.tensor([[0, 1, 2], [4, 5, 6]]), torch.tensor([[1, 2, 3], [5, 6, 7]])),
            (torch.tensor([[6, 7, 8], [10, 11, 12]]), torch.tensor([[7, 8, 9], [11, 12, 13]])),
            (torch.tensor([[12, 13, 14], [16, 17, 18]]), torch.tensor([[13, 14, 15], [17, 18, 19]])),
        ]

        for (Xb, yb), (X_ref, y_ref) in zip(batches, expected):
            assert(torch.equal(Xb.cpu(), X_ref))
            assert(torch.equal(yb.cpu(), y_ref))

        batches2 = list(loader)
        for (Xb1, yb1), (Xb2, yb2) in zip(batches, batches2):
            assert(torch.equal(Xb1, Xb2))
            assert(torch.equal(yb1, yb2))


def submit_DataLoader(DataLoader):
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "tokens.bin")
        _write_token_file(filename, list(range(20)))

        loader = DataLoader(filename, seq_len=3, batch_size=2)
        batches = list(loader)

    mugrade.submit(len(batches))
    for Xb, yb in batches:
        mugrade.submit(Xb.cpu().numpy())
        mugrade.submit(yb.cpu().numpy())


def test_Adam(Adam):
    torch.manual_seed(12)
    model = nn.Sequential(nn.Linear(6, 4, bias=False), nn.Tanh(), nn.Linear(4, 3, bias=False))
    ref_model = copy.deepcopy(model)

    opt = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    ref_opt = optim.Adam(ref_model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(5):
        X = torch.randn(16, 6)
        y = torch.randint(0, 3, (16,))

        opt.zero_grad()
        ref_opt.zero_grad()

        loss = loss_fn(model(X), y)
        ref_loss = loss_fn(ref_model(X), y)

        loss.backward()
        ref_loss.backward()

        opt.step()
        ref_opt.step()

    for p, p_ref in zip(model.parameters(), ref_model.parameters()):
        assert(torch.allclose(p, p_ref, atol=1e-6))

    loss = loss_fn(model(X), y)
    loss.backward()
    opt.zero_grad()
    for p in model.parameters():
        if p.grad is not None:
            assert(torch.allclose(p.grad, torch.zeros_like(p.grad)))


def submit_Adam(Adam):
    w = nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))
    b = nn.Parameter(torch.tensor([0.25, -0.75]))
    opt = Adam([w, b], lr=0.1, betas=(0.8, 0.9), eps=1e-6)

    w.grad = torch.tensor([[0.1, -0.4], [-0.2, 0.3]])
    b.grad = torch.tensor([0.5, -0.25])
    opt.step()

    w.grad = torch.tensor([[0.2, 0.1], [-0.1, -0.3]])
    b.grad = torch.tensor([-0.5, 0.5])
    opt.step()

    mugrade.submit(w.detach().numpy())
    mugrade.submit(b.detach().numpy())
    mugrade.submit(opt.u[0].detach().numpy())
    mugrade.submit(opt.v[1].detach().numpy())


def test_train_llm(train_llm):
    torch.manual_seed(13)
    model = _TinyTrainModel()
    ref_model = copy.deepcopy(model)
    opt = optim.SGD(model.parameters(), lr=0.2)
    ref_opt = optim.SGD(ref_model.parameters(), lr=0.2)

    loader = [
        (torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.tensor([[1, 2, 3], [2, 3, 4]])),
        (torch.tensor([[2, 3, 4], [0, 2, 4]]), torch.tensor([[3, 4, 0], [2, 4, 1]])),
    ]

    old_loss = train_llm.__globals__["cross_entropy_loss"]
    try:
        train_llm.__globals__["cross_entropy_loss"] = _checked_cross_entropy
        with redirect_stdout(io.StringIO()):
            train_llm(model, loader, opt)
    finally:
        train_llm.__globals__["cross_entropy_loss"] = old_loss

    for x, y in loader:
        loss = F.cross_entropy(ref_model(x).float().reshape(-1, 5), y.reshape(-1))
        ref_opt.zero_grad()
        loss.backward()
        ref_opt.step()

    for p, p_ref in zip(model.parameters(), ref_model.parameters()):
        assert(torch.allclose(p, p_ref, atol=1e-6))


def submit_train_llm(train_llm):
    torch.manual_seed(14)
    model = _TinyTrainModel()
    opt = optim.SGD(model.parameters(), lr=0.1)
    loader = [
        (torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.tensor([[1, 2, 3], [2, 3, 4]])),
        (torch.tensor([[2, 3, 4], [0, 2, 4]]), torch.tensor([[3, 4, 0], [2, 4, 1]])),
    ]

    old_loss = train_llm.__globals__["cross_entropy_loss"]
    try:
        train_llm.__globals__["cross_entropy_loss"] = _checked_cross_entropy
        with redirect_stdout(io.StringIO()):
            train_llm(model, loader, opt)
    finally:
        train_llm.__globals__["cross_entropy_loss"] = old_loss

    mugrade.submit(model.embed.weight[:2].detach().numpy())
    mugrade.submit(model.proj.weight[:2].detach().numpy())
    mugrade.submit(model(torch.tensor([[0, 1, 2]])).float().detach().numpy())


def test_generate(generate):
    model = _ToyGenerationModel([3, 4])
    tokenizer = _ToyGenerationTokenizer(eot_token=4)

    out = io.StringIO()
    with redirect_stdout(out):
        generated = generate(model, [1, 2], tokenizer, temp=0.7, max_tokens=5, verbose=True)

    assert(generated == [3, 4])
    assert(out.getvalue() == "A!")
    assert(len(model.calls) == 2)
    assert(model.calls[0][0].tolist() == [[1, 2]])
    assert(model.calls[0][1] == 0 and model.calls[0][2] is True)
    assert(model.calls[0][3] == torch.long)
    assert(model.calls[1][0].tolist() == [[3]])
    assert(model.calls[1][1] == 2 and model.calls[1][2] is True)


def submit_generate(generate):
    model = _ToyGenerationModel([3, 5, 4])
    tokenizer = _ToyGenerationTokenizer(eot_token=4)

    out = io.StringIO()
    with redirect_stdout(out):
        generated = generate(model, [1, 2], tokenizer, temp=0.7, max_tokens=6, verbose=True)

    mugrade.submit(generated)
    mugrade.submit(out.getvalue())
    mugrade.submit([seq_pos for _, seq_pos, _, _, _ in model.calls])


def test_eval_llm(eval_llm):
    model = eval_llm()
    assert(isinstance(model, nn.Module))
    assert(hasattr(model, "layers"))
    assert(len(model.layers) > 0)

    tokens = _tiny_stories_eval_tokens(start_token=0)
    phrase_loss = _sequence_loss(model, tokens)
    corrupted = tokens.clone()
    corrupted[:, 1:] = corrupted[:, 1:].flip(-1)
    corrupted_loss = _sequence_loss(model, corrupted)

    with torch.inference_mode():
        device = next(model.parameters()).device
        tokens = tokens.to(device)
        full = model(tokens[:, :-1])
        model(tokens[:, :-2], seq_pos=0, use_kv_cache=True)
        tail = model(tokens[:, -2:-1], seq_pos=tokens.shape[1] - 2, use_kv_cache=True)

    assert(phrase_loss < 7.0)
    assert(phrase_loss < corrupted_loss)
    assert(full.shape == (1, tokens.shape[1] - 1, model.output.weight.shape[0]))
    assert(torch.isfinite(full[:, :, :16].float()).all())
    assert(torch.allclose(full[:, -1:].float(), tail.float(), atol=3e-4, rtol=3e-4))


def submit_eval_llm(eval_llm):
    model = eval_llm()
    tokens = _tiny_stories_eval_tokens(start_token=64)
    phrase_loss = _sequence_loss(model, tokens)
    corrupted = tokens.clone()
    corrupted[:, 1:] = corrupted[:, 1:].flip(-1)
    corrupted_loss = _sequence_loss(model, corrupted)

    with torch.inference_mode():
        device = next(model.parameters()).device
        tokens = tokens.to(device)
        full = model(tokens[:, :-1])
        model(tokens[:, :-2], seq_pos=0, use_kv_cache=True)
        tail = model(tokens[:, -2:-1], seq_pos=tokens.shape[1] - 2, use_kv_cache=True)

    mugrade.submit(phrase_loss < 7.0)
    mugrade.submit(phrase_loss < corrupted_loss)
    mugrade.submit(torch.allclose(full[:, -1:].float(), tail.float(), atol=3e-4, rtol=3e-4))
