import io
from contextlib import redirect_stdout

import mugrade
import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_mask(length):
    return torch.triu(torch.full((length, length), float("-inf")), diagonal=1)


def _copy_linear_weight(linear, weight):
    with torch.no_grad():
        linear.weight.copy_(weight)


def _copy_mha_weights(layer, ref_layer):
    with torch.no_grad():
        ref_layer.in_proj_weight.copy_(torch.cat([layer.wq.weight, layer.wk.weight, layer.wv.weight], dim=0))
        ref_layer.out_proj.weight.copy_(layer.wp.weight)


def _zero_linear(linear):
    with torch.no_grad():
        linear.weight.zero_()


def _identity_linear(linear):
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight.copy_(torch.eye(linear.weight.shape[0], linear.weight.shape[1]))


class _ToyTokenizer:
    def __init__(self, stop_tokens):
        self.stop_tokens = set(stop_tokens)
        self._vocab = {
            3: "A",
            4: "!",
            5: "B",
        }

    def decode(self, tokens):
        if isinstance(tokens, int):
            tokens = [tokens]
        return "".join(self._vocab[t] for t in tokens)


class _ToyModel:
    def __init__(self, next_tokens, vocab_size=6):
        self.next_tokens = list(next_tokens)
        self.vocab_size = vocab_size
        self.calls = []

    def __call__(self, tokens, seq_pos=0, use_kv_cache=False):
        self.calls.append((tokens.clone(), seq_pos, use_kv_cache))
        logits = torch.full((1, tokens.shape[1], self.vocab_size), -1e9)
        next_token = self.next_tokens[len(self.calls) - 1]
        logits[0, -1, next_token] = 0.0
        return logits


def test_Linear(Linear):
    torch.manual_seed(0)
    layer = Linear(10, 20)
    assert(hasattr(layer, "weight"))
    assert(isinstance(layer.weight, nn.Parameter))
    assert(layer.weight.shape == (20, 10))

    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))

    ref_layer = nn.Linear(10, 20, bias=False)
    ref_layer.weight.data = layer.weight.data.clone()

    X = torch.randn(50, 10)
    assert(layer(X).shape == (50, 20))
    assert(torch.allclose(ref_layer(X), layer(X), atol=1e-6))

    X = torch.randn(7, 9, 10)
    assert(layer(X).shape == (7, 9, 20))
    assert(torch.allclose(ref_layer(X), layer(X), atol=1e-6))


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

    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))

    ref_layer = nn.Embedding(200, 20)
    ref_layer.weight.data = layer.weight.data.clone()

    Y = torch.randint(0, 200, size=(13,))
    assert(torch.allclose(ref_layer(Y), layer(Y), atol=1e-6))

    Y = torch.randint(0, 200, size=(7, 11))
    assert(layer(Y).shape == (7, 11, 20))
    assert(torch.allclose(ref_layer(Y), layer(Y), atol=1e-6))


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


def test_RMSNorm(RMSNorm):
    torch.manual_seed(3)
    layer = RMSNorm(20, eps=1e-3)
    assert(hasattr(layer, "weight"))
    assert(hasattr(layer, "eps"))
    assert(isinstance(layer.weight, nn.Parameter))
    assert(layer.weight.shape == (20,))
    assert(torch.allclose(layer.weight, torch.ones(20)))

    ref_layer = nn.RMSNorm(20, eps=1e-3)
    with torch.no_grad():
        layer.weight.copy_(torch.ones(20) + 0.1 * torch.randn(20))
        ref_layer.weight.copy_(layer.weight)

    X = torch.randn(100, 20)
    assert(torch.allclose(layer(X), ref_layer(X), atol=1e-6))

    X = torch.randn(10, 7, 20)
    assert(torch.allclose(layer(X), ref_layer(X), atol=1e-6))


def submit_RMSNorm(RMSNorm):
    layer = RMSNorm(4, eps=1e-4)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([1.0, 1.5, -0.5, 0.25]))
    X = torch.tensor([[1.0, -1.0, 0.5, 0.5],
                      [2.0, 0.0, -2.0, 1.0]])
    mugrade.submit(layer(X).detach().numpy())
    mugrade.submit(layer(X.unsqueeze(0)).detach().numpy())
    mugrade.submit(type(layer.weight))


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


def test_MultiHeadAttention(MultiHeadAttention):
    torch.manual_seed(5)
    attn = MultiHeadAttention(12, 3)
    ref_attn = nn.MultiheadAttention(12, 3, bias=False, batch_first=True)

    with torch.no_grad():
        attn.wq.weight.copy_(torch.randn_like(attn.wq.weight))
        attn.wk.weight.copy_(torch.randn_like(attn.wk.weight))
        attn.wv.weight.copy_(torch.randn_like(attn.wv.weight))
        attn.wp.weight.copy_(torch.randn_like(attn.wp.weight))
    _copy_mha_weights(attn, ref_attn)

    X = torch.randn(2, 5, 12)
    ref = ref_attn(X, X, X, need_weights=False)[0]
    out = attn(X)
    assert(out.shape == (2, 5, 12))
    assert(torch.allclose(out, ref, atol=1e-6))

    mask = _causal_mask(5)
    ref = ref_attn(X, X, X, attn_mask=mask, need_weights=False)[0]
    out = attn(X, mask=mask)
    assert(torch.allclose(out, ref, atol=1e-6))


def submit_MultiHeadAttention(MultiHeadAttention):
    attn = MultiHeadAttention(4, 2)
    with torch.no_grad():
        attn.wq.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0, 1.0]]))
        attn.wk.weight.copy_(torch.tensor([[1.0, 0.0, 0.5, 0.0],
                                           [0.0, 1.0, 0.0, 0.5],
                                           [0.5, 0.0, 1.0, 0.0],
                                           [0.0, 0.5, 0.0, 1.0]]))
        attn.wv.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                           [0.0, -1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0, -1.0]]))
        attn.wp.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0, 1.0]]))
    X = torch.tensor([[[1.0, 0.0, -1.0, 0.5],
                       [0.0, 1.0, 0.5, -0.5],
                       [1.0, 1.0, 0.0, 1.0]]])
    mask = _causal_mask(3)
    mugrade.submit(attn(X).detach().numpy())
    mugrade.submit(attn(X, mask=mask).detach().numpy())
    mugrade.submit(type(attn.wq))


def test_MultiHeadAttentionKVCache(MultiHeadAttentionKVCache):
    torch.manual_seed(6)
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
        attn.wq.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0, 1.0]]))
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


def test_GatedMLP(GatedMLP):
    mlp = GatedMLP(2, 3)
    assert(hasattr(mlp, "w1"))
    assert(hasattr(mlp, "w2"))
    assert(hasattr(mlp, "w3"))
    assert(mlp.w1.weight.shape == (3, 2))
    assert(mlp.w2.weight.shape == (2, 3))
    assert(mlp.w3.weight.shape == (3, 2))

    with torch.no_grad():
        mlp.w1.weight.copy_(torch.tensor([[1.0, 0.0],
                                          [0.0, 1.0],
                                          [1.0, -1.0]]))
        mlp.w2.weight.copy_(torch.tensor([[1.0, 0.0, -1.0],
                                          [0.0, 1.0, 1.0]]))
        mlp.w3.weight.copy_(torch.tensor([[1.0, 1.0],
                                          [2.0, 0.0],
                                          [0.0, -1.0]]))

    X = torch.tensor([[0.0, 0.0],
                      [1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0]])
    expected = torch.tensor([[0.0, 0.0],
                             [0.7310585975646973, 0.0],
                             [-0.2689414322376251, 0.2689414322376251],
                             [1.4621171951293945, 1.4621171951293945]])
    assert(torch.allclose(mlp(X), expected, atol=1e-6))
    assert(mlp(torch.randn(2, 4, 2)).shape == (2, 4, 2))


def submit_GatedMLP(GatedMLP):
    mlp = GatedMLP(2, 3)
    with torch.no_grad():
        mlp.w1.weight.copy_(torch.tensor([[1.0, 0.0],
                                          [0.0, 1.0],
                                          [1.0, -1.0]]))
        mlp.w2.weight.copy_(torch.tensor([[1.0, 0.0, -1.0],
                                          [0.0, 1.0, 1.0]]))
        mlp.w3.weight.copy_(torch.tensor([[1.0, 1.0],
                                          [2.0, 0.0],
                                          [0.0, -1.0]]))
    X = torch.tensor([[1.0, -1.0],
                      [2.0, 0.0],
                      [0.5, 0.5]])
    mugrade.submit(mlp(X).detach().numpy())
    mugrade.submit(type(mlp.w1))


def test_TransformerBlock(TransformerBlock):
    X = torch.tensor([[[1.0, 1.0],
                       [1.0, -1.0]]])

    block = TransformerBlock(2, 1, 3, 4)
    with torch.no_grad():
        block.norm1.weight.fill_(1.0)
        block.norm2.weight.fill_(1.0)
        block.norm1.eps = 0.0
        block.norm2.eps = 0.0
        _zero_linear(block.attn.wq)
        _zero_linear(block.attn.wk)
        _zero_linear(block.attn.wv)
        _zero_linear(block.attn.wp)
        block.mlp.w1.weight.copy_(torch.tensor([[1.0, 0.0],
                                                [0.0, 1.0],
                                                [1.0, -1.0]]))
        block.mlp.w2.weight.copy_(torch.tensor([[1.0, 0.0, -1.0],
                                                [0.0, 1.0, 1.0]]))
        block.mlp.w3.weight.copy_(torch.tensor([[1.0, 1.0],
                                                [2.0, 0.0],
                                                [0.0, -1.0]]))
    expected = torch.tensor([[[2.4621171951293945, 2.4621171951293945],
                              [-0.7615940570831299, 0.22371125221252441]]])
    assert(torch.allclose(block(X), expected, atol=1e-6))

    block = TransformerBlock(2, 1, 3, 4)
    with torch.no_grad():
        block.norm1.weight.fill_(1.0)
        block.norm2.weight.fill_(1.0)
        block.norm1.eps = 0.0
        block.norm2.eps = 0.0
        _identity_linear(block.attn.wq)
        _identity_linear(block.attn.wk)
        _identity_linear(block.attn.wv)
        _identity_linear(block.attn.wp)
        _zero_linear(block.mlp.w1)
        _zero_linear(block.mlp.w2)
        _zero_linear(block.mlp.w3)
    mask = _causal_mask(2)
    expected = torch.tensor([[[2.0, 2.0],
                              [2.0, -1.6088594198226929]]])
    assert(torch.allclose(block(X, mask=mask), expected, atol=1e-6))


def submit_TransformerBlock(TransformerBlock):
    X = torch.tensor([[[1.0, 1.0],
                       [1.0, -1.0]]])

    block = TransformerBlock(2, 1, 3, 4)
    with torch.no_grad():
        block.norm1.weight.fill_(1.0)
        block.norm2.weight.fill_(1.0)
        block.norm1.eps = 0.0
        block.norm2.eps = 0.0
        _zero_linear(block.attn.wq)
        _zero_linear(block.attn.wk)
        _zero_linear(block.attn.wv)
        _zero_linear(block.attn.wp)
        block.mlp.w1.weight.copy_(torch.tensor([[1.0, 0.0],
                                                [0.0, 1.0],
                                                [1.0, -1.0]]))
        block.mlp.w2.weight.copy_(torch.tensor([[1.0, 0.0, -1.0],
                                                [0.0, 1.0, 1.0]]))
        block.mlp.w3.weight.copy_(torch.tensor([[1.0, 1.0],
                                                [2.0, 0.0],
                                                [0.0, -1.0]]))
    mugrade.submit(block(X).detach().numpy())

    block = TransformerBlock(2, 1, 3, 4)
    with torch.no_grad():
        block.norm1.weight.fill_(1.0)
        block.norm2.weight.fill_(1.0)
        block.norm1.eps = 0.0
        block.norm2.eps = 0.0
        _identity_linear(block.attn.wq)
        _identity_linear(block.attn.wk)
        _identity_linear(block.attn.wv)
        _identity_linear(block.attn.wp)
        _zero_linear(block.mlp.w1)
        _zero_linear(block.mlp.w2)
        _zero_linear(block.mlp.w3)
    mugrade.submit(block(X, mask=_causal_mask(2)).detach().numpy())


def test_Llama3Simplified(Llama3Simplified):
    model = Llama3Simplified(num_tokens=5, dim=4, n_heads=2, max_seq_len=6, ffn_dim=6, num_layers=1)
    assert(model.embedding.weight.shape == (5, 4))
    assert(model.pos_embeddings.shape == (6, 4))
    assert(len(model.layers) == 1)
    assert(model.output.weight.shape == (5, 4))
    buffers = dict(model.named_buffers())
    assert("mask" in buffers)
    assert(model.mask.shape == (6, 6))
    assert(model.mask[0, 0].item() == 0.0)
    assert(torch.isneginf(model.mask[0, 1]))
    assert(model.mask[1, 0].item() == 0.0)

    with torch.no_grad():
        model.embedding.weight.zero_()
        model.pos_embeddings.zero_()
        model.pos_embeddings[0].copy_(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        model.pos_embeddings[1].copy_(torch.tensor([1.0, -1.0, 1.0, -1.0]))
        model.pos_embeddings[2].copy_(torch.tensor([-1.0, 1.0, -1.0, 1.0]))
        model.pos_embeddings[3].copy_(torch.tensor([1.0, 1.0, -1.0, -1.0]))
        model.norm.weight.fill_(1.0)
        model.norm.eps = 0.0
        model.output.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0]]))
        for layer in model.layers:
            layer.norm1.weight.fill_(1.0)
            layer.norm2.weight.fill_(1.0)
            _zero_linear(layer.attn.wq)
            _zero_linear(layer.attn.wk)
            _zero_linear(layer.attn.wv)
            _zero_linear(layer.attn.wp)
            _zero_linear(layer.mlp.w1)
            _zero_linear(layer.mlp.w2)
            _zero_linear(layer.mlp.w3)

    tokens = torch.tensor([[2, 3, 4]])
    out = model(tokens)
    expected = torch.tensor([[[1.0, 1.0, 1.0, 1.0, 4.0],
                              [1.0, -1.0, 1.0, -1.0, 0.0],
                              [-1.0, 1.0, -1.0, 1.0, 0.0],
                              ]])
    assert(out.shape == (1, 3, 5))
    assert(torch.allclose(out, expected, atol=1e-6))


def submit_Llama3Simplified(Llama3Simplified):
    model = Llama3Simplified(num_tokens=5, dim=4, n_heads=2, max_seq_len=6, ffn_dim=6, num_layers=1)
    with torch.no_grad():
        model.embedding.weight.zero_()
        model.pos_embeddings.zero_()
        model.pos_embeddings[0].copy_(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        model.pos_embeddings[1].copy_(torch.tensor([1.0, -1.0, 1.0, -1.0]))
        model.pos_embeddings[2].copy_(torch.tensor([-1.0, 1.0, -1.0, 1.0]))
        model.pos_embeddings[3].copy_(torch.tensor([1.0, 1.0, -1.0, -1.0]))
        model.norm.weight.fill_(1.0)
        model.norm.eps = 0.0
        model.output.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0],
                                                [1.0, 1.0, 1.0, 1.0]]))
        for layer in model.layers:
            layer.norm1.weight.fill_(1.0)
            layer.norm2.weight.fill_(1.0)
            _zero_linear(layer.attn.wq)
            _zero_linear(layer.attn.wk)
            _zero_linear(layer.attn.wv)
            _zero_linear(layer.attn.wp)
            _zero_linear(layer.mlp.w1)
            _zero_linear(layer.mlp.w2)
            _zero_linear(layer.mlp.w3)

    tokens = torch.tensor([[2, 3, 4]])
    mugrade.submit(model(tokens).detach().numpy())
    mugrade.submit(torch.isneginf(model.mask[:4, :4]).numpy())
    mugrade.submit(type(model.layers))


def test_eval_llama3(eval_llama3):
    model = eval_llama3()
    assert(isinstance(model, nn.Module))
    assert(len(model.layers) > 0)

    tokens = torch.tensor([[0, 1, 2, 3]])
    with torch.inference_mode():
        full = model(tokens)
        prefix = model(tokens[:, :3], seq_pos=0, use_kv_cache=True)
        tail = model(tokens[:, 3:], seq_pos=3, use_kv_cache=True)

    assert(full.shape[0] == 1 and full.shape[1] == 4)
    assert(prefix.shape[1] == 3)
    assert(tail.shape[1] == 1)
    assert(full.shape[-1] == model.output.weight.shape[0])
    assert(torch.isfinite(full[:, :, :16]).all())
    assert(torch.allclose(full[:, 3:], tail, atol=3e-4, rtol=3e-4))


def submit_eval_llama3(eval_llama3):
    model = eval_llama3()
    tokens = torch.tensor([[0, 1, 2, 3]])
    with torch.inference_mode():
        full = model(tokens)
        model(tokens[:, :3], seq_pos=0, use_kv_cache=True)
        tail = model(tokens[:, 3:], seq_pos=3, use_kv_cache=True)

    mugrade.submit(full[0, -1, :8].detach().numpy())
    mugrade.submit(tail[0, 0, :8].detach().numpy())
    mugrade.submit(int(full[0, -1].argmax().item()))


def test_generate(generate):
    model = _ToyModel([3, 4])
    tokenizer = _ToyTokenizer(stop_tokens={4})

    out = io.StringIO()
    with redirect_stdout(out):
        generated = generate(model, [1, 2], tokenizer, temp=0.7, max_tokens=5, verbose=True)

    assert(generated == [3, 4])
    assert(out.getvalue() == "A!")
    assert(len(model.calls) == 2)
    assert(model.calls[0][0].tolist() == [[1, 2]])
    assert(model.calls[0][1] == 0 and model.calls[0][2] is True)
    assert(model.calls[1][0].tolist() == [[3]])
    assert(model.calls[1][1] == 2 and model.calls[1][2] is True)


def submit_generate(generate):
    model = _ToyModel([3, 5, 4])
    tokenizer = _ToyTokenizer(stop_tokens={4})

    out = io.StringIO()
    with redirect_stdout(out):
        generated = generate(model, [1, 2], tokenizer, temp=0.7, max_tokens=6, verbose=True)

    mugrade.submit(generated)
    mugrade.submit(out.getvalue())
    mugrade.submit([seq_pos for _, seq_pos, _ in model.calls])
