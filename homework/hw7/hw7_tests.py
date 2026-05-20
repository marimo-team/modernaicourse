import copy
import io
import json
import math
import os
import tempfile
from contextlib import redirect_stdout

import mugrade
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


class _FixedEncodeTokenizer:
    def __init__(self, outputs):
        self.outputs = {k: list(v) for k, v in outputs.items()}
        self.calls = []

    def encode(self, text, allowed_special="all"):
        self.calls.append((text, allowed_special))
        return list(self.outputs[text])


class _ToyGsmTokenizer:
    def __init__(self):
        self._special_tokens = {
            "<QUESTION>": 91,
            "</QUESTION>": 92,
            "<THINK>": 93,
            "</THINK>": 94,
            "<TOOL>": 95,
            "</TOOL>": 96,
            "<RESPONSE>": 97,
            "</RESPONSE>": 98,
            "<ANSWER>": 99,
            "</ANSWER>": 100,
        }


class _CharSpecialTokenizer:
    def __init__(self):
        specials = [
            "<QUESTION>",
            "</QUESTION>",
            "<THINK>",
            "</THINK>",
            "<TOOL>",
            "</TOOL>",
            "<RESPONSE>",
            "</RESPONSE>",
            "<ANSWER>",
            "</ANSWER>",
        ]
        self._special_tokens = {token: i + 1 for i, token in enumerate(specials)}
        self._id_to_special = {idx: token for token, idx in self._special_tokens.items()}
        self._char_to_id = {}
        self._id_to_char = {}
        self._next_id = len(self._special_tokens) + 1
        for ch in "0123456789+-*/(). ERRORabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ?:,\n ":
            self._get_char_id(ch)

    def _get_char_id(self, ch):
        if ch not in self._char_to_id:
            idx = self._next_id
            self._next_id += 1
            self._char_to_id[ch] = idx
            self._id_to_char[idx] = ch
        return self._char_to_id[ch]

    def encode(self, text, allowed_special="all"):
        out = []
        i = 0
        special_tokens = sorted(self._special_tokens, key=len, reverse=True)
        while i < len(text):
            matched = None
            if allowed_special == "all":
                for token in special_tokens:
                    if text.startswith(token, i):
                        matched = token
                        break
            if matched is not None:
                out.append(self._special_tokens[matched])
                i += len(matched)
            else:
                out.append(self._get_char_id(text[i]))
                i += 1
        return out

    def decode(self, tokens):
        pieces = []
        for token in tokens:
            token = int(token)
            if token == 0:
                continue
            if token in self._id_to_special:
                pieces.append(self._id_to_special[token])
            else:
                pieces.append(self._id_to_char[token])
        return "".join(pieces)


class _BatchedToyGenerationModel(nn.Module):
    def __init__(self, next_tokens, vocab_size=64):
        super().__init__()
        self.next_tokens = [list(step) for step in next_tokens]
        self.vocab_size = vocab_size
        self.calls = []
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, seq_pos=0, use_kv_cache=False):
        step = len(self.calls)
        assert(step < len(self.next_tokens))
        row_tokens = self.next_tokens[step]
        assert(len(row_tokens) == tokens.shape[0])

        self.calls.append((tokens.detach().clone(), seq_pos, use_kv_cache))
        logits = torch.full((tokens.shape[0], tokens.shape[1], self.vocab_size), -1e9, device=tokens.device)
        for i, token in enumerate(row_tokens):
            logits[i, -1, token] = 0.0
        return logits


class _TinyTrainModel(nn.Module):
    def __init__(self, vocab_size=8, dim=4, output_dtype=torch.float32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, vocab_size, bias=False)
        self.output_dtype = output_dtype
        self.last_logits = None

    def forward(self, tokens):
        self.last_logits = self.proj(self.embed(tokens))
        return self.last_logits.to(self.output_dtype)


class _MaskedCrossEntropySpy:
    def __init__(self, model, batches):
        self.model = model
        self.batches = batches
        self.calls = 0

    def __call__(self, logits, y):
        _, y_ref, mask_ref = self.batches[self.calls]
        expected_logits = self.model.last_logits.float()[mask_ref]
        assert(torch.allclose(logits, expected_logits, atol=1e-6))
        assert(torch.equal(y, y_ref[mask_ref]))
        self.calls += 1
        return F.cross_entropy(logits, y)


class _LookupLM(nn.Module):
    def __init__(self, table):
        super().__init__()
        self.table = nn.Parameter(table.clone())

    def forward(self, tokens):
        return self.table[tokens]


class _ScalarModel(nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(value)))


class _SequenceGenerator:
    def __init__(self, outputs):
        self.outputs = [out.clone() for out in outputs]
        self.calls = []

    def __call__(self, model, prompt_tokens, tokenizer, num_completions=1, eot_token=None, temp=0.7, max_tokens=500):
        idx = len(self.calls)
        self.calls.append((list(prompt_tokens), num_completions, eot_token, temp, max_tokens))
        return self.outputs[idx].clone()


class _BatchSizeOneLoader:
    def __init__(self, batches):
        self.batches = list(batches)
        self.batch_size = 1

    def __iter__(self):
        return iter(self.batches)


class _RlLossSpy:
    def __init__(self, model, tokenizer, expected_tokens, expected_rewards, targets):
        self.model = model
        self.tokenizer = tokenizer
        self.expected_tokens = [tokens.clone() for tokens in expected_tokens]
        self.expected_rewards = [torch.tensor(r, dtype=torch.float32) for r in expected_rewards]
        self.targets = list(targets)
        self.calls = 0

    def __call__(self, model, tokenizer, tokens, rewards):
        assert(model is self.model)
        assert(tokenizer is self.tokenizer)
        assert(torch.equal(tokens, self.expected_tokens[self.calls]))
        assert(torch.allclose(rewards.float(), self.expected_rewards[self.calls].to(rewards.device), atol=1e-6))
        target = torch.tensor(self.targets[self.calls], device=model.scale.device)
        self.calls += 1
        return (model.scale - target) ** 2


def _pad_token_rows(rows, device="cpu"):
    max_len = max(len(row) for row in rows)
    padded = [row + [0] * (max_len - len(row)) for row in rows]
    return torch.tensor(padded, dtype=torch.long, device=device)


def _manual_log_probs(logits, y, mask):
    out = []
    for i in range(logits.shape[0]):
        row_logits = logits[i][mask[i]]
        row_targets = y[i][mask[i]]
        vals = row_logits[torch.arange(row_targets.shape[0]), row_targets] - torch.logsumexp(row_logits, dim=-1)
        out.append(vals.sum())
    return torch.stack(out)


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


def test_silu(silu):
    torch.manual_seed(2)
    X = torch.randn(10, 20)
    assert(torch.allclose(silu(X), F.silu(X), atol=1e-6))

    X = torch.randn(3, 4, 5, 6)
    assert(torch.allclose(silu(X), F.silu(X), atol=1e-6))


def test_rms_norm(rms_norm):
    torch.manual_seed(3)
    X = torch.randn(100, 20)
    assert(torch.allclose(rms_norm(X), F.rms_norm(X, (20,)), atol=1e-6))

    X = torch.randn(10, 7, 20)
    assert(torch.allclose(rms_norm(X, eps=1e-3), F.rms_norm(X, (20,), eps=1e-3), atol=1e-6))


def test_self_attention(self_attention):
    torch.manual_seed(4)
    Q = torch.randn(5, 8)
    K = torch.randn(5, 8)
    V = torch.randn(5, 6)
    mask = _causal_mask(5)

    ref = F.scaled_dot_product_attention(
        Q.unsqueeze(0).unsqueeze(0),
        K.unsqueeze(0).unsqueeze(0),
        V.unsqueeze(0).unsqueeze(0),
        attn_mask=mask,
        dropout_p=0.0,
    )[0, 0]
    out = self_attention(Q, K, V, mask)
    assert(out.shape == (5, 6))
    assert(torch.allclose(out, ref, atol=1e-5))

    Q = torch.randn(2, 3, 5, 8)
    K = torch.randn(2, 3, 5, 8)
    V = torch.randn(2, 3, 5, 4)
    ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=0.0)
    out = self_attention(Q, K, V, mask)
    assert(out.shape == (2, 3, 5, 4))
    assert(torch.allclose(out, ref, atol=1e-5))


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


def test_log_probs(log_probs):
    logits = torch.tensor([
        [[2.0, 0.0, -1.0], [0.5, 1.5, -0.5], [1.0, -1.0, 0.0]],
        [[-0.5, 1.0, 0.0], [2.0, 0.0, -2.0], [0.25, 0.25, 0.25]],
    ])
    y = torch.tensor([[0, 1, 2], [1, 0, 2]])
    mask = torch.tensor([[True, False, True], [True, True, False]])
    out = log_probs(logits, y, mask)
    assert(out.shape == (2,))
    assert(torch.allclose(out, torch.tensor([-1.5775, -0.6073]), atol=1e-4))


def test_MultiHeadAttentionKVCache(MultiHeadAttentionKVCache):
    torch.manual_seed(13)
    attn = MultiHeadAttentionKVCache(12, 3, max_cache_size=8, max_cache_batches=4)
    ref_attn = nn.MultiheadAttention(12, 3, bias=False, batch_first=True)

    with torch.no_grad():
        attn.wq.weight.copy_(torch.randn_like(attn.wq.weight))
        attn.wk.weight.copy_(torch.randn_like(attn.wk.weight))
        attn.wv.weight.copy_(torch.randn_like(attn.wv.weight))
        attn.wp.weight.copy_(torch.randn_like(attn.wp.weight))
    _copy_mha_weights(attn, ref_attn)

    buffers = dict(attn.named_buffers())
    assert("k_cache" in buffers and "v_cache" in buffers)
    assert(attn.k_cache.shape == (4, 8, 12))
    assert(attn.v_cache.shape == (4, 8, 12))

    X = torch.randn(2, 5, 12)
    mask = _causal_mask(5)
    ref = ref_attn(X, X, X, attn_mask=mask, need_weights=False)[0]

    out = attn(X, mask=mask, use_kv_cache=False)
    assert(out.shape == (2, 5, 12))
    assert(torch.allclose(out, ref, atol=1e-5))

    prefix = attn(X[:, :3], mask=mask[:3, :3], seq_pos=0, use_kv_cache=True)
    tail = attn(X[:, 3:], mask=mask[3:, :], seq_pos=3, use_kv_cache=True)
    assert(torch.allclose(prefix, ref[:, :3], atol=1e-5))
    assert(torch.allclose(tail, ref[:, 3:], atol=1e-5))

    with torch.no_grad():
        K = attn.wk(X)
        V = attn.wv(X)
    assert(torch.allclose(attn.k_cache[:2, :5], K, atol=1e-6))
    assert(torch.allclose(attn.v_cache[:2, :5], V, atol=1e-6))


def submit_MultiHeadAttentionKVCache(MultiHeadAttentionKVCache):
    torch.manual_seed(14)
    layer = MultiHeadAttentionKVCache(8, 2, max_cache_size=6, max_cache_batches=3)

    def clone_layer(src):
        dst = MultiHeadAttentionKVCache(8, 2, max_cache_size=6, max_cache_batches=3)
        state = {k: v.clone() for k, v in src.state_dict().items()}
        dst.load_state_dict(state)
        return dst

    with torch.no_grad():
        layer.wq.weight.copy_(torch.arange(layer.wq.weight.numel(), dtype=torch.float32).reshape_as(layer.wq.weight) / 50)
        layer.wk.weight.copy_(torch.arange(layer.wk.weight.numel(), dtype=torch.float32).reshape_as(layer.wk.weight) / 40)
        layer.wv.weight.copy_(torch.arange(layer.wv.weight.numel(), dtype=torch.float32).reshape_as(layer.wv.weight) / 30)
        layer.wp.weight.copy_(torch.arange(layer.wp.weight.numel(), dtype=torch.float32).reshape_as(layer.wp.weight) / 20)

    X = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8) / 10
    mask = _causal_mask(4)
    mugrade.submit(layer(X, mask=mask, use_kv_cache=False).detach().numpy())

    layer = clone_layer(layer)
    layer(X[:, :2], mask=mask[:2, :2], seq_pos=0, use_kv_cache=True)
    mugrade.submit(layer(X[:, 2:], mask=mask[2:, :], seq_pos=2, use_kv_cache=True).detach().numpy())

    with torch.no_grad():
        layer = clone_layer(layer)
        layer(X[:, :3], mask=mask[:3, :3], seq_pos=0, use_kv_cache=True)
    mugrade.submit(layer.k_cache[:2, :3].detach().numpy())


def test_generate_parallel(generate_parallel):
    model = _BatchedToyGenerationModel(
        [
            [3, 4, 5],
            [6, 8, 6],
            [9, 6, 7],
        ],
        vocab_size=12,
    )

    tokens = generate_parallel(model, [1, 2], num_completions=3, eot_token=6, temp=0.7, max_tokens=5)
    assert(torch.equal(tokens.cpu(), torch.tensor([
        [1, 2, 3, 6, 9],
        [1, 2, 4, 8, 6],
        [1, 2, 5, 6, 7],
    ])))

    assert(len(model.calls) == 3)
    assert(model.calls[0][0].tolist() == [[1, 2], [1, 2], [1, 2]])
    assert(model.calls[0][1] == 0 and model.calls[0][2] is True)
    assert(model.calls[1][0].tolist() == [[3], [4], [5]])
    assert(model.calls[1][1] == 2 and model.calls[1][2] is True)
    assert(model.calls[2][0].tolist() == [[6], [8], [6]])
    assert(model.calls[2][1] == 3 and model.calls[2][2] is True)

    model = _BatchedToyGenerationModel([[4, 5], [6, 7]], vocab_size=10)
    tokens = generate_parallel(model, [1], num_completions=2, eot_token=None, temp=0.4, max_tokens=3)
    assert(torch.equal(tokens.cpu(), torch.tensor([[1, 4, 6], [1, 5, 7]])))


def submit_generate_parallel(generate_parallel):
    cases = [
        (
            _BatchedToyGenerationModel([[3, 4], [5, 6]], vocab_size=10),
            [1],
            2,
            None,
            3,
        ),
        (
            _BatchedToyGenerationModel([[3, 4, 5], [6, 7, 6], [8, 6, 9]], vocab_size=12),
            [1, 2],
            3,
            6,
            5,
        ),
        (
            _BatchedToyGenerationModel([[8], [9], [7]], vocab_size=12),
            [2, 3],
            1,
            7,
            5,
        ),
    ]

    for model, prompt, num_completions, eot_token, max_tokens in cases:
        tokens = generate_parallel(
            model,
            prompt,
            num_completions=num_completions,
            eot_token=eot_token,
            temp=0.7,
            max_tokens=max_tokens,
        )
        mugrade.submit(tokens.cpu().numpy())


def test_gsm8k_to_text(gsm8k_to_text):
    message = {
        "question": "What is 6 plus 7?",
        "answer": "First compute <<6+7=13>>.\nThen compute <<13*2=26>>.\n#### 26",
    }
    expected = (
        "<QUESTION>What is 6 plus 7?</QUESTION>"
        "<THINK>First compute <TOOL>6+7</TOOL><RESPONSE>13</RESPONSE>.\n"
        "Then compute <TOOL>13*2</TOOL><RESPONSE>26</RESPONSE>.</THINK>"
        "<ANSWER>26</ANSWER>"
    )
    assert(gsm8k_to_text(message) == expected)


def submit_gsm8k_to_text(gsm8k_to_text):
    cases = [
        {
            "question": "How much is 3 plus 4?",
            "answer": "Compute <<3+4=7>>.\n#### 7",
        },
        {
            "question": "No tool call here?",
            "answer": "Think carefully.\n#### 11",
        },
        {
            "question": "Two steps?",
            "answer": "Start <<8/2=4>> and finish <<4+9=13>>.\n#### 13",
        },
    ]
    for case in cases:
        mugrade.submit(gsm8k_to_text(case))


def test_pretokenize_gsm8k(pretokenize_gsm8k):
    messages = [
        {"question": "Q1", "answer": "Use <<2+3=5>>.\n#### 5"},
        {"question": "Q2", "answer": "No tool.\n#### 9"},
    ]
    text0 = "<QUESTION>Q1</QUESTION><THINK>Use <TOOL>2+3</TOOL><RESPONSE>5</RESPONSE>.</THINK><ANSWER>5</ANSWER>"
    text1 = "<QUESTION>Q2</QUESTION><THINK>No tool.</THINK><ANSWER>9</ANSWER>"
    tokenizer = _FixedEncodeTokenizer({
        text0: [11, 12, 13],
        text1: [21, 22],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        in_filename = os.path.join(tmpdir, "gsm8k.json")
        out_filename = os.path.join(tmpdir, "tokens.json")
        with open(in_filename, "wt") as f:
            json.dump(messages, f)

        pretokenize_gsm8k(tokenizer, in_filename, out_filename)

        with open(out_filename, "rt") as f:
            tokens = json.load(f)

    assert(tokenizer.calls == [(text0, "all"), (text1, "all")])
    assert(tokens == [[11, 12, 13], [21, 22]])


def submit_pretokenize_gsm8k(pretokenize_gsm8k):
    cases = [
        (
            [
                {"question": "A", "answer": "Do <<1+1=2>>.\n#### 2"},
                {"question": "B", "answer": "Skip tools.\n#### 4"},
            ],
            {
                "<QUESTION>A</QUESTION><THINK>Do <TOOL>1+1</TOOL><RESPONSE>2</RESPONSE>.</THINK><ANSWER>2</ANSWER>": [1, 2, 3],
                "<QUESTION>B</QUESTION><THINK>Skip tools.</THINK><ANSWER>4</ANSWER>": [4, 5],
            },
        ),
        (
            [
                {"question": "C", "answer": "First <<9-2=7>> then <<7*3=21>>.\n#### 21"},
            ],
            {
                "<QUESTION>C</QUESTION><THINK>First <TOOL>9-2</TOOL><RESPONSE>7</RESPONSE> then <TOOL>7*3</TOOL><RESPONSE>21</RESPONSE>.</THINK><ANSWER>21</ANSWER>": [6, 7, 8, 9],
            },
        ),
        (
            [
                {"question": "D", "answer": "Reasoning only.\n#### 10"},
                {"question": "E", "answer": "Compute <<6/3=2>>.\n#### 2"},
            ],
            {
                "<QUESTION>D</QUESTION><THINK>Reasoning only.</THINK><ANSWER>10</ANSWER>": [10, 11],
                "<QUESTION>E</QUESTION><THINK>Compute <TOOL>6/3</TOOL><RESPONSE>2</RESPONSE>.</THINK><ANSWER>2</ANSWER>": [12, 13, 14],
            },
        ),
    ]

    for messages, mapping in cases:
        tokenizer = _FixedEncodeTokenizer(mapping)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_filename = os.path.join(tmpdir, "gsm8k.json")
            out_filename = os.path.join(tmpdir, "tokens.json")
            with open(in_filename, "wt") as f:
                json.dump(messages, f)

            pretokenize_gsm8k(tokenizer, in_filename, out_filename)

            with open(out_filename, "rt") as f:
                tokens = json.load(f)

        mugrade.submit(tokens)


def test_get_loss_mask(get_loss_mask):
    tokenizer = _ToyGsmTokenizer()
    tokens = [
        tokenizer._special_tokens["<QUESTION>"],
        1,
        tokenizer._special_tokens["</QUESTION>"],
        tokenizer._special_tokens["<THINK>"],
        2,
        tokenizer._special_tokens["<TOOL>"],
        3,
        tokenizer._special_tokens["</TOOL>"],
        tokenizer._special_tokens["<RESPONSE>"],
        4,
        tokenizer._special_tokens["</RESPONSE>"],
        5,
        tokenizer._special_tokens["</THINK>"],
        tokenizer._special_tokens["<ANSWER>"],
        6,
        tokenizer._special_tokens["</ANSWER>"],
        7,
    ]
    expected = [
        False, False, False, False,
        True, True, True, True,
        False, False, False,
        True, True, True, True, True,
        False,
    ]
    assert(get_loss_mask(tokens, tokenizer) == expected)


def submit_get_loss_mask(get_loss_mask):
    tokenizer = _ToyGsmTokenizer()
    cases = [
        [
            tokenizer._special_tokens["<QUESTION>"],
            1,
            tokenizer._special_tokens["</QUESTION>"],
            tokenizer._special_tokens["<THINK>"],
            2,
            tokenizer._special_tokens["<ANSWER>"],
            3,
            tokenizer._special_tokens["</ANSWER>"],
            4,
        ],
        [
            tokenizer._special_tokens["<THINK>"],
            5,
            tokenizer._special_tokens["<TOOL>"],
            6,
            tokenizer._special_tokens["</TOOL>"],
            tokenizer._special_tokens["<RESPONSE>"],
            7,
            tokenizer._special_tokens["</RESPONSE>"],
            8,
        ],
        [
            tokenizer._special_tokens["<QUESTION>"],
            9,
            tokenizer._special_tokens["</QUESTION>"],
            tokenizer._special_tokens["<THINK>"],
            10,
            tokenizer._special_tokens["<TOOL>"],
            11,
            tokenizer._special_tokens["</TOOL>"],
            tokenizer._special_tokens["<RESPONSE>"],
            12,
            tokenizer._special_tokens["</RESPONSE>"],
            13,
            tokenizer._special_tokens["</THINK>"],
            tokenizer._special_tokens["<ANSWER>"],
            14,
            tokenizer._special_tokens["</ANSWER>"],
        ],
    ]
    for tokens in cases:
        mugrade.submit(get_loss_mask(tokens, tokenizer))


def test_DataLoader(DataLoader):
    tokenizer = _ToyGsmTokenizer()
    chats = [
        [1, tokenizer._special_tokens["<THINK>"], 10, 11, tokenizer._special_tokens["</ANSWER>"]],
        [2, tokenizer._special_tokens["<THINK>"], 12, tokenizer._special_tokens["</ANSWER>"]],
        [3, tokenizer._special_tokens["<THINK>"], 13, 14, 15, tokenizer._special_tokens["</ANSWER>"]],
        [4, tokenizer._special_tokens["<THINK>"], 16, tokenizer._special_tokens["</ANSWER>"]],
        [5, tokenizer._special_tokens["<THINK>"], 17, 18, tokenizer._special_tokens["</ANSWER>"]],
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "gsm_tokens.json")
        with open(filename, "wt") as f:
            json.dump(chats, f)

        loader = DataLoader(filename, batch_size=2, tokenizer=tokenizer)
        assert(iter(loader) is loader)
        batches = list(loader)
        assert(len(batches) == 2)

        X0, y0, m0 = batches[0]
        assert(torch.equal(X0.cpu(), torch.tensor([
            [1, 93, 10, 11],
            [2, 93, 12, 100],
        ])))
        assert(torch.equal(y0.cpu(), torch.tensor([
            [93, 10, 11, 100],
            [93, 12, 100, 0],
        ])))
        assert(torch.equal(m0.cpu(), torch.tensor([
            [False, True, True, True],
            [False, True, True, False],
        ])))

        X1, y1, m1 = batches[1]
        assert(torch.equal(X1.cpu(), torch.tensor([
            [3, 93, 13, 14, 15],
            [4, 93, 16, 100, 0],
        ])))
        assert(torch.equal(y1.cpu(), torch.tensor([
            [93, 13, 14, 15, 100],
            [93, 16, 100, 0, 0],
        ])))
        assert(torch.equal(m1.cpu(), torch.tensor([
            [False, True, True, True, True],
            [False, True, True, False, False],
        ])))

        batches2 = list(loader)
        for (X_a, y_a, m_a), (X_b, y_b, m_b) in zip(batches, batches2):
            assert(torch.equal(X_a, X_b))
            assert(torch.equal(y_a, y_b))
            assert(torch.equal(m_a, m_b))


def submit_DataLoader(DataLoader):
    tokenizer = _ToyGsmTokenizer()
    cases = [
        (
            [
                [1, tokenizer._special_tokens["<THINK>"], 10, tokenizer._special_tokens["</ANSWER>"]],
                [2, tokenizer._special_tokens["<THINK>"], 11, 12, tokenizer._special_tokens["</ANSWER>"]],
            ],
            2,
            0,
        ),
        (
            [
                [3, tokenizer._special_tokens["<THINK>"], 13, 14, tokenizer._special_tokens["</ANSWER>"]],
                [4, tokenizer._special_tokens["<THINK>"], 15, tokenizer._special_tokens["</ANSWER>"]],
                [5, tokenizer._special_tokens["<THINK>"], 16, 17, 18, tokenizer._special_tokens["</ANSWER>"]],
                [6, tokenizer._special_tokens["<THINK>"], 19, tokenizer._special_tokens["</ANSWER>"]],
            ],
            2,
            1,
        ),
        (
            [
                [7, tokenizer._special_tokens["<THINK>"], 20, 21, 22, tokenizer._special_tokens["</ANSWER>"]],
            ],
            1,
            0,
        ),
    ]

    for chats, batch_size, batch_idx in cases:
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "gsm_tokens.json")
            with open(filename, "wt") as f:
                json.dump(chats, f)

            batches = list(DataLoader(filename, batch_size=batch_size, tokenizer=tokenizer))

        X, y, mask = batches[batch_idx]
        mugrade.submit(np.array([len(batches)], dtype=np.int64))
        mugrade.submit(X.cpu().numpy())
        mugrade.submit(mask.cpu().numpy())


def test_train_llm_sft(train_llm_sft):
    batches = [
        (
            torch.tensor([[0, 1, 2], [1, 2, 3]]),
            torch.tensor([[1, 2, 3], [2, 3, 4]]),
            torch.tensor([[False, True, True], [True, False, True]]),
        ),
        (
            torch.tensor([[2, 3, 4], [0, 2, 4]]),
            torch.tensor([[3, 4, 0], [2, 4, 1]]),
            torch.tensor([[True, True, False], [False, True, True]]),
        ),
    ]

    torch.manual_seed(0)
    model = _TinyTrainModel(vocab_size=6, dim=4)
    init = copy.deepcopy(model.state_dict())
    opt = optim.SGD(model.parameters(), lr=0.1)
    spy = _MaskedCrossEntropySpy(model, batches)

    old_loss = train_llm_sft.__globals__["cross_entropy_loss"]
    try:
        train_llm_sft.__globals__["cross_entropy_loss"] = spy
        with redirect_stdout(io.StringIO()):
            train_llm_sft(model, batches, opt, max_iter=1)
    finally:
        train_llm_sft.__globals__["cross_entropy_loss"] = old_loss

    assert(spy.calls == 1)
    assert(any(not torch.allclose(v, init[k]) for k, v in model.state_dict().items()))

    torch.manual_seed(0)
    model = _TinyTrainModel(vocab_size=6, dim=4)
    opt = optim.SGD(model.parameters(), lr=0.1)
    spy = _MaskedCrossEntropySpy(model, batches)
    old_loss = train_llm_sft.__globals__["cross_entropy_loss"]
    try:
        train_llm_sft.__globals__["cross_entropy_loss"] = spy
        with redirect_stdout(io.StringIO()):
            train_llm_sft(model, batches, opt)
    finally:
        train_llm_sft.__globals__["cross_entropy_loss"] = old_loss

    assert(spy.calls == 2)


def submit_train_llm_sft(train_llm_sft):
    def run_case(seed, loader, eval_tokens, max_iter=None):
        torch.manual_seed(seed)
        model = _TinyTrainModel(vocab_size=6, dim=4)
        opt = optim.SGD(model.parameters(), lr=0.1)
        old_loss = train_llm_sft.__globals__["cross_entropy_loss"]
        try:
            train_llm_sft.__globals__["cross_entropy_loss"] = lambda logits, y: F.cross_entropy(logits, y)
            with redirect_stdout(io.StringIO()):
                train_llm_sft(model, loader, opt, max_iter=max_iter)
        finally:
            train_llm_sft.__globals__["cross_entropy_loss"] = old_loss
        return model(eval_tokens).detach().numpy()

    cases = [
        (
            1,
            [
                (
                    torch.tensor([[0, 1, 2], [1, 2, 3]]),
                    torch.tensor([[1, 2, 3], [2, 3, 4]]),
                    torch.tensor([[False, True, True], [True, False, True]]),
                ),
            ],
            torch.tensor([[0, 1, 2]]),
            1,
        ),
        (
            2,
            [
                (
                    torch.tensor([[0, 1, 2], [1, 2, 3]]),
                    torch.tensor([[1, 2, 3], [2, 3, 4]]),
                    torch.tensor([[False, True, True], [True, False, True]]),
                ),
                (
                    torch.tensor([[2, 3, 4], [0, 2, 4]]),
                    torch.tensor([[3, 4, 0], [2, 4, 1]]),
                    torch.tensor([[True, True, False], [False, True, True]]),
                ),
            ],
            torch.tensor([[2, 3, 4]]),
            None,
        ),
        (
            3,
            [
                (
                    torch.tensor([[4, 3, 2], [1, 0, 5]]),
                    torch.tensor([[3, 2, 1], [0, 5, 4]]),
                    torch.tensor([[True, False, True], [False, True, True]]),
                ),
                (
                    torch.tensor([[5, 4, 3], [2, 1, 0]]),
                    torch.tensor([[4, 3, 2], [1, 0, 5]]),
                    torch.tensor([[True, True, False], [True, False, True]]),
                ),
            ],
            torch.tensor([[1, 0, 5]]),
            2,
        ),
    ]

    for seed, loader, eval_tokens, max_iter in cases:
        mugrade.submit(run_case(seed, loader, eval_tokens, max_iter=max_iter))


def test_eval_tool(eval_tool):
    assert(eval_tool("6/3") == 2)
    assert(eval_tool("7/2") == 3.5)
    assert(eval_tool("1/0") == "ERROR")
    assert(eval_tool("3.00001") == 3)


def submit_eval_tool(eval_tool):
    for expr in ["48/2", "7/4", "bad + 1"]:
        mugrade.submit(eval_tool(expr))


def test_generate_parallel_tool(generate_parallel_tool):
    tokenizer = _CharSpecialTokenizer()
    prompt_text = "<QUESTION>Q</QUESTION><THINK>"
    prompt = tokenizer.encode(prompt_text)
    tool_open = tokenizer._special_tokens["<TOOL>"]
    tool_close = tokenizer._special_tokens["</TOOL>"]
    answer_open = tokenizer._special_tokens["<ANSWER>"]
    answer_close = tokenizer._special_tokens["</ANSWER>"]
    think_close = tokenizer._special_tokens["</THINK>"]
    placeholder = tokenizer.encode("x")[0]

    model = _BatchedToyGenerationModel(
        [
            [tokenizer.encode("2")[0], tokenizer.encode("7")[0]],
            [tokenizer.encode("+")[0], think_close],
            [tokenizer.encode("3")[0], answer_open],
            [tokenizer.encode("=")[0], tokenizer.encode("7")[0]],
            [tool_open, answer_close],
            [tokenizer.encode("2")[0], placeholder],
            [tokenizer.encode("+")[0], placeholder],
            [tokenizer.encode("3")[0], placeholder],
            [tool_close, placeholder],
            [placeholder, placeholder],
            [placeholder, placeholder],
            [placeholder, placeholder],
            [tokenizer.encode(".")[0], placeholder],
            [think_close, placeholder],
            [answer_open, placeholder],
            [tokenizer.encode("5")[0], placeholder],
            [answer_close, placeholder],
        ],
        vocab_size=tokenizer._next_id + 8,
    )

    calls = []

    def fake_eval_tool(text):
        calls.append(text)
        return 5

    old_eval_tool = generate_parallel_tool.__globals__["eval_tool"]
    try:
        generate_parallel_tool.__globals__["eval_tool"] = fake_eval_tool
        tokens = generate_parallel_tool(
            model,
            prompt,
            tokenizer,
            num_completions=2,
            eot_token=answer_close,
            temp=0.7,
            max_tokens=40,
        )
    finally:
        generate_parallel_tool.__globals__["eval_tool"] = old_eval_tool

    row1 = tokenizer.encode(
        prompt_text + "2+3=<TOOL>2+3</TOOL><RESPONSE>5</RESPONSE>.</THINK><ANSWER>5</ANSWER>"
    )
    row2 = tokenizer.encode(prompt_text + "7</THINK><ANSWER>7</ANSWER>")
    expected = _pad_token_rows([
        row1,
        row2,
    ], device=tokens.device)
    assert(torch.equal(tokens[:, : expected.shape[1]], expected))
    assert(torch.equal(tokens[1, len(row2):], torch.zeros(tokens.shape[1] - len(row2), dtype=torch.long, device=tokens.device)))
    assert(calls == ["2+3"])
    assert(len(model.calls) == 17)
    assert(model.calls[0][0].tolist() == [prompt, prompt])
    assert(model.calls[0][1] == 0 and model.calls[0][2] is True)

    # Check that eot_token is respected as a passed argument, not just for </ANSWER>.
    model = _BatchedToyGenerationModel(
        [
            [tokenizer.encode("a")[0], tokenizer.encode("b")[0]],
            [think_close, tokenizer.encode("c")[0]],
            [placeholder, think_close],
        ],
        vocab_size=tokenizer._next_id + 8,
    )
    tokens = generate_parallel_tool(
        model,
        prompt,
        tokenizer,
        num_completions=2,
        eot_token=think_close,
        temp=0.7,
        max_tokens=20,
    )
    expected = _pad_token_rows([
        tokenizer.encode(prompt_text + "a</THINK>x"),
        tokenizer.encode(prompt_text + "bc</THINK>"),
    ], device=tokens.device)
    assert(torch.equal(tokens[:, : expected.shape[1]], expected))
    assert(torch.equal(tokens[:, expected.shape[1]:], torch.zeros_like(tokens[:, expected.shape[1]:])))
    assert(len(model.calls) == 3)


def submit_generate_parallel_tool(generate_parallel_tool):
    tokenizer = _CharSpecialTokenizer()
    tool_open = tokenizer._special_tokens["<TOOL>"]
    tool_close = tokenizer._special_tokens["</TOOL>"]
    answer_open = tokenizer._special_tokens["<ANSWER>"]
    answer_close = tokenizer._special_tokens["</ANSWER>"]
    think_close = tokenizer._special_tokens["</THINK>"]
    placeholder = tokenizer.encode("x")[0]

    cases = [
        (
            _BatchedToyGenerationModel(
                [
                    [tokenizer.encode("4")[0]],
                    [tokenizer.encode("+")[0]],
                    [tokenizer.encode("5")[0]],
                    [tokenizer.encode("=")[0]],
                    [tool_open],
                    [tokenizer.encode("4")[0]],
                    [tokenizer.encode("+")[0]],
                    [tokenizer.encode("5")[0]],
                    [tool_close],
                    [placeholder],
                    [placeholder],
                    [placeholder],
                    [tokenizer.encode(".")[0]],
                    [think_close],
                    [answer_open],
                    [tokenizer.encode("9")[0]],
                    [answer_close],
                ],
                vocab_size=tokenizer._next_id + 8,
            ),
            tokenizer.encode("<QUESTION>A</QUESTION><THINK>"),
            answer_close,
        ),
        (
            _BatchedToyGenerationModel(
                [
                    [tool_open],
                    [tokenizer.encode("1")[0]],
                    [tokenizer.encode("/")[0]],
                    [tokenizer.encode("0")[0]],
                    [tool_close],
                    [placeholder],
                    [placeholder],
                    [placeholder],
                    [placeholder],
                    [placeholder],
                    [placeholder],
                    [placeholder],
                    [tokenizer.encode(".")[0]],
                    [think_close],
                    [answer_open],
                    [tokenizer.encode("0")[0]],
                    [answer_close],
                ],
                vocab_size=tokenizer._next_id + 8,
            ),
            tokenizer.encode("<QUESTION>B</QUESTION><THINK>"),
            answer_close,
        ),
        (
            _BatchedToyGenerationModel(
                [
                    [tokenizer.encode("8")[0], tokenizer.encode("8")[0]],
                    [tokenizer.encode("-")[0], think_close],
                    [tokenizer.encode("5")[0], answer_open],
                    [tokenizer.encode("=")[0], tokenizer.encode("8")[0]],
                    [tool_open, answer_close],
                    [tokenizer.encode("8")[0], placeholder],
                    [tokenizer.encode("-")[0], placeholder],
                    [tokenizer.encode("5")[0], placeholder],
                    [tool_close, placeholder],
                    [placeholder, placeholder],
                    [placeholder, placeholder],
                    [placeholder, placeholder],
                    [tokenizer.encode(".")[0], placeholder],
                    [think_close, placeholder],
                    [answer_open, placeholder],
                    [tokenizer.encode("3")[0], placeholder],
                    [answer_close, placeholder],
                ],
                vocab_size=tokenizer._next_id + 8,
            ),
            tokenizer.encode("<QUESTION>C</QUESTION><THINK>"),
            answer_close,
            2,
        ),
        (
            _BatchedToyGenerationModel(
                [
                    [tokenizer.encode("a")[0], tokenizer.encode("b")[0]],
                    [think_close, tokenizer.encode("c")[0]],
                    [placeholder, think_close],
                ],
                vocab_size=tokenizer._next_id + 8,
            ),
            tokenizer.encode("<QUESTION>D</QUESTION><THINK>"),
            think_close,
            2,
        ),
    ]

    for case in cases:
        if len(case) == 4:
            model, prompt, eot_token, num_completions = case
        else:
            model, prompt, eot_token = case
            num_completions = 1
        tokens = generate_parallel_tool(
            model,
            prompt,
            tokenizer,
            num_completions=num_completions,
            eot_token=eot_token,
            temp=0.7,
            max_tokens=48,
        )
        mugrade.submit(tokens.cpu().numpy())
        mugrade.submit(np.array([len(model.calls)], dtype=np.int64))
        mugrade.submit(np.count_nonzero(tokens.cpu().numpy(), axis=1))


def test_extract_answer(extract_answer):
    tokenizer = _CharSpecialTokenizer()
    assert(extract_answer(tokenizer, tokenizer.encode("<THINK>x</THINK><ANSWER>12</ANSWER>")) == 12)
    assert(extract_answer(tokenizer, tokenizer.encode("<ANSWER>-7</ANSWER>")) == -7)
    assert(extract_answer(tokenizer, tokenizer.encode("<ANSWER>oops</ANSWER>")) is None)
    assert(extract_answer(tokenizer, tokenizer.encode("<THINK>no answer</THINK>")) is None)


def submit_extract_answer(extract_answer):
    tokenizer = _CharSpecialTokenizer()
    mugrade.submit(extract_answer(tokenizer, tokenizer.encode("<ANSWER>101</ANSWER>")))
    mugrade.submit(extract_answer(tokenizer, tokenizer.encode("<QUESTION>Q</QUESTION><ANSWER>-4</ANSWER>")))
    mugrade.submit(extract_answer(tokenizer, tokenizer.encode("<ANSWER>bad</ANSWER>")) is None)


def test_grade_responses(grade_responses):
    tokenizer = _CharSpecialTokenizer()
    rows = _pad_token_rows([
        tokenizer.encode("<THINK>a</THINK><ANSWER>9</ANSWER>"),
        tokenizer.encode("<THINK>b</THINK><ANSWER>4</ANSWER>"),
        tokenizer.encode("<THINK>c</THINK>"),
    ])
    ground_truth = torch.tensor(tokenizer.encode("<ANSWER>9</ANSWER>"))
    scores = grade_responses(tokenizer, rows, ground_truth, correct_weight=1.0, format_weight=0.2)
    assert(scores == [1.2, 0.2, 0.0])


def submit_grade_responses(grade_responses):
    tokenizer = _CharSpecialTokenizer()
    cases = [
        (
            _pad_token_rows([
                tokenizer.encode("<ANSWER>5</ANSWER>"),
                tokenizer.encode("<ANSWER>3</ANSWER>"),
            ]),
            torch.tensor(tokenizer.encode("<ANSWER>5</ANSWER>")),
            1.0,
            0.1,
        ),
        (
            _pad_token_rows([
                tokenizer.encode("<THINK>x</THINK>"),
                tokenizer.encode("<ANSWER>-2</ANSWER>"),
                tokenizer.encode("<ANSWER>-2</ANSWER>"),
            ]),
            torch.tensor(tokenizer.encode("<ANSWER>-2</ANSWER>")),
            2.0,
            0.5,
        ),
        (
            _pad_token_rows([
                tokenizer.encode("<ANSWER>8</ANSWER>"),
                tokenizer.encode("<ANSWER>oops</ANSWER>"),
            ]),
            torch.tensor(tokenizer.encode("<ANSWER>7</ANSWER>")),
            1.5,
            0.25,
        ),
    ]
    for tokens, ground_truth, correct_weight, format_weight in cases:
        mugrade.submit(grade_responses(tokenizer, tokens, ground_truth, correct_weight=correct_weight, format_weight=format_weight))


def test_eval_model(eval_model):
    tokenizer = _CharSpecialTokenizer()
    prompt1 = tokenizer.encode("<QUESTION>Q1</QUESTION><THINK>")
    prompt2 = tokenizer.encode("<QUESTION>Q2</QUESTION><THINK>")
    loader = _BatchSizeOneLoader([
        (
            torch.tensor([prompt1 + tokenizer.encode("abc")]),
            torch.tensor([tokenizer.encode("<ANSWER>5</ANSWER>")]),
            torch.tensor([[False] * (len(prompt1) + 3)]),
        ),
        (
            torch.tensor([prompt2 + tokenizer.encode("xyz")]),
            torch.tensor([tokenizer.encode("<ANSWER>3</ANSWER>")]),
            torch.tensor([[False] * (len(prompt2) + 3)]),
        ),
    ])

    generator = _SequenceGenerator([
        _pad_token_rows([
            prompt1 + tokenizer.encode("2+3=<TOOL>2+3</TOOL><RESPONSE>5</RESPONSE>.</THINK><ANSWER>5</ANSWER>"),
            prompt1 + tokenizer.encode("4.</THINK><ANSWER>4</ANSWER>"),
            prompt1 + tokenizer.encode("oops"),
        ]),
        _pad_token_rows([
            prompt2 + tokenizer.encode("1.</THINK><ANSWER>1</ANSWER>"),
            prompt2 + tokenizer.encode("bad"),
            prompt2 + tokenizer.encode("still bad"),
        ]),
    ])

    old_generate = eval_model.__globals__["generate_parallel_tool"]
    try:
        eval_model.__globals__["generate_parallel_tool"] = generator
        correct, formatted, passk = eval_model(
            loader,
            model=object(),
            tokenizer=tokenizer,
            num_completions=3,
            max_tokens=20,
            temp=0.6,
            max_cases=2,
        )
    finally:
        eval_model.__globals__["generate_parallel_tool"] = old_generate

    assert(math.isclose(correct, 1 / 6))
    assert(math.isclose(formatted, 0.5))
    assert(math.isclose(passk, 0.5))
    assert(generator.calls[0][0] == prompt1)
    assert(generator.calls[1][0] == prompt2)


def submit_eval_model(eval_model):
    tokenizer = _CharSpecialTokenizer()
    prompt1 = tokenizer.encode("<QUESTION>A</QUESTION><THINK>")
    prompt2 = tokenizer.encode("<QUESTION>B</QUESTION><THINK>")
    prompt3 = tokenizer.encode("<QUESTION>C</QUESTION><THINK>")

    cases = [
        (
            _BatchSizeOneLoader([
                (
                    torch.tensor([prompt1 + tokenizer.encode("x")]),
                    torch.tensor([tokenizer.encode("<ANSWER>2</ANSWER>")]),
                    torch.tensor([[False] * (len(prompt1) + 1)]),
                ),
            ]),
            [
                _pad_token_rows([
                    prompt1 + tokenizer.encode("1+1=<TOOL>1+1</TOOL><RESPONSE>2</RESPONSE>.</THINK><ANSWER>2</ANSWER>"),
                    prompt1 + tokenizer.encode("9.</THINK><ANSWER>9</ANSWER>"),
                ]),
            ],
            2,
            1,
        ),
        (
            _BatchSizeOneLoader([
                (
                    torch.tensor([prompt1 + tokenizer.encode("x")]),
                    torch.tensor([tokenizer.encode("<ANSWER>2</ANSWER>")]),
                    torch.tensor([[False] * (len(prompt1) + 1)]),
                ),
                (
                    torch.tensor([prompt2 + tokenizer.encode("y")]),
                    torch.tensor([tokenizer.encode("<ANSWER>3</ANSWER>")]),
                    torch.tensor([[False] * (len(prompt2) + 1)]),
                ),
                (
                    torch.tensor([prompt3 + tokenizer.encode("z")]),
                    torch.tensor([tokenizer.encode("<ANSWER>4</ANSWER>")]),
                    torch.tensor([[False] * (len(prompt3) + 1)]),
                ),
            ]),
            [
                _pad_token_rows([prompt1 + tokenizer.encode("2.</THINK><ANSWER>2</ANSWER>"), prompt1 + tokenizer.encode("bad")]),
                _pad_token_rows([prompt2 + tokenizer.encode("1.</THINK><ANSWER>1</ANSWER>"), prompt2 + tokenizer.encode("3.</THINK><ANSWER>3</ANSWER>")]),
                _pad_token_rows([prompt3 + tokenizer.encode("bad"), prompt3 + tokenizer.encode("bad")]),
            ],
            2,
            2,
        ),
        (
            _BatchSizeOneLoader([
                (
                    torch.tensor([prompt1 + tokenizer.encode("x")]),
                    torch.tensor([tokenizer.encode("<ANSWER>2</ANSWER>")]),
                    torch.tensor([[False] * (len(prompt1) + 1)]),
                ),
                (
                    torch.tensor([prompt2 + tokenizer.encode("y")]),
                    torch.tensor([tokenizer.encode("<ANSWER>3</ANSWER>")]),
                    torch.tensor([[False] * (len(prompt2) + 1)]),
                ),
            ]),
            [
                _pad_token_rows([prompt1 + tokenizer.encode("2.</THINK><ANSWER>2</ANSWER>")]),
                _pad_token_rows([prompt2 + tokenizer.encode("3.</THINK><ANSWER>3</ANSWER>")]),
            ],
            1,
            1,
        ),
    ]

    for loader, outputs, num_completions, max_cases in cases:
        generator = _SequenceGenerator(outputs)
        old_generate = eval_model.__globals__["generate_parallel_tool"]
        try:
            eval_model.__globals__["generate_parallel_tool"] = generator
            metrics = eval_model(
                loader,
                model=object(),
                tokenizer=tokenizer,
                num_completions=num_completions,
                max_tokens=20,
                temp=0.6,
                max_cases=max_cases,
            )
        finally:
            eval_model.__globals__["generate_parallel_tool"] = old_generate
        mugrade.submit(np.array(metrics, dtype=np.float32))


def test_rl_loss(rl_loss):
    tokenizer = _CharSpecialTokenizer()
    tokens = torch.tensor([
        tokenizer.encode("<QUESTION>Q</QUESTION><THINK>a</THINK><ANSWER>5</ANSWER>"),
        tokenizer.encode("<QUESTION>R</QUESTION><THINK>b</THINK><ANSWER>2</ANSWER>"),
    ])
    rewards = torch.tensor([1.5, -0.5])
    vocab_size = max(tokens.max().item() + 3, 32)
    table = torch.linspace(-1.0, 1.0, steps=vocab_size * vocab_size, dtype=torch.float32).reshape(vocab_size, vocab_size)
    model = _LookupLM(table)

    loss = rl_loss(model, tokenizer, tokens, rewards)
    get_loss_mask = rl_loss.__globals__["get_loss_mask"]
    mask = torch.tensor([get_loss_mask(row.tolist(), tokenizer) for row in tokens], dtype=torch.bool)
    logits = model(tokens[:, :-1])
    logp = _manual_log_probs(logits, tokens[:, 1:], mask[:, 1:])
    ref = -(logp * (rewards - rewards.mean())).sum() / mask.sum()
    assert(torch.allclose(loss, ref, atol=1e-6))

    loss.backward()
    assert(model.table.grad is not None)


def submit_rl_loss(rl_loss):
    tokenizer = _CharSpecialTokenizer()

    def run_case(text_rows, rewards):
        token_rows = [tokenizer.encode(text) for text in text_rows]
        tokens = torch.tensor(token_rows)
        vocab_size = max(tokens.max().item() + 5, 40)
        table = torch.arange(vocab_size * vocab_size, dtype=torch.float32).reshape(vocab_size, vocab_size) / 50
        model = _LookupLM(table)
        return rl_loss(model, tokenizer, tokens, torch.tensor(rewards, dtype=torch.float32)).detach().numpy()

    cases = [
        (
            [
                "<QUESTION>Q</QUESTION><THINK>a</THINK><ANSWER>5</ANSWER>",
                "<QUESTION>R</QUESTION><THINK>b</THINK><ANSWER>2</ANSWER>",
            ],
            [1.0, 0.0],
        ),
        (
            [
                "<QUESTION>X</QUESTION><THINK>c</THINK><ANSWER>9</ANSWER>",
                "<QUESTION>Y</QUESTION><THINK>d</THINK><ANSWER>4</ANSWER>",
            ],
            [0.25, 1.75],
        ),
        (
            [
                "<QUESTION>M</QUESTION><THINK>e</THINK><ANSWER>1</ANSWER>",
                "<QUESTION>N</QUESTION><THINK>f</THINK><ANSWER>3</ANSWER>",
            ],
            [-1.0, 2.0],
        ),
    ]

    for text_rows, rewards in cases:
        mugrade.submit(run_case(text_rows, rewards))


def test_train_llm_rl(train_llm_rl):
    tokenizer = _CharSpecialTokenizer()
    prompt1 = tokenizer.encode("<QUESTION>Q1</QUESTION><THINK>")
    prompt2 = tokenizer.encode("<QUESTION>Q2</QUESTION><THINK>")
    ground_truth1 = torch.tensor(tokenizer.encode("<ANSWER>5</ANSWER>"))
    ground_truth2 = torch.tensor(tokenizer.encode("<ANSWER>3</ANSWER>"))

    outputs = [
        _pad_token_rows([
            prompt1 + tokenizer.encode("2+3=<TOOL>2+3</TOOL><RESPONSE>5</RESPONSE>.</THINK><ANSWER>5</ANSWER>"),
            prompt1 + tokenizer.encode("4.</THINK><ANSWER>4</ANSWER>"),
        ]),
        _pad_token_rows([
            prompt2 + tokenizer.encode("bad"),
            prompt2 + tokenizer.encode("1+2=<TOOL>1+2</TOOL><RESPONSE>3</RESPONSE>.</THINK><ANSWER>3</ANSWER>"),
        ]),
    ]

    loader = _BatchSizeOneLoader([
        (torch.tensor([prompt1 + tokenizer.encode("x")]), torch.tensor([ground_truth1.tolist()]), torch.tensor([[False] * (len(prompt1) + 1)])),
        (torch.tensor([prompt2 + tokenizer.encode("y")]), torch.tensor([ground_truth2.tolist()]), torch.tensor([[False] * (len(prompt2) + 1)])),
    ])

    model = _ScalarModel(0.0)
    opt = optim.SGD(model.parameters(), lr=0.1)
    generator = _SequenceGenerator(outputs)
    spy = _RlLossSpy(model, tokenizer, outputs, [[1.25, 0.25], [0.0, 1.25]], targets=[1.0, -0.5])

    old_generate = train_llm_rl.__globals__["generate_parallel_tool"]
    old_rl_loss = train_llm_rl.__globals__["rl_loss"]
    try:
        train_llm_rl.__globals__["generate_parallel_tool"] = generator
        train_llm_rl.__globals__["rl_loss"] = spy
        with redirect_stdout(io.StringIO()):
            train_llm_rl(model, loader, opt, tokenizer, num_completions=2, max_iter=1, format_weight=0.25)
    finally:
        train_llm_rl.__globals__["generate_parallel_tool"] = old_generate
        train_llm_rl.__globals__["rl_loss"] = old_rl_loss

    assert(spy.calls == 1)
    assert(generator.calls[0][0] == prompt1)
    assert(not torch.allclose(model.scale, torch.tensor(0.0)))

    model = _ScalarModel(0.0)
    opt = optim.SGD(model.parameters(), lr=0.1)
    generator = _SequenceGenerator(outputs)
    spy = _RlLossSpy(model, tokenizer, outputs, [[1.25, 0.25], [0.0, 1.25]], targets=[1.0, -0.5])
    old_generate = train_llm_rl.__globals__["generate_parallel_tool"]
    old_rl_loss = train_llm_rl.__globals__["rl_loss"]
    try:
        train_llm_rl.__globals__["generate_parallel_tool"] = generator
        train_llm_rl.__globals__["rl_loss"] = spy
        with redirect_stdout(io.StringIO()):
            train_llm_rl(model, loader, opt, tokenizer, num_completions=2, format_weight=0.25)
    finally:
        train_llm_rl.__globals__["generate_parallel_tool"] = old_generate
        train_llm_rl.__globals__["rl_loss"] = old_rl_loss

    assert(spy.calls == 2)


def submit_train_llm_rl(train_llm_rl):
    tokenizer = _CharSpecialTokenizer()

    def run_case(outputs, ground_truths, targets, max_iter=None, init=0.0):
        loader_batches = []
        expected_rewards = []
        for i, (tokens, answer_text) in enumerate(zip(outputs, ground_truths)):
            prompt = tokens[0].tolist()
            prompt = prompt[: prompt.index(tokenizer._special_tokens["<THINK>"]) + 1]
            x = torch.tensor([prompt + tokenizer.encode(chr(ord("a") + i))])
            y = torch.tensor([tokenizer.encode(answer_text)])
            loader_batches.append((x, y, torch.zeros_like(x, dtype=torch.bool)))
            rewards = []
            correct_answer = int(answer_text.split(">")[1].split("<")[0])
            for row in tokens:
                text = tokenizer.decode(row.tolist())
                if f"<ANSWER>{correct_answer}</ANSWER>" in text:
                    rewards.append(1.25)
                elif "<ANSWER>" in text and "</ANSWER>" in text:
                    rewards.append(0.25)
                else:
                    rewards.append(0.0)
            expected_rewards.append(rewards)

        loader = _BatchSizeOneLoader(loader_batches)
        model = _ScalarModel(init)
        opt = optim.SGD(model.parameters(), lr=0.1)
        generator = _SequenceGenerator(outputs)
        spy = _RlLossSpy(model, tokenizer, outputs, expected_rewards, targets=targets)

        old_generate = train_llm_rl.__globals__["generate_parallel_tool"]
        old_rl_loss = train_llm_rl.__globals__["rl_loss"]
        try:
            train_llm_rl.__globals__["generate_parallel_tool"] = generator
            train_llm_rl.__globals__["rl_loss"] = spy
            with redirect_stdout(io.StringIO()):
                train_llm_rl(model, loader, opt, tokenizer, num_completions=outputs[0].shape[0], max_iter=max_iter, format_weight=0.25)
        finally:
            train_llm_rl.__globals__["generate_parallel_tool"] = old_generate
            train_llm_rl.__globals__["rl_loss"] = old_rl_loss
        return np.array([model.scale.item(), spy.calls], dtype=np.float32)

    prompt1 = tokenizer.encode("<QUESTION>A</QUESTION><THINK>")
    prompt2 = tokenizer.encode("<QUESTION>B</QUESTION><THINK>")

    cases = [
        (
            [
                _pad_token_rows([
                    prompt1 + tokenizer.encode("2+3=<TOOL>2+3</TOOL><RESPONSE>5</RESPONSE>.</THINK><ANSWER>5</ANSWER>"),
                    prompt1 + tokenizer.encode("4.</THINK><ANSWER>4</ANSWER>"),
                ]),
            ],
            ["<ANSWER>5</ANSWER>"],
            [1.0],
            1,
            0.0,
        ),
        (
            [
                _pad_token_rows([
                    prompt1 + tokenizer.encode("2+3=<TOOL>2+3</TOOL><RESPONSE>5</RESPONSE>.</THINK><ANSWER>5</ANSWER>"),
                    prompt1 + tokenizer.encode("bad"),
                ]),
                _pad_token_rows([
                    prompt2 + tokenizer.encode("1+2=<TOOL>1+2</TOOL><RESPONSE>3</RESPONSE>.</THINK><ANSWER>3</ANSWER>"),
                    prompt2 + tokenizer.encode("1.</THINK><ANSWER>1</ANSWER>"),
                ]),
            ],
            ["<ANSWER>5</ANSWER>", "<ANSWER>3</ANSWER>"],
            [0.5, -0.5],
            None,
            0.0,
        ),
        (
            [
                _pad_token_rows([
                    prompt1 + tokenizer.encode("5+3=<TOOL>5+3</TOOL><RESPONSE>8</RESPONSE>.</THINK><ANSWER>8</ANSWER>"),
                    prompt1 + tokenizer.encode("2.</THINK><ANSWER>2</ANSWER>"),
                ]),
                _pad_token_rows([
                    prompt2 + tokenizer.encode("oops"),
                    prompt2 + tokenizer.encode("3+3=<TOOL>3+3</TOOL><RESPONSE>6</RESPONSE>.</THINK><ANSWER>6</ANSWER>"),
                ]),
            ],
            ["<ANSWER>8</ANSWER>", "<ANSWER>6</ANSWER>"],
            [0.25, 0.75],
            2,
            0.3,
        ),
    ]

    for outputs, ground_truths, targets, max_iter, init in cases:
        mugrade.submit(run_case(outputs, ground_truths, targets, max_iter=max_iter, init=init))
