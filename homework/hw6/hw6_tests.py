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


class _FixedEncodeTokenizer:
    def __init__(self, outputs):
        self.outputs = {k: list(v) for k, v in outputs.items()}
        self.calls = []

    def encode(self, text, allowed_special="all"):
        self.calls.append((text, allowed_special))
        return list(self.outputs[text])


class _FakeChatTokenizer:
    def __init__(self):
        self._special_tokens = {
            "<USER>": 91,
            "</USER>": 92,
            "<ASSISTANT>": 93,
            "</ASSISTANT>": 94,
        }


class _TinyChatTrainModel(nn.Module):
    def __init__(self, vocab_size=6, dim=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, vocab_size, bias=False)
        self.last_logits = None

    def forward(self, tokens):
        self.last_logits = self.proj(self.embed(tokens))
        return self.last_logits


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


class _DpoLossSpy:
    def __init__(self, model, model_ref, pos_batches, neg_batches, beta):
        self.model = model
        self.model_ref = model_ref
        self.pos_batches = pos_batches
        self.neg_batches = neg_batches
        self.beta = beta
        self.calls = 0

    def __call__(self, model, model_ref, xp, yp, maskp, xn, yn, maskn, beta):
        assert(model is self.model)
        assert(model_ref is self.model_ref)
        xp_ref, yp_ref, maskp_ref = self.pos_batches[self.calls]
        xn_ref, yn_ref, maskn_ref = self.neg_batches[self.calls]
        assert(torch.equal(xp, xp_ref))
        assert(torch.equal(yp, yp_ref))
        assert(torch.equal(maskp, maskp_ref))
        assert(torch.equal(xn, xn_ref))
        assert(torch.equal(yn, yn_ref))
        assert(torch.equal(maskn, maskn_ref))
        assert(beta == self.beta)
        self.calls += 1
        targets = torch.tensor([1.0, -0.5], device=model.scale.device)
        return (model.scale - targets) ** 2


def _chat_tokenizer():
    base_tokenizer = tiktoken.get_encoding("gpt2")
    return tiktoken.Encoding(
        name="gpt2_chat",
        pat_str=base_tokenizer._pat_str,
        mergeable_ranks=base_tokenizer._mergeable_ranks,
        special_tokens={
            **base_tokenizer._special_tokens,
            "<USER>": 50257,
            "</USER>": 50258,
            "<ASSISTANT>": 50259,
            "</ASSISTANT>": 50300,
        },
    )

def _require_files(*filenames):
    for filename in filenames:
        assert(os.path.exists(filename)), f"Expected `{filename}` to exist before running this test."


def _take_batches(loader, batch_start, num_batches):
    iterator = iter(loader)
    for _ in range(batch_start):
        next(iterator)
    return [next(iterator) for _ in range(num_batches)]


def _load_checkpoint_model(eval_fn, checkpoint_filename):
    _require_files("params.json", checkpoint_filename)
    globals_ = eval_fn.__globals__
    with open("params.json", "rt") as f:
        params = json.load(f)
    model = globals_["LLM"](
        params["num_tokens"],
        params["dim"],
        params["n_heads"],
        params["max_seq_len"],
        params["ffn_dim"],
        params["n_layers"],
    )
    model.load_weights(checkpoint_filename)
    model.float().cpu().eval()
    return model


def _masked_nll_and_close_prob(model, loader, close_token):
    total_loss = 0.0
    total_tokens = 0
    close_probs = []

    model.float().cpu().eval()
    with torch.inference_mode():
        for x, y, mask in loader:
            logits = model(x).float()
            flat_logits = logits[mask]
            flat_y = y[mask]
            token_losses = -flat_logits[torch.arange(flat_y.shape[0]), flat_y] + torch.logsumexp(flat_logits, -1)
            total_loss += token_losses.sum().item()
            total_tokens += token_losses.numel()

            close_mask = mask & (y == close_token)
            if close_mask.any():
                probs = torch.softmax(logits[close_mask], dim=-1)[:, close_token]
                close_probs.extend(probs.detach().cpu().tolist())

    assert(total_tokens > 0)
    assert(close_probs)
    return total_loss / total_tokens, float(np.mean(close_probs))


def _heldout_chat_metrics(eval_fn, tokenized_filename, checkpoint_filename, batch_start=13, num_batches=1):
    _require_files(tokenized_filename, checkpoint_filename, "params.json")
    globals_ = eval_fn.__globals__
    tokenizer = globals_["tokenizer"]
    DataLoaderChat = globals_["DataLoaderChat"]
    close_token = tokenizer._special_tokens["</ASSISTANT>"]

    loader = DataLoaderChat(tokenized_filename, seq_len=1024, batch_size=2, tokenizer=tokenizer, device="cpu")
    holdout_batches = _take_batches(loader, batch_start, num_batches)

    model = eval_fn().float().cpu().eval()
    ref_model = _load_checkpoint_model(eval_fn, checkpoint_filename)
    return (
        _masked_nll_and_close_prob(model, holdout_batches, close_token),
        _masked_nll_and_close_prob(ref_model, holdout_batches, close_token),
    )


def _heldout_dpo_loss(eval_fn, batch_start=11, num_batches=1, beta=0.1):
    _require_files("ultrachat_pos_tokenized.json", "ultrachat_neg_tokenized.json", "model_chat.pth", "params.json")
    globals_ = eval_fn.__globals__
    tokenizer = globals_["tokenizer"]
    DataLoaderChat = globals_["DataLoaderChat"]
    dpo_loss = globals_["dpo_loss"]

    loader_pos = DataLoaderChat("ultrachat_pos_tokenized.json", seq_len=1024, batch_size=2, tokenizer=tokenizer, device="cpu")
    loader_neg = DataLoaderChat("ultrachat_neg_tokenized.json", seq_len=1024, batch_size=2, tokenizer=tokenizer, device="cpu")
    holdout_pos = _take_batches(loader_pos, batch_start, num_batches)
    holdout_neg = _take_batches(loader_neg, batch_start, num_batches)

    model = eval_fn().float().cpu().eval()
    model_ref = _load_checkpoint_model(eval_fn, "model_chat.pth")

    losses = []
    with torch.inference_mode():
        for (xp, yp, maskp), (xn, yn, maskn) in zip(holdout_pos, holdout_neg):
            losses.append(dpo_loss(model, model_ref, xp, yp, maskp, xn, yn, maskn, beta).mean().item())
    return float(np.mean(losses))


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
    assert(torch.allclose(out, ref, atol=1e-6))

    Q = torch.randn(2, 3, 5, 8)
    K = torch.randn(2, 3, 5, 8)
    V = torch.randn(2, 3, 5, 4)
    ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=0.0)
    out = self_attention(Q, K, V, mask)
    assert(out.shape == (2, 3, 5, 4))
    assert(torch.allclose(out, ref, atol=1e-6))


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


def test_generate(generate):
    model = _ToyGenerationModel([3, 4])
    tokenizer = _ToyGenerationTokenizer(eot_token=4)

    out = io.StringIO()
    with redirect_stdout(out):
        generated = generate(model, [1, 2], tokenizer, eot_token=4, temp=0.7, max_tokens=5, verbose=True)

    assert(generated == [3, 4])
    assert(out.getvalue() == "A!")
    assert(len(model.calls) == 2)
    assert(model.calls[0][0].tolist() == [[1, 2]])
    assert(model.calls[0][1] == 0 and model.calls[0][2] is True)
    assert(model.calls[0][3] == torch.long)
    assert(model.calls[1][0].tolist() == [[3]])
    assert(model.calls[1][1] == 2 and model.calls[1][2] is True)


def test_convert_to_chat_format(convert_to_chat_format):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "Bye"},
    ]
    expected = "<USER>Hello</USER><ASSISTANT>Hi!</ASSISTANT><USER>Bye</USER>"
    assert(convert_to_chat_format(messages) == expected)


def submit_convert_to_chat_format(convert_to_chat_format):
    cases = [
        [
            {"role": "user", "content": "What time is it?"},
            {"role": "assistant", "content": "It is noon."},
            {"role": "user", "content": "Thanks."},
            {"role": "assistant", "content": "You're welcome."},
        ],
        [
            {"role": "assistant", "content": "System ready."},
            {"role": "assistant", "content": "Awaiting prompt."},
        ],
        [
            {"role": "user", "content": "Line one.\nLine two."},
            {"role": "assistant", "content": "Tabbed\tresponse."},
            {"role": "user", "content": ""},
        ],
    ]
    for messages in cases:
        mugrade.submit(convert_to_chat_format(messages))


def test_pretokenize_chat(pretokenize_chat):
    chats = [
        [{"role": "user", "content": "One"}, {"role": "assistant", "content": "Two"}],
        [{"role": "user", "content": "Three"}],
    ]
    text0 = "<USER>One</USER><ASSISTANT>Two</ASSISTANT>"
    text1 = "<USER>Three</USER>"
    tokenizer = _FixedEncodeTokenizer({
        text0: [11, 12, 13],
        text1: [21, 22],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        in_filename = os.path.join(tmpdir, "chats.json")
        out_filename = os.path.join(tmpdir, "tokens.json")
        with open(in_filename, "wt") as f:
            json.dump(chats, f)

        pretokenize_chat(tokenizer, in_filename, out_filename)

        with open(out_filename, "rt") as f:
            tokens = json.load(f)

    assert(tokenizer.calls == [(text0, "all"), (text1, "all")])
    assert(tokens == [[11, 12, 13], [21, 22]])


def submit_pretokenize_chat(pretokenize_chat):
    cases = [
        (
            [
                [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}],
                [{"role": "assistant", "content": "C"}],
            ],
            {
                "<USER>A</USER><ASSISTANT>B</ASSISTANT>": [1, 2, 3],
                "<ASSISTANT>C</ASSISTANT>": [4, 5],
            },
        ),
        (
            [
                [{"role": "user", "content": "First"}],
                [{"role": "user", "content": "Second"}, {"role": "assistant", "content": "Third"}],
            ],
            {
                "<USER>First</USER>": [10, 11],
                "<USER>Second</USER><ASSISTANT>Third</ASSISTANT>": [12, 13, 14, 15],
            },
        ),
        (
            [
                [{"role": "assistant", "content": "Alpha"}, {"role": "user", "content": "Beta"}],
                [{"role": "user", "content": ""}, {"role": "assistant", "content": "Gamma"}],
            ],
            {
                "<ASSISTANT>Alpha</ASSISTANT><USER>Beta</USER>": [21, 22, 23],
                "<USER></USER><ASSISTANT>Gamma</ASSISTANT>": [24, 25],
            },
        ),
    ]

    for chats, mapping in cases:
        tokenizer = _FixedEncodeTokenizer(mapping)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_filename = os.path.join(tmpdir, "chats.json")
            out_filename = os.path.join(tmpdir, "tokens.json")
            with open(in_filename, "wt") as f:
                json.dump(chats, f)

            pretokenize_chat(tokenizer, in_filename, out_filename)

            with open(out_filename, "rt") as f:
                tokens = json.load(f)

        mugrade.submit(tokens)


def test_get_loss_mask(get_loss_mask):
    tokenizer = _FakeChatTokenizer()
    tokens = [7, tokenizer._special_tokens["<ASSISTANT>"], 1, 2, tokenizer._special_tokens["</ASSISTANT>"], 8]
    expected = [False, False, True, True, True, False]
    assert(get_loss_mask(tokens, tokenizer) == expected)


def submit_get_loss_mask(get_loss_mask):
    tokenizer = _FakeChatTokenizer()
    cases = [
        [
            tokenizer._special_tokens["<USER>"],
            10,
            tokenizer._special_tokens["<ASSISTANT>"],
            20,
            21,
            tokenizer._special_tokens["</ASSISTANT>"],
            tokenizer._special_tokens["<ASSISTANT>"],
            30,
            tokenizer._special_tokens["</ASSISTANT>"],
        ],
        [
            tokenizer._special_tokens["<ASSISTANT>"],
            31,
            32,
            tokenizer._special_tokens["</ASSISTANT>"],
            9,
        ],
        [
            tokenizer._special_tokens["<USER>"],
            7,
            8,
            tokenizer._special_tokens["</USER>"],
            9,
        ],
    ]
    for tokens in cases:
        mugrade.submit(get_loss_mask(tokens, tokenizer))


def test_DataLoaderChat(DataLoaderChat):
    tokenizer = _FakeChatTokenizer()
    chats = [
        [1, tokenizer._special_tokens["<ASSISTANT>"], 10, 11, tokenizer._special_tokens["</ASSISTANT>"]],
        [2, tokenizer._special_tokens["<ASSISTANT>"], 12, tokenizer._special_tokens["</ASSISTANT>"]],
        [3, tokenizer._special_tokens["<ASSISTANT>"], 13, 14, 15, tokenizer._special_tokens["</ASSISTANT>"]],
        [4, tokenizer._special_tokens["<ASSISTANT>"], 16, tokenizer._special_tokens["</ASSISTANT>"]],
        [5, tokenizer._special_tokens["<ASSISTANT>"], 17, tokenizer._special_tokens["</ASSISTANT>"]],
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "chat_tokens.json")
        with open(filename, "wt") as f:
            json.dump(chats, f)

        loader = DataLoaderChat(filename, seq_len=6, batch_size=2, tokenizer=tokenizer)
        assert(iter(loader) is loader)
        batches = list(loader)
        assert(len(batches) == 2)

        X0, y0, m0 = batches[0]
        assert(torch.equal(X0.cpu(), torch.tensor([
            [1, 93, 10, 11, 94, 0],
            [2, 93, 12, 94, 0, 0],
        ])))
        assert(torch.equal(y0.cpu(), torch.tensor([
            [93, 10, 11, 94, 0, 0],
            [93, 12, 94, 0, 0, 0],
        ])))
        assert(torch.equal(m0.cpu(), torch.tensor([
            [False, True, True, True, False, False],
            [False, True, True, False, False, False],
        ])))

        X1, y1, m1 = batches[1]
        assert(torch.equal(X1.cpu(), torch.tensor([
            [3, 93, 13, 14, 15, 94],
            [4, 93, 16, 94, 0, 0],
        ])))
        assert(torch.equal(y1.cpu(), torch.tensor([
            [93, 13, 14, 15, 94, 0],
            [93, 16, 94, 0, 0, 0],
        ])))
        assert(torch.equal(m1.cpu(), torch.tensor([
            [False, True, True, True, True, False],
            [False, True, True, False, False, False],
        ])))

        batches2 = list(loader)
        for (X_a, y_a, m_a), (X_b, y_b, m_b) in zip(batches, batches2):
            assert(torch.equal(X_a, X_b))
            assert(torch.equal(y_a, y_b))
            assert(torch.equal(m_a, m_b))


def submit_DataLoaderChat(DataLoaderChat):
    tokenizer = _FakeChatTokenizer()
    cases = [
        (
            [
                [1, tokenizer._special_tokens["<ASSISTANT>"], 10, 11, tokenizer._special_tokens["</ASSISTANT>"]],
                [2, tokenizer._special_tokens["<ASSISTANT>"], 12, tokenizer._special_tokens["</ASSISTANT>"]],
            ],
            5,
            2,
            0,
        ),
        (
            [
                [3, tokenizer._special_tokens["<ASSISTANT>"], 13, 14, tokenizer._special_tokens["</ASSISTANT>"]],
                [4, tokenizer._special_tokens["<ASSISTANT>"], 15, tokenizer._special_tokens["</ASSISTANT>"]],
                [5, tokenizer._special_tokens["<ASSISTANT>"], 16, 17, tokenizer._special_tokens["</ASSISTANT>"]],
                [6, tokenizer._special_tokens["<ASSISTANT>"], 18, tokenizer._special_tokens["</ASSISTANT>"]],
            ],
            5,
            2,
            1,
        ),
        (
            [
                [7, tokenizer._special_tokens["<ASSISTANT>"], 19, 20, 21, tokenizer._special_tokens["</ASSISTANT>"]],
            ],
            6,
            1,
            0,
        ),
    ]

    for chats, seq_len, batch_size, batch_idx in cases:
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "chat_tokens.json")
            with open(filename, "wt") as f:
                json.dump(chats, f)

            batches = list(DataLoaderChat(filename, seq_len=seq_len, batch_size=batch_size, tokenizer=tokenizer))

        X, y, mask = batches[batch_idx]
        mugrade.submit(np.array([len(batches)], dtype=np.int64))
        mugrade.submit(X.cpu().numpy())
        mugrade.submit(mask.cpu().numpy())


def test_train_llm_chat(train_llm_chat):
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
    model = _TinyChatTrainModel(vocab_size=6, dim=4)
    init = copy.deepcopy(model.state_dict())
    opt = optim.SGD(model.parameters(), lr=0.1)
    spy = _MaskedCrossEntropySpy(model, batches)

    old_loss = train_llm_chat.__globals__["cross_entropy_loss"]
    try:
        train_llm_chat.__globals__["cross_entropy_loss"] = spy
        with redirect_stdout(io.StringIO()):
            train_llm_chat(model, batches, opt, max_iter=1)
    finally:
        train_llm_chat.__globals__["cross_entropy_loss"] = old_loss

    assert(spy.calls == 1)
    assert(any(not torch.allclose(v, init[k]) for k, v in model.state_dict().items()))

    torch.manual_seed(0)
    model = _TinyChatTrainModel(vocab_size=6, dim=4)
    opt = optim.SGD(model.parameters(), lr=0.1)
    spy = _MaskedCrossEntropySpy(model, batches)
    old_loss = train_llm_chat.__globals__["cross_entropy_loss"]
    try:
        train_llm_chat.__globals__["cross_entropy_loss"] = spy
        with redirect_stdout(io.StringIO()):
            train_llm_chat(model, batches, opt)
    finally:
        train_llm_chat.__globals__["cross_entropy_loss"] = old_loss

    assert(spy.calls == 2)


def submit_train_llm_chat(train_llm_chat):
    def run_case(seed, loader, eval_tokens, max_iter=None):
        torch.manual_seed(seed)
        model = _TinyChatTrainModel(vocab_size=6, dim=4)
        opt = optim.SGD(model.parameters(), lr=0.1)
        old_loss = train_llm_chat.__globals__["cross_entropy_loss"]
        try:
            train_llm_chat.__globals__["cross_entropy_loss"] = lambda logits, y: F.cross_entropy(logits, y)
            with redirect_stdout(io.StringIO()):
                train_llm_chat(model, loader, opt, max_iter=max_iter)
        finally:
            train_llm_chat.__globals__["cross_entropy_loss"] = old_loss
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


def submit_log_probs(log_probs):
    cases = [
        (
            torch.tensor([
                [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]],
                [[-0.5, 0.5], [1.5, 1.5], [0.25, -0.25]],
            ]),
            torch.tensor([[0, 1, 0], [1, 0, 1]]),
            torch.tensor([[True, True, False], [False, True, True]]),
        ),
        (
            torch.tensor([
                [[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[-1.0, 2.0, 1.0], [3.0, 0.0, -3.0]],
            ]),
            torch.tensor([[1, 2], [0, 0]]),
            torch.tensor([[True, False], [True, True]]),
        ),
        (
            torch.tensor([
                [[0.5, -0.5], [1.0, -1.0], [0.0, 0.0]],
            ]),
            torch.tensor([[0, 0, 1]]),
            torch.tensor([[False, True, True]]),
        ),
    ]
    for logits, y, mask in cases:
        mugrade.submit(log_probs(logits, y, mask).detach().numpy())


def test_softplus(softplus):
    x = torch.tensor([-3.0, -0.5, 0.0, 2.0])
    out = softplus(x, 0.7)
    ref = torch.tensor([0.1155, 0.5334, 0.6931, 1.6204])
    assert(torch.allclose(out, ref, atol=1e-4))

    x = torch.tensor([[1.0, -1.0], [0.25, -0.25]])
    assert(torch.allclose(softplus(x, 0.3), torch.logaddexp(0.3 * x, torch.zeros_like(x)), atol=1e-6))


def submit_softplus(softplus):
    cases = [
        (torch.tensor([-2.0, 0.0, 1.5, 3.0]), 0.4),
        (torch.tensor([[1.0, -1.0], [2.5, -2.5]]), 0.8),
        (torch.tensor([0.25]), 0.2),
    ]
    for x, beta in cases:
        mugrade.submit(softplus(x, beta).detach().numpy())


def test_dpo_loss(dpo_loss):
    table = torch.tensor([
        [2.0, 0.0, -1.0],
        [0.0, 2.0, -0.5],
        [1.0, -0.5, 0.5],
        [-1.0, 0.5, 1.5],
    ])
    ref_table = torch.tensor([
        [1.0, 0.0, -0.5],
        [0.0, 1.0, 0.0],
        [0.5, -0.5, 0.25],
        [-0.5, 0.0, 1.0],
    ])
    model = _LookupLM(table)
    model_ref = _LookupLM(ref_table)

    xp = torch.tensor([[0, 1, 2], [1, 2, 3]])
    yp = torch.tensor([[1, 2, 0], [2, 0, 1]])
    maskp = torch.tensor([[True, True, False], [False, True, True]])
    xn = torch.tensor([[0, 2, 3], [3, 1, 0]])
    yn = torch.tensor([[2, 0, 1], [1, 0, 2]])
    maskn = torch.tensor([[True, False, True], [True, True, False]])

    loss = dpo_loss(model, model_ref, xp, yp, maskp, xn, yn, maskn, beta=0.3)
    assert(loss.shape == (2,))
    assert(torch.allclose(loss, torch.tensor([0.8100, 0.5797]), atol=1e-4))

    loss.sum().backward()
    assert(model.table.grad is not None)
    assert(model_ref.table.grad is None)


def submit_dpo_loss(dpo_loss):
    def run_case(table, ref_table, xp, yp, maskp, xn, yn, maskn, beta):
        model = _LookupLM(table)
        model_ref = _LookupLM(ref_table)
        return dpo_loss(model, model_ref, xp, yp, maskp, xn, yn, maskn, beta).detach().numpy()

    cases = [
        (
            torch.tensor([
                [1.5, 0.0, -0.5],
                [0.5, 1.5, -1.0],
                [0.25, -0.75, 1.25],
            ]),
            torch.tensor([
                [1.0, 0.0, -0.25],
                [0.0, 1.0, -0.5],
                [0.0, -0.5, 1.0],
            ]),
            torch.tensor([[0, 1, 2]]),
            torch.tensor([[1, 2, 0]]),
            torch.tensor([[True, True, False]]),
            torch.tensor([[2, 1, 0]]),
            torch.tensor([[2, 0, 1]]),
            torch.tensor([[True, False, True]]),
            0.2,
        ),
        (
            torch.tensor([
                [2.0, -1.0],
                [0.5, 1.0],
            ]),
            torch.tensor([
                [1.0, -0.5],
                [0.0, 0.75],
            ]),
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([[1, 0], [0, 1]]),
            torch.tensor([[True, False], [True, True]]),
            torch.tensor([[1, 0], [0, 1]]),
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([[True, True], [False, True]]),
            0.5,
        ),
        (
            torch.tensor([
                [1.0, 0.0, -1.0, 0.5],
                [0.25, 1.5, -0.5, 0.0],
                [1.25, -0.25, 0.5, -1.0],
                [0.75, 0.25, -0.75, 1.0],
            ]),
            torch.tensor([
                [0.5, 0.0, -0.5, 0.25],
                [0.0, 1.0, -0.25, 0.0],
                [1.0, -0.5, 0.25, -0.5],
                [0.5, 0.0, -0.25, 0.75],
            ]),
            torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([[1, 2, 3, 0]]),
            torch.tensor([[False, True, True, False]]),
            torch.tensor([[3, 2, 1, 0]]),
            torch.tensor([[2, 1, 0, 3]]),
            torch.tensor([[True, False, True, True]]),
            0.1,
        ),
    ]

    for case in cases:
        mugrade.submit(run_case(*case))


def test_train_dpo(train_dpo):
    pos_batches = [
        (torch.tensor([[0, 1]]), torch.tensor([[1, 2]]), torch.tensor([[True, False]])),
        (torch.tensor([[2, 3]]), torch.tensor([[3, 4]]), torch.tensor([[False, True]])),
    ]
    neg_batches = [
        (torch.tensor([[4, 5]]), torch.tensor([[5, 0]]), torch.tensor([[True, True]])),
        (torch.tensor([[1, 0]]), torch.tensor([[0, 2]]), torch.tensor([[True, False]])),
    ]

    model = _ScalarModel(0.0)
    model_ref = _ScalarModel(0.0)
    opt = optim.SGD(model.parameters(), lr=0.1)
    spy = _DpoLossSpy(model, model_ref, pos_batches, neg_batches, beta=0.2)

    old_loss = train_dpo.__globals__["dpo_loss"]
    try:
        train_dpo.__globals__["dpo_loss"] = spy
        with redirect_stdout(io.StringIO()):
            train_dpo(model, model_ref, pos_batches, neg_batches, opt, beta=0.2, max_iter=1)
    finally:
        train_dpo.__globals__["dpo_loss"] = old_loss

    assert(spy.calls == 1)
    assert(not torch.allclose(model.scale, torch.tensor(0.0)))
    assert(torch.allclose(model_ref.scale, torch.tensor(0.0)))

    model = _ScalarModel(0.0)
    model_ref = _ScalarModel(0.0)
    opt = optim.SGD(model.parameters(), lr=0.1)
    spy = _DpoLossSpy(model, model_ref, pos_batches, neg_batches, beta=0.2)
    old_loss = train_dpo.__globals__["dpo_loss"]
    try:
        train_dpo.__globals__["dpo_loss"] = spy
        with redirect_stdout(io.StringIO()):
            train_dpo(model, model_ref, pos_batches, neg_batches, opt, beta=0.2)
    finally:
        train_dpo.__globals__["dpo_loss"] = old_loss

    assert(spy.calls == 2)


def submit_train_dpo(train_dpo):
    def run_case(pos_batches, neg_batches, beta, lr, max_iter=None, init=0.0):
        model = _ScalarModel(init)
        model_ref = _ScalarModel(init)
        opt = optim.SGD(model.parameters(), lr=lr)
        old_loss = train_dpo.__globals__["dpo_loss"]
        try:
            train_dpo.__globals__["dpo_loss"] = _DpoLossSpy(model, model_ref, pos_batches, neg_batches, beta=beta)
            with redirect_stdout(io.StringIO()):
                train_dpo(model, model_ref, pos_batches, neg_batches, opt, beta=beta, max_iter=max_iter)
        finally:
            train_dpo.__globals__["dpo_loss"] = old_loss
        return np.array([model.scale.item(), model_ref.scale.item()], dtype=np.float32)

    base_pos = [
        (torch.tensor([[0, 1]]), torch.tensor([[1, 2]]), torch.tensor([[True, False]])),
        (torch.tensor([[2, 3]]), torch.tensor([[3, 4]]), torch.tensor([[False, True]])),
    ]
    base_neg = [
        (torch.tensor([[4, 5]]), torch.tensor([[5, 0]]), torch.tensor([[True, True]])),
        (torch.tensor([[1, 0]]), torch.tensor([[0, 2]]), torch.tensor([[True, False]])),
    ]
    extra_pos = base_pos + [(torch.tensor([[3, 1]]), torch.tensor([[1, 4]]), torch.tensor([[True, True]]))]
    extra_neg = base_neg + [(torch.tensor([[5, 2]]), torch.tensor([[2, 3]]), torch.tensor([[False, True]]))]

    cases = [
        (base_pos, base_neg, 0.2, 0.1, 1, 0.0),
        (base_pos, base_neg, 0.2, 0.1, None, 0.0),
        (extra_pos, extra_neg, 0.2, 0.05, 2, 0.5),
    ]

    for case in cases:
        mugrade.submit(run_case(*case))


def test_eval_llm_chat(eval_llm_chat):
    model = eval_llm_chat()
    assert(isinstance(model, nn.Module))
    assert(hasattr(model, "layers"))
    assert(len(model.layers) > 0)

    (loss, close_prob), (base_loss, base_close_prob) = _heldout_chat_metrics(
        eval_llm_chat,
        tokenized_filename="ultrachat_tokenized.json",
        checkpoint_filename="model_base.pth",
        batch_start=13,
        num_batches=1,
    )

    assert(loss < base_loss)
    assert(loss < 2.2)
    assert(close_prob > max(0.005, 1000 * base_close_prob))


def submit_eval_llm_chat(eval_llm_chat):
    (loss, close_prob), (base_loss, base_close_prob) = _heldout_chat_metrics(
        eval_llm_chat,
        tokenized_filename="ultrachat_tokenized.json",
        checkpoint_filename="model_base.pth",
        batch_start=12,
        num_batches=1,
    )
    mugrade.submit(loss < base_loss)
    mugrade.submit(loss < 2.4)
    mugrade.submit(close_prob > max(0.005, 1000 * base_close_prob))


def test_eval_llm_dpo(eval_llm_dpo):
    model = eval_llm_dpo()
    assert(isinstance(model, nn.Module))
    assert(hasattr(model, "layers"))
    assert(len(model.layers) > 0)

    heldout_loss = _heldout_dpo_loss(eval_llm_dpo, batch_start=11, num_batches=1, beta=0.1)
    assert(heldout_loss < 0.55)


def submit_eval_llm_dpo(eval_llm_dpo):
    heldout_loss = _heldout_dpo_loss(eval_llm_dpo, batch_start=12, num_batches=1, beta=0.1)
    mugrade.submit(heldout_loss < math.log(2) - 1e-5)
    mugrade.submit(heldout_loss < 0.55)
    mugrade.submit(heldout_loss < 0.53)
