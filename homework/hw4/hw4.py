# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "pytest==9.0.2",
#     "requests==2.32.5",
#     "mugrade @ git+https://github.com/locuslab/mugrade.git",
#     "torch",
#     "huggingface-hub",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo

    import subprocess

    # Run this cell to download and install the necessary modules for the homework
    subprocess.call(
        [
            "wget",
            "-nc",
            "https://raw.githubusercontent.com/modernaicourse/hw4/refs/heads/main/hw4_tests.py",
        ]
    )

    import os
    import math
    import json
    import mugrade
    import torch
    from torch.nn import Module, ModuleList, Parameter, Buffer

    from hw4_tests import (
        test_Linear,
        submit_Linear,
        test_Embedding,
        submit_Embedding,
        test_silu,
        submit_silu,
        test_RMSNorm,
        submit_RMSNorm,
        test_self_attention,
        submit_self_attention,
        test_MultiHeadAttention,
        submit_MultiHeadAttention,
        test_MultiHeadAttentionKVCache,
        submit_MultiHeadAttentionKVCache,
        test_GatedMLP,
        submit_GatedMLP,
        test_TransformerBlock,
        submit_TransformerBlock,
        test_Llama3Simplified,
        submit_Llama3Simplified,
        test_eval_llama3,
        submit_eval_llama3,
        test_generate,
        submit_generate,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Homework 4 - Transformers

    In this homework, you will build the components of a Transformer-based LLM, load the weights of a Llama 3.2 1B model, and use it to perform inference. You should only import `Module`, `ModuleList`, `Parameter`, and `Buffer` from `torch.nn` (no other `torch.nn` modules are allowed).
    """)
    return


@app.cell
def _():
    os.environ["MUGRADE_HW"] = "Homework 4"
    os.environ["MUGRADE_KEY"] = ""  ### Your key here
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 1 - Linear Layer

    Implement a `Linear` layer, similar to the one from Homework 3, but using `torch.empty()` instead of `torch.randn()` to initialize the weight. The weight should be stored as a `Parameter` of shape `(out_dim, in_dim)` called `.weight`. The forward pass computes $XW^T$.
    """)
    return


@app.class_definition
class Linear(Module):
    """
    A linear (fully connected) layer without bias, implementing f(X) = X W^T.
    The weight is stored as a Parameter of shape (out_dim, in_dim).
    """

    def __init__(self, in_dim, out_dim):
        """
        Initialize the Linear layer.

        Input:
            in_dim: int - input dimension
            out_dim: int - output dimension
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X):
        """
        Compute the forward pass of the linear layer.

        Input:
            X: torch.Tensor - input tensor of shape (..., in_dim)
        Output:
            torch.Tensor - output tensor of shape (..., out_dim)
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_Linear_local():
    test_Linear(Linear)


@app.cell(hide_code=True)
def _():
    submit_Linear_button = mo.ui.run_button(label="submit `Linear`")
    submit_Linear_button
    return (submit_Linear_button,)


@app.cell
def _(submit_Linear_button):
    mugrade.submit_tests(Linear) if submit_Linear_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 2 - Embedding Layer

    Implement an `Embedding` layer that converts token indices to dense vectors. The layer stores a `.weight` Parameter of shape `(num_tokens, dim)`, initialized with `torch.empty()`. The forward pass indexes into the weight matrix to retrieve embeddings for the given token indices.
    """)
    return


@app.class_definition
class Embedding(Module):
    """
    An embedding layer that maps token indices to dense vectors.
    The weight is stored as a Parameter of shape (num_tokens, dim).
    """

    def __init__(self, num_tokens, dim):
        """
        Initialize the Embedding layer.

        Input:
            num_tokens: int - number of tokens in the vocabulary
            dim: int - embedding dimension
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, Y):
        """
        Compute the forward pass of the embedding layer.

        Input:
            Y: torch.Tensor[int] - token indices of arbitrary shape
        Output:
            torch.Tensor - embeddings of shape (*Y.shape, dim)
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_Embedding_local():
    test_Embedding(Embedding)


@app.cell(hide_code=True)
def _():
    submit_Embedding_button = mo.ui.run_button(label="submit `Embedding`")
    submit_Embedding_button
    return (submit_Embedding_button,)


@app.cell
def _(submit_Embedding_button):
    mugrade.submit_tests(Embedding) if submit_Embedding_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 3 - SiLU Nonlinearity

    Implement the SiLU (Sigmoid Linear Unit) activation function, defined as:

    $$\text{silu}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}$$
    """)
    return


@app.function
def silu(x):
    """
    Compute the SiLU (Sigmoid Linear Unit) activation function.

    Input:
        x: torch.Tensor - input tensor of arbitrary shape
    Output:
        torch.Tensor - silu(x) = x * sigmoid(x)
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_silu_local():
    test_silu(silu)


@app.cell(hide_code=True)
def _():
    submit_silu_button = mo.ui.run_button(label="submit `silu`")
    submit_silu_button
    return (submit_silu_button,)


@app.cell
def _(submit_silu_button):
    mugrade.submit_tests(silu) if submit_silu_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 4 - RMS Norm

    Implement RMS (Root Mean Square) Normalization. Given an input $X \in \mathbb{R}^{n \times d}$ and a learnable weight $w \in \mathbb{R}^d$, RMSNorm is defined as:

    $$\text{RMSNorm}(X) = \frac{X}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} X_{j}^2 + \epsilon}} \odot w$$

    The `.weight` parameter should be initialized to ones.
    """)
    return


@app.class_definition
class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization.
    The weight is stored as a Parameter of shape (dim,), initialized to ones.
    """

    def __init__(self, dim, eps=1e-5):
        """
        Initialize the RMSNorm layer.

        Input:
            dim: int - feature dimension
            eps: float - epsilon for numerical stability
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X):
        """
        Compute the forward pass of RMSNorm.

        Input:
            X: torch.Tensor - input tensor of shape (..., dim)
        Output:
            torch.Tensor - normalized tensor of shape (..., dim)
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_RMSNorm_local():
    test_RMSNorm(RMSNorm)


@app.cell(hide_code=True)
def _():
    submit_RMSNorm_button = mo.ui.run_button(label="submit `RMSNorm`")
    submit_RMSNorm_button
    return (submit_RMSNorm_button,)


@app.cell
def _(submit_RMSNorm_button):
    mugrade.submit_tests(RMSNorm) if submit_RMSNorm_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 5 - Masked Self Attention

    Implement scaled dot-product self attention. Given queries $Q$, keys $K$, and values $V$, and an optional mask, self attention is computed as:

    $$\text{self\_attention}(Q, K, V, \text{mask}) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \text{mask}\right) V$$

    where $d$ is the dimension of the queries/keys (the last dimension of $Q$). If `mask` is `None`, the attention is computed without masking.
    """)
    return


@app.function
def self_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product self attention.

    Input:
        Q: torch.Tensor - queries of shape (..., seq_len, d)
        K: torch.Tensor - keys of shape (..., seq_len, d)
        V: torch.Tensor - values of shape (..., seq_len, d_v)
        mask: torch.Tensor or None - additive attention mask
    Output:
        torch.Tensor - attention output of shape (..., seq_len, d_v)
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_self_attention_local():
    test_self_attention(self_attention)


@app.cell(hide_code=True)
def _():
    submit_self_attention_button = mo.ui.run_button(label="submit `self_attention`")
    submit_self_attention_button
    return (submit_self_attention_button,)


@app.cell
def _(submit_self_attention_button):
    mugrade.submit_tests(self_attention) if submit_self_attention_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 6 - Multi-head Attention (no caching)

    Implement multi-head attention with the following steps:

    1. Form $Q = XW_q^T$, $K = XW_k^T$, $V = XW_v^T$ using Linear layers `wq`, `wk`, `wv`.
    2. Split the $d$-dimensional vectors into `n_heads` heads, each of dimension $d / n\_heads$. Reshape from `(batch, seq, dim)` to `(batch, n_heads, seq, head_dim)`.
    3. Apply `self_attention` to each head (with optional mask).
    4. Concatenate the heads back and project with `wp`: $\text{output} = \text{concat}(heads) W_p^T$.

    For this first version, ignore the `seq_pos` and `use_kv_cache` arguments.
    """)
    return


@app.class_definition
class MultiHeadAttention(Module):
    """
    Multi-head attention without KV caching.
    Contains Linear layers wq, wk, wv, wp.
    """

    def __init__(self, dim, n_heads):
        """
        Initialize multi-head attention.

        Input:
            dim: int - model dimension
            n_heads: int - number of attention heads
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X, mask=None, seq_pos=0, use_kv_cache=False):
        """
        Compute multi-head attention.

        Input:
            X: torch.Tensor - input of shape (batch, seq_len, dim)
            mask: torch.Tensor or None - attention mask
            seq_pos: int - (unused in this version)
            use_kv_cache: bool - (unused in this version)
        Output:
            torch.Tensor - output of shape (batch, seq_len, dim)
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_MultiHeadAttention_local():
    test_MultiHeadAttention(MultiHeadAttention)


@app.cell(hide_code=True)
def _():
    submit_MultiHeadAttention_button = mo.ui.run_button(
        label="submit `MultiHeadAttention`"
    )
    submit_MultiHeadAttention_button
    return (submit_MultiHeadAttention_button,)


@app.cell
def _(submit_MultiHeadAttention_button):
    mugrade.submit_tests(
        MultiHeadAttention
    ) if submit_MultiHeadAttention_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 6 (cont.) - Multi-head Attention with KV Cache

    Now implement multi-head attention with KV caching. The KV cache stores previously computed keys and values so that during autoregressive generation, we only need to compute Q, K, V for the new tokens, then concatenate K and V with the cached values.

    The class should have `k_cache` and `v_cache` Buffers of shape `(1, max_cache_size, dim)`, initialized to zeros.

    When `use_kv_cache=True`:
    - Compute K, V only for the new tokens.
    - Store them in the cache at positions `seq_pos : seq_pos + seq_len`.
    - Use the full cached K, V (up to `seq_pos + seq_len`) for attention.

    When `use_kv_cache=False`:
    - Behave exactly like the non-caching version.
    """)
    return


@app.class_definition
class MultiHeadAttentionKVCache(Module):
    """
    Multi-head attention with KV caching.
    Contains Linear layers wq, wk, wv, wp and Buffers k_cache, v_cache.
    """

    def __init__(self, dim, n_heads, max_cache_size):
        """
        Initialize multi-head attention with KV cache.

        Input:
            dim: int - model dimension
            n_heads: int - number of attention heads
            max_cache_size: int - maximum sequence length for the cache
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X, mask=None, seq_pos=0, use_kv_cache=False):
        """
        Compute multi-head attention, optionally using KV cache.

        Input:
            X: torch.Tensor - input of shape (batch, seq_len, dim)
            mask: torch.Tensor or None - attention mask
            seq_pos: int - starting position in the sequence (for caching)
            use_kv_cache: bool - whether to use the KV cache
        Output:
            torch.Tensor - output of shape (batch, seq_len, dim)
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_MultiHeadAttentionKVCache_local():
    test_MultiHeadAttentionKVCache(MultiHeadAttentionKVCache)


@app.cell(hide_code=True)
def _():
    submit_MultiHeadAttentionKVCache_button = mo.ui.run_button(
        label="submit `MultiHeadAttentionKVCache`"
    )
    submit_MultiHeadAttentionKVCache_button
    return (submit_MultiHeadAttentionKVCache_button,)


@app.cell
def _(submit_MultiHeadAttentionKVCache_button):
    mugrade.submit_tests(
        MultiHeadAttentionKVCache
    ) if submit_MultiHeadAttentionKVCache_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 7 - Gated MLP

    Implement the Gated MLP (as used in Llama), defined as:

    $$\text{GatedMLP}(X) = (\text{silu}(XW_1^T) \odot XW_3^T) W_2^T$$

    where $W_1, W_3 \in \mathbb{R}^{d_{ffn} \times d}$ and $W_2 \in \mathbb{R}^{d \times d_{ffn}}$ are Linear layers `w1`, `w2`, `w3`, and $\odot$ denotes element-wise multiplication.
    """)
    return


@app.class_definition
class GatedMLP(Module):
    """
    Gated MLP as used in Llama models.
    Contains Linear layers w1, w2, w3.
    """

    def __init__(self, dim, ffn_dim):
        """
        Initialize the Gated MLP.

        Input:
            dim: int - model dimension
            ffn_dim: int - feed-forward hidden dimension
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X):
        """
        Compute the forward pass of the Gated MLP.

        Input:
            X: torch.Tensor - input of shape (..., dim)
        Output:
            torch.Tensor - output of shape (..., dim)
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_GatedMLP_local():
    test_GatedMLP(GatedMLP)


@app.cell(hide_code=True)
def _():
    submit_GatedMLP_button = mo.ui.run_button(label="submit `GatedMLP`")
    submit_GatedMLP_button
    return (submit_GatedMLP_button,)


@app.cell
def _(submit_GatedMLP_button):
    mugrade.submit_tests(GatedMLP) if submit_GatedMLP_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 8 - Transformer Block

    Implement a single Transformer block with pre-norm residual connections:

    $$Z = X + \text{MultiHeadAttentionKVCache}(\text{RMSNorm}_1(X))$$
    $$Y = Z + \text{GatedMLP}(\text{RMSNorm}_2(Z))$$

    The block should contain: `attn` (MultiHeadAttentionKVCache), `norm1` and `norm2` (RMSNorm), and `mlp` (GatedMLP).
    """)
    return


@app.class_definition
class TransformerBlock(Module):
    """
    A single Transformer block with pre-norm residual connections.
    Contains attn (MultiHeadAttentionKVCache), norm1, norm2 (RMSNorm), mlp (GatedMLP).
    """

    def __init__(self, dim, n_heads, ffn_dim, max_cache_size):
        """
        Initialize the Transformer block.

        Input:
            dim: int - model dimension
            n_heads: int - number of attention heads
            ffn_dim: int - feed-forward hidden dimension
            max_cache_size: int - maximum cache size for KV cache
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X, mask=None, seq_pos=0, use_kv_cache=False):
        """
        Compute the forward pass of the Transformer block.

        Input:
            X: torch.Tensor - input of shape (batch, seq_len, dim)
            mask: torch.Tensor or None - attention mask
            seq_pos: int - starting position in the sequence
            use_kv_cache: bool - whether to use KV cache
        Output:
            torch.Tensor - output of shape (batch, seq_len, dim)
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_TransformerBlock_local():
    test_TransformerBlock(TransformerBlock)


@app.cell(hide_code=True)
def _():
    submit_TransformerBlock_button = mo.ui.run_button(
        label="submit `TransformerBlock`"
    )
    submit_TransformerBlock_button
    return (submit_TransformerBlock_button,)


@app.cell
def _(submit_TransformerBlock_button):
    mugrade.submit_tests(
        TransformerBlock
    ) if submit_TransformerBlock_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 9 - Llama3 Model

    Implement the full Llama 3 simplified model. The model consists of:

    - `embedding`: an `Embedding` layer mapping token indices to vectors of dimension `dim`.
    - `pos_embeddings`: a `Buffer` of shape `(max_seq_len, dim)` for positional embeddings (initialized with `torch.empty()`).
    - `layers`: a `ModuleList` of `num_layers` `TransformerBlock` layers.
    - `norm`: a final `RMSNorm` layer.
    - `output`: a `Linear` layer mapping from `dim` to `num_tokens`.
    - `mask`: a `Buffer` containing a causal mask of shape `(max_seq_len, max_seq_len)`, i.e., an upper-triangular matrix of `-inf` values (use `torch.triu` with `diagonal=1`).

    The forward pass:
    1. Look up token embeddings and add positional embeddings for positions `seq_pos` through `seq_pos + seq_len`.
    2. Pass through each Transformer block (with appropriate mask and caching arguments).
    3. Apply final RMSNorm.
    4. Apply the output Linear layer.

    The `load_llama_weights` method is provided for you to load pretrained weights.
    """)
    return


@app.class_definition
class Llama3Simplified(Module):
    """
    A simplified Llama 3 model with positional embeddings (instead of RoPE).
    Contains embedding, pos_embeddings, layers, norm, output, and mask.
    """

    def __init__(self, num_tokens, dim, n_heads, max_seq_len, ffn_dim, num_layers):
        """
        Initialize the Llama3Simplified model.

        Input:
            num_tokens: int - vocabulary size
            dim: int - model dimension
            n_heads: int - number of attention heads
            max_seq_len: int - maximum sequence length
            ffn_dim: int - feed-forward hidden dimension
            num_layers: int - number of Transformer blocks
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, tokens, seq_pos=0, use_kv_cache=False):
        """
        Compute the forward pass of the Llama3Simplified model.

        Input:
            tokens: torch.Tensor[int] - token indices of shape (batch, seq_len)
            seq_pos: int - starting position in the sequence
            use_kv_cache: bool - whether to use KV cache
        Output:
            torch.Tensor - logits of shape (batch, seq_len, num_tokens)
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def load_llama_weights(self, checkpoint):
        self.embedding.weight.data = checkpoint["tok_embeddings.weight"]
        self.pos_embeddings.data = checkpoint["pos_embeddings.weight"]
        self.norm.weight.data = checkpoint["norm.weight"]
        self.output.weight.data = checkpoint["output.weight"]

        for i, layer in enumerate(self.layers):
            layer.attn.wq.weight.data = checkpoint[
                f"layers.{i}.attention.wq.weight"
            ]
            layer.attn.wk.weight.data = checkpoint[
                f"layers.{i}.attention.wk.weight"
            ]
            layer.attn.wv.weight.data = checkpoint[
                f"layers.{i}.attention.wv.weight"
            ]
            layer.attn.wp.weight.data = checkpoint[
                f"layers.{i}.attention.wo.weight"
            ]

            layer.mlp.w1.weight.data = checkpoint[
                f"layers.{i}.feed_forward.w1.weight"
            ]
            layer.mlp.w2.weight.data = checkpoint[
                f"layers.{i}.feed_forward.w2.weight"
            ]
            layer.mlp.w3.weight.data = checkpoint[
                f"layers.{i}.feed_forward.w3.weight"
            ]

            layer.norm1.weight.data = checkpoint[
                f"layers.{i}.attention_norm.weight"
            ]
            layer.norm2.weight.data = checkpoint[
                f"layers.{i}.ffn_norm.weight"
            ]


@app.function(hide_code=True)
def test_Llama3Simplified_local():
    test_Llama3Simplified(Llama3Simplified)


@app.cell(hide_code=True)
def _():
    submit_Llama3Simplified_button = mo.ui.run_button(
        label="submit `Llama3Simplified`"
    )
    submit_Llama3Simplified_button
    return (submit_Llama3Simplified_button,)


@app.cell
def _(submit_Llama3Simplified_button):
    mugrade.submit_tests(
        Llama3Simplified
    ) if submit_Llama3Simplified_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If you implemented the above correctly, you should be able to download the model configuration and weights from HuggingFace, load them into your model, and run inference.
    """)
    return


@app.cell
def _():
    from huggingface_hub import hf_hub_download

    repo = "zkolter/Llama-3.2-1B-Instruct-Simplified"
    for filename in [
        "consolidated.00.pth",
        "params.json",
        "tokenizer.model",
        "tokenizer.py",
    ]:
        if not os.path.exists(filename):
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                repo_type="model",
                local_dir=".",
            )

    checkpoint = torch.load(
        "consolidated.00.pth", map_location=torch.device("cpu")
    )
    with open("params.json", "rt") as f:
        params = json.load(f)

    model = Llama3Simplified(
        params["vocab_size"],
        params["dim"],
        params["n_heads"],
        params["max_seq_len"],
        params["dim"] * params["ffn_dim_multiplier"],
        params["n_layers"],
    )
    model.load_llama_weights(checkpoint)
    model = model.float()
    return (model,)


@app.cell
def _(model):
    def eval_llama3():
        """
        Return the loaded Llama3Simplified model for evaluation.
        """
        return model

    return (eval_llama3,)


@app.cell(hide_code=True)
def _(eval_llama3):
    def test_eval_llama3_local():
        test_eval_llama3(eval_llama3)

    return


@app.cell(hide_code=True)
def _():
    submit_eval_llama3_button = mo.ui.run_button(label="submit `eval_llama3`")
    submit_eval_llama3_button
    return (submit_eval_llama3_button,)


@app.cell
def _(eval_llama3, submit_eval_llama3_button):
    mugrade.submit_tests(eval_llama3) if submit_eval_llama3_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 10 - Generation

    Implement autoregressive text generation with KV caching. Given a model, a list of prompt tokens, and a tokenizer:

    1. Run the model on the full prompt (as a batch of size 1) with `seq_pos=0` and `use_kv_cache=True`.
    2. Sample the next token from the logits of the last position using temperature-scaled sampling: divide logits by `temp`, apply softmax, and sample from the resulting distribution using `torch.multinomial`.
    3. If the sampled token is a stop token (`tokenizer.stop_tokens`), stop generating.
    4. Otherwise, if `verbose=True`, print the decoded token (using `tokenizer.decode`) with `end="\"` and `flush=True`.
    5. Feed the sampled token back into the model (as a single-token sequence) with the appropriate `seq_pos` and `use_kv_cache=True`.
    6. Repeat until a stop token is generated or `max_tokens` new tokens have been generated.
    7. Return the list of all generated tokens (not including the prompt, but including the stop token if one was generated).
    """)
    return


@app.function
def generate(model, prompt_tokens, tokenizer, temp=0.7, max_tokens=500, verbose=True):
    """
    Generate tokens autoregressively using KV caching.

    Input:
        model: Llama3Simplified - the language model
        prompt_tokens: list[int] - the prompt token ids
        tokenizer: Tokenizer - tokenizer with .decode() and .stop_tokens
        temp: float - temperature for sampling
        max_tokens: int - maximum number of tokens to generate
        verbose: bool - whether to print generated tokens
    Output:
        list[int] - the generated token ids (not including the prompt)
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_generate_local():
    test_generate(generate)


@app.cell(hide_code=True)
def _():
    submit_generate_button = mo.ui.run_button(label="submit `generate`")
    submit_generate_button
    return (submit_generate_button,)


@app.cell
def _(submit_generate_button):
    mugrade.submit_tests(generate) if submit_generate_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If you have implemented this all correctly, the following code should generate a response from the Llama 3.2 1B model.
    """)
    return


@app.cell
def _(model):
    from tokenizer import Tokenizer, Message, ChatFormat

    tokenizer = Tokenizer("tokenizer.model")
    chat = ChatFormat(tokenizer)
    msg = Message(role="user", content="Why is the sky blue?")
    prompt = chat.encode_dialog_prompt([msg])

    generate(model, prompt, tokenizer)
    return


if __name__ == "__main__":
    app.run()
