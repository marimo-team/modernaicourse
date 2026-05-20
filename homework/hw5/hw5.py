# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "pytest==9.0.2",
#     "requests==2.32.5",
#     "mugrade @ git+https://github.com/locuslab/mugrade.git",
#     "torch",
#     "tiktoken",
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
            "https://raw.githubusercontent.com/modernaicourse/hw5/refs/heads/main/hw5_tests.py",
        ]
    )

    import os
    import math
    import re
    import mugrade
    import numpy as np
    import torch
    import tiktoken
    from collections import Counter
    from torch.nn import Module, ModuleList, Parameter, Buffer

    from hw5_tests import (
        test_text_to_corpus,
        submit_text_to_corpus,
        test_most_common_pair,
        submit_most_common_pair,
        test_merge_pair,
        submit_merge_pair,
        test_train_bpe,
        submit_train_bpe,
        test_bpe_encode,
        submit_bpe_encode,
        test_bpe_decode,
        submit_bpe_decode,
        test_Linear,
        submit_Linear,
        test_Embedding,
        submit_Embedding,
        test_silu,
        submit_silu,
        test_rms_norm,
        submit_rms_norm,
        test_self_attention,
        submit_self_attention,
        test_MultiHeadAttentionKVCache,
        submit_MultiHeadAttentionKVCache,
        test_MLP,
        submit_MLP,
        test_TransformerBlock,
        submit_TransformerBlock,
        test_LLM,
        submit_LLM,
        test_cross_entropy_loss,
        submit_cross_entropy_loss,
        test_pretokenize_data,
        submit_pretokenize_data,
        test_DataLoader,
        submit_DataLoader,
        test_Adam,
        submit_Adam,
        test_train_llm,
        submit_train_llm,
        test_generate,
        submit_generate,
        test_eval_llm,
        submit_eval_llm,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Homework 5 - Training an LLM

    This homework will walk you through the process of training an LLM from scratch, using the code that you developed in the previous assignments, plus some slight changes and additional logic.  Your LLM will be built on the [Tiny Stories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, a synthetic dataset used for LLM research).  If you train the full model (which will require you to use a GPU), it will be able to generate simple stories like this one:
    > Once upon a time, there was a little girl named Lily. She loved to play outside in her garden. One sunny day, she saw a little red tomato on the ground. Lily thought it was very pretty and wanted to pick it up.
    Lily walked to her friend, Tom. "Tom, look at this big red tomato!" she said. Tom looked at it and said, "Wow, that's a big red tomato! Let's pick it up!" They both picked up the tomato and started to walk back to their house.
    As they walked, Lily saw a big dog. The dog was hungry too. It looked sad and hungry. Lily and Tom decided to help the dog find something to eat. They found a big stick and brought it to the dog. The dog was so happy and wagged its tail. Tom, Lily, and the dog all sat down and enjoyed their tasty, red tomato.

    [I found this one particularly suspenseful, being worried that Lily and Tom were going to feed the stick to the dog instead of sharing their tomato.  Fortunately, it seems to have worked out in the end].

    In addition, although you won't use it directly to train your network, you will also build a small tokenizer to convert raw text files to tokens.

    As with all previous assignment **no routines within the `torch.nn` library can be used, except for the imported `Module`, `ModuleList`, `Buffer` and `Parameter`**.
    """)
    return


@app.cell
def _():
    os.environ["MUGRADE_HW"] = "Homework 5"
    os.environ["MUGRADE_KEY"] = ""  ### Your key here
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part I - Tokenization

    In this section, you will build a simple tokenizer.  Unlike most of the components you build throughout this course, this portion will _not_ be used later in the assignment to actually tokenize the data (it would be much too slow compared to the off-the-shelf libraries that exist build on the CPU).  Nonetheless, you _will_ build a tokenizer as they work for larger models, just doing so within a slower python library.

    Note that for a "real" tokenizer implementation, you would want to do all this manipulation directly as integers indicating the tokens, not as actual character sequences.  However, the way we do things here is conceptually simple and hopefully also easier to debug.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 1 - Splitting a document into unique words

    A a first step in the tokenization process is to split text on whitespace into a collection of "words" (but including the whitespace).  Implement the following function to convert a text document into a list of unique words and their corresponding counts (you can use the `collections.Counter` object to help with this).

    Function like the Python `.split()` function don't easily allow you to split on whitespace, so instead you should use the following command to perform the split a piece of text by whitespace, while still keeping whatever whitespace:
    ```python
    import re
    splits = re.split(r'(?=\s)', text)
    ```

    Your function should convert a text in a "corpus", a list of unique words split by whitespace, where each word in then stored as a list of characters (really tokens, but at the outset each token is just a single charcater), along with a count of the number of times each word occurs in the text.  For example, suppose the original text was the following:
    ```
     test target test
    ```
    Then the `text_to_corpus` functions would return two arguments: first a list of lists
    ```
    [[' ', 't', 'e', 's', 't'], [' ', 't', 'a', 'r', 'g', 'e', 't']]
    ```
    and then a list of the counts:
    ```
    [2, 1]
    ```
    Indicating that the first word occured twice in the document and the second occured once.
    """)
    return


@app.function
def text_to_corpus(text):
    """
    Convert raw text into unique whitespace-delimited words and counts.

    Input:
        text : str - input text document
    Output:
        tuple(list[list[str]], list[int]) - unique words as token lists and their occurrence counts
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_text_to_corpus_local():
    test_text_to_corpus(text_to_corpus)


@app.cell(hide_code=True)
def _():
    submit_text_to_corpus_button = mo.ui.run_button(label="submit `text_to_corpus`")
    submit_text_to_corpus_button
    return (submit_text_to_corpus_button,)


@app.cell
def _(submit_text_to_corpus_button):
    mugrade.submit_tests(
        text_to_corpus
    ) if submit_text_to_corpus_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 2 - Most common pair

    Write a function that computes the most common pair of tokens in the corpus.  This would loop through all pairs of tokens included in each list within the corpus, and return the one with the highest occurence count.

    For example, consider our instance above, which has the corpus
    ```
    [[' ', 't', 'e', 's', 't'], [' ', 't', 'a', 'r', 'g', 'e', 't']]
    ```
    and counts
    ```
    [2, 1]
    ```
    Then this function should return a tuple
    ```
    (' ', 't')
    ```
    because this pair occurs three times in the corpus (when each occurrance in multiplied by its count).

    It may convenient to use the `.most_common()` function of the `Counter` object for this implementation.
    """)
    return


@app.function
def most_common_pair(corpus, counts):
    """
    Find the most frequent adjacent token pair in a weighted corpus.

    Inputs:
        corpus : list[list[str]] - unique words represented as token lists
        counts : list[int] - occurrence count for each word in the corpus
    Output:
        tuple(str, str) - most common adjacent token pair, weighted by counts
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_most_common_pair_local():
    test_most_common_pair(most_common_pair)


@app.cell(hide_code=True)
def _():
    submit_most_common_pair_button = mo.ui.run_button(
        label="submit `most_common_pair`"
    )
    submit_most_common_pair_button
    return (submit_most_common_pair_button,)


@app.cell
def _(submit_most_common_pair_button):
    mugrade.submit_tests(
        most_common_pair
    ) if submit_most_common_pair_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 3 - Merging a pair

    Now write a function that merges a given pair of tokens together in all the elements of corpus (this one doesn't depend on the counts at all).  For example, again suppose the corpus:
    ```
    [[' ', 't', 'e', 's', 't'], [' ', 't', 'a', 'r', 'g', 'e', 't']]
    ```
    and the merge (i.e., pair of tokens)
    ```
    (' ', 't')
    ```
    Then calling this function would merge together all instances of the pair `(' ', 't')` resulting in the new corpus
    ```
    [[' t', 'e', 's', 't'], [' t', 'a', 'r', 'g', 'e', 't']].
    ```

    Note that this function should modify `corpus` _in place_, i.e., it doesn't return anything, but it beforms the modification directly on corpus.
    """)
    return


@app.function
def merge_pair(corpus, pair):
    """
    Merge a given token pair everywhere it appears in the corpus.

    Inputs:
        corpus : list[list[str]] - corpus of tokenized words to modify in place
        pair : tuple(str, str) - adjacent token pair to merge
    Output:
        None - modifies corpus directly
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_merge_pair_local():
    test_merge_pair(merge_pair)


@app.cell(hide_code=True)
def _():
    submit_merge_pair_button = mo.ui.run_button(label="submit `merge_pair`")
    submit_merge_pair_button
    return (submit_merge_pair_button,)


@app.cell
def _(submit_merge_pair_button):
    mugrade.submit_tests(merge_pair) if submit_merge_pair_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 4 - Training a BPE Tokenizer

    Finally, let's use the above functions to train a BPE tokenizer, consistent of N different tokens.  The basic process for building such a tokenizer is the following:
    1. Initialize the tokens to the set of all base characters, which you can produce with the Python code `[chr(i) for i in range(256)]`
    2. Convert the text to corpus and counts, and initialize an empty list of merges.
    3. Repeat until the number of tokens meets the desired `num_tokens`:
      - Compute the most common pair of tokens
      - Add this pair to the list of merges, and add the merged token to the list of tokens
      - Merge the pair within the corpus

    When you are finished, you will have a list of tokens and a list of merges (pairs of tokens).

    Your function should return two items:
    1. The list of tokens, but processed to be returned as a dictionary `{<token string>:<token index>}` where `<token string>` is the token itself and `<token index>` is its index in the list of tokens you originally created.  We return things in this manner so that it's faster to encode a sequence of tokens to their indices.  The dictionary keys should be in the same order as the original list.
    2. The list of merges as you created it (i.e., a list of tuples of pairs of tokens).
    """)
    return


@app.function
def train_bpe(text, num_tokens):
    """
    Train a simple BPE tokenizer from raw text.

    Inputs:
        text : str - training text used to build the tokenizer
        num_tokens : int - total vocabulary size after adding merges
    Output:
        tuple(dict[str, int], list[tuple[str, str]]) - token-to-index mapping and merge list
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_train_bpe_local():
    test_train_bpe(train_bpe)


@app.cell(hide_code=True)
def _():
    submit_train_bpe_button = mo.ui.run_button(label="submit `train_bpe`")
    submit_train_bpe_button
    return (submit_train_bpe_button,)


@app.cell
def _(submit_train_bpe_button):
    mugrade.submit_tests(train_bpe) if submit_train_bpe_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 5 - Encoding and decoding with a BPE tokenizer

    Now write functions to encode and decode a piece of text to and from tokens using your trained BPE tokenizer.  The process for doing so is the following.

    To encode text:
    1. Split the text into a corpus using the same splitting on whitespace as above (convert to a list of lists, where each subelement is a list of individual character tokens in the word).  Here, do _not_ combine duplicate strings, just leave everything as a list of lists.
    2. For each merge in the list, merge that pair in the corpus.
    3. After you have merged all the pairs, convert each token into its numerical id, and return a single (not nested) list of all the tokens in the sequence.


    To decode text:
    1. For each numerical id, find the corresponding token string, and convert join these all together.
    """)
    return


@app.function
def bpe_encode(text, merges, tokens):
    """
    Encode text into token ids using a learned BPE tokenizer.

    Inputs:
        text : str - text to encode
        merges : list[tuple[str, str]] - learned merge operations in order
        tokens : dict[str, int] - mapping from token string to token id
    Output:
        list[int] - encoded token ids for the full text
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function
def bpe_decode(seq, tokens):
    """
    Decode a sequence of BPE token ids back into raw text.

    Inputs:
        seq : list[int] - token ids to decode
        tokens : dict[str, int] - mapping from token string to token id
    Output:
        str - decoded text
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_bpe_encode_local():
    test_bpe_encode(bpe_encode)


@app.function(hide_code=True)
def test_bpe_decode_local():
    test_bpe_decode(bpe_decode)


@app.cell(hide_code=True)
def _():
    submit_bpe_encode_button = mo.ui.run_button(label="submit `bpe_encode`")
    submit_bpe_encode_button
    return (submit_bpe_encode_button,)


@app.cell
def _(submit_bpe_encode_button):
    mugrade.submit_tests(bpe_encode) if submit_bpe_encode_button.value else None
    return


@app.cell(hide_code=True)
def _():
    submit_bpe_decode_button = mo.ui.run_button(label="submit `bpe_decode`")
    submit_bpe_decode_button
    return (submit_bpe_decode_button,)


@app.cell
def _(submit_bpe_decode_button):
    mugrade.submit_tests(bpe_decode) if submit_bpe_decode_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Tokenizing data from Tiny Stories

    If you have implemented all the above correctly, you should be able to build a tokenizer on the Tiny Stories data set using the following code.
    """)
    return


@app.cell(hide_code=True)
def _():
    download_tinystories_button = mo.ui.run_button(
        label="Download the Tiny Stories dataset"
    )
    download_tinystories_button
    return (download_tinystories_button,)


@app.cell
def _(download_tinystories_button):
    mo.stop(not download_tinystories_button.value)

    from huggingface_hub import hf_hub_download

    tinystories_repo = "roneneldan/TinyStories"
    tinystories_filename = "TinyStoriesV2-GPT4-train.txt"
    if not os.path.exists(tinystories_filename):
        hf_hub_download(
            repo_id=tinystories_repo,
            filename=tinystories_filename,
            repo_type="dataset",
            local_dir=".",
        )
    return


@app.cell(hide_code=True)
def _():
    build_tokenizer_button = mo.ui.run_button(label="Build BPE tokenizer")
    build_tokenizer_button
    return (build_tokenizer_button,)


@app.cell
def _(build_tokenizer_button):
    mo.stop(not build_tokenizer_button.value)

    with open(
        "TinyStoriesV2-GPT4-train.txt", mode="rt", encoding="latin-1"
    ) as bpe_file:
        train_text = bpe_file.read(100000)
        test_text = bpe_file.read(1000)
    tokens, merges = train_bpe(train_text, 2000)

    print("Original text: ", test_text)
    tokenized_text = bpe_encode(test_text, merges, tokens)
    print("\nTokens: ", tokenized_text)
    assert test_text == bpe_decode(tokenized_text, tokens)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part II - A (Slightly) Simpler Transformer
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 6 - The Transformer Architecture

    In this section, you will implement the basic architecture of the Transformer you will use to build your LLM.  These components are exactly as in the previous assigment `hw4`, and therefore we say very little about them (you should mostly be copying the solution from that previous homework).  Therefore, we will not include any additional descriptions of each implementation block (see the previous homework for this).  However, there are the following important differences from the solutions in the last homework:

    1. Since we're actually doing training now, for all layers with weights (`Linear`, `Embedding` and the positional embedding in `LLM`), you have to now initialize these with random values, not just with `torch.empty()`.  For `Linear` initialize weight to be random normal entries scaled by $\sqrt{2/\text{in\_dim}}$.  For the `Embedding` and `LLM` layers, make the embeddings _just_ random normal entires (not scaled by any factor).
    2. We're going to implement RMS Norm just as a function without any scaling weight.  We do this because the per-norm scaling doesn't really make any difference for the size of network we're considering here.  It _does_, however, mean you'll need to change your later layer so that they just call this function `rms_norm()`, instead of including a module for each norm layer in the network.
    3. Instead of the `GatedMLP` of the Llama modelss, our Transformer block will just use a normal two-layer MLP,ie. $\sigma(X W_1^T) W_2^T$.  Again, there isn't any real advantage to the gated version for the size of network we're using.
    """)
    return


@app.class_definition
class Linear(Module):
    """ Linear layer with no bias term.  The parameters of the layer are stored in a .weight Parameter"""
    def __init__(self, in_dim, out_dim):
        """
        Initialize a linear layer without a bias term.

        Inputs:
            in_dim : int - input feature dimension
            out_dim : int - output feature dimension
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X):
        """
        Apply the linear layer to one or more input vectors.

        Input:
            X : torch.Tensor[float] (... x in_dim) - input tensor
        Output:
            torch.Tensor[float] (... x out_dim) - transformed tensor
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


@app.class_definition
class Embedding(Module):
    def __init__(self, num_tokens, dim):
        """
        Initialize an embedding table over a fixed vocabulary.

        Inputs:
            num_tokens : int - vocabulary size
            dim : int - embedding dimension
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, Y):
        """
        Look up embeddings for an integer tensor of token ids.

        Input:
            Y : torch.Tensor[int] (...) - tensor of token indices in [0, num_tokens)
        Output:
            torch.Tensor[float] (... x dim) - embedding vectors for each token id
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


@app.function
def silu(x):
    """
    Apply the SiLU nonlinearity elementwise.

    Input:
        x : torch.Tensor[float] (...) - input tensor
    Output:
        torch.Tensor[float] (...) - tensor after applying SiLU
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


@app.function
def rms_norm(X, eps=1e-5):
    """
    Apply RMS normalization along the final dimension of the input.

    Inputs:
        X : torch.Tensor[float] (... x dim) - input tensor
        eps : float - numerical stability constant
    Output:
        torch.Tensor[float] (... x dim) - RMS-normalized tensor
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_rms_norm_local():
    test_rms_norm(rms_norm)


@app.cell(hide_code=True)
def _():
    submit_rms_norm_button = mo.ui.run_button(label="submit `rms_norm`")
    submit_rms_norm_button
    return (submit_rms_norm_button,)


@app.cell
def _(submit_rms_norm_button):
    mugrade.submit_tests(rms_norm) if submit_rms_norm_button.value else None
    return


@app.function
def self_attention(Q, K, V, mask=None):
    """
    Apply scaled dot-product attention, optionally with an additive mask.

    Inputs:
        Q : torch.Tensor[float] (... x query_len x d) - query tensor
        K : torch.Tensor[float] (... x key_len x d) - key tensor
        V : torch.Tensor[float] (... x key_len x d_v) - value tensor
        mask : torch.Tensor[float] (... x query_len x key_len) or None - additive attention mask
    Output:
        torch.Tensor[float] (... x query_len x d_v) - attention output tensor
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


@app.class_definition
class MultiHeadAttentionKVCache(Module):
    def __init__(self, dim, n_heads, max_cache_size):
        """
        Initialize a multi-head self-attention layer with KV cache buffers.

        Inputs:
            dim : int - total embedding dimension
            n_heads : int - number of attention heads
            max_cache_size : int - maximum sequence length stored in the cache
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X, mask=None, seq_pos=0, use_kv_cache=False):
        """
        Apply multi-head self-attention, optionally updating and using the KV cache.

        Inputs:
            X : torch.Tensor[float] (batch_size x seq_len x dim) - input sequence embeddings
            mask : torch.Tensor[float] (seq_len x total_len) or None - additive attention mask
            seq_pos : int - starting sequence position for cached tokens
            use_kv_cache : bool - whether to update and use the KV cache
        Output:
            torch.Tensor[float] (batch_size x seq_len x dim) - attention output
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


@app.class_definition
class MLP(Module):
    def __init__(self, dim, ffn_dim):
        """
        Initialize a simple two layer feed-forward network used in the transformer block.

        Inputs:
            dim : int - model dimension
            ffn_dim : int - hidden feed-forward dimension
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X):
        """
        Apply a simple two layer feed-forward network to the input tensor.

        Input:
            X : torch.Tensor[float] (... x dim) - input tensor
        Output:
            torch.Tensor[float] (... x dim) - transformed tensor
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_MLP_local():
    test_MLP(MLP)


@app.cell(hide_code=True)
def _():
    submit_MLP_button = mo.ui.run_button(label="submit `MLP`")
    submit_MLP_button
    return (submit_MLP_button,)


@app.cell
def _(submit_MLP_button):
    mugrade.submit_tests(MLP) if submit_MLP_button.value else None
    return


@app.class_definition
class TransformerBlock(Module):
    def __init__(self, dim, n_heads, ffn_dim, max_cache_size):
        """
        Initialize a transformer block with attention, normalization, and gated MLP layers.

        Inputs:
            dim : int - model dimension
            n_heads : int - number of attention heads
            ffn_dim : int - hidden feed-forward dimension
            max_cache_size : int - maximum sequence length stored in the attention cache
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X, mask=None, seq_pos=0, use_kv_cache=False):
        """
        Apply one transformer block with residual connections.

        Inputs:
            X : torch.Tensor[float] (batch_size x seq_len x dim) - input sequence embeddings
            mask : torch.Tensor[float] (seq_len x total_len) or None - additive attention mask
            seq_pos : int - starting sequence position for cached tokens
            use_kv_cache : bool - whether to update and use the attention cache
        Output:
            torch.Tensor[float] (batch_size x seq_len x dim) - transformed sequence embeddings
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


@app.class_definition
class LLM(Module):
    def __init__(self, num_tokens, dim, n_heads, max_seq_len, ffn_dim, num_layers):
        """
        Initialize the simplified Llama 3 model used in this homework.

        Inputs:
            num_tokens : int - vocabulary size
            dim : int - model dimension
            n_heads : int - number of attention heads per layer
            max_seq_len : int - maximum supported sequence length
            ffn_dim : int - hidden feed-forward dimension in each block
            num_layers : int - number of transformer blocks
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, tokens, seq_pos=0, use_kv_cache=False):
        """
        Apply the full language model to a batch of token sequences.

        Inputs:
            tokens : torch.Tensor[int] (batch_size x seq_len) - input token ids
            seq_pos : int - starting sequence position for positional embeddings and cached attention
            use_kv_cache : bool - whether to update and use cached keys and values
        Output:
            torch.Tensor[float] (batch_size x seq_len x num_tokens) - output logits
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_LLM_local():
    test_LLM(LLM)


@app.cell(hide_code=True)
def _():
    submit_LLM_button = mo.ui.run_button(label="submit `LLM`")
    submit_LLM_button
    return (submit_LLM_button,)


@app.cell
def _(submit_LLM_button):
    mugrade.submit_tests(LLM) if submit_LLM_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part III - Training your LLM
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 7 - Cross Entropy Loss

    Implement cross entropy loss as a function (as opposed to as a PyTorch Module, which you did before).  The other difference is that this implementation lets you use _any_ size tensor with dimension $k$.  I.e., `logits` could be a `10 x 20 x 30 x 1000` real-valued tensor (where here $k=1000$), and then $y$ would be a `10 x 20 x 30` integer tensor (where you compute the cross entropy loss over every item in the tensor).  You can make this work with a similar logic to your last implementation by first reshaping `logits` to be 2D and reshaping `y` to be 1D.
    """)
    return


@app.function
def cross_entropy_loss(logits, y):
    """
    Compute average cross entropy loss over a minibatch.

    Inputs:
        logits : ND torch.Tensor[float] (... x k) - predicted logits for each example
        y : (N-1)D torch.Tensor[int] (...) - desired class for each example
    Output:
        scalar torch.Tensor[float] - average cross entropy loss
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_cross_entropy_loss_local():
    test_cross_entropy_loss(cross_entropy_loss)


@app.cell(hide_code=True)
def _():
    submit_cross_entropy_loss_button = mo.ui.run_button(
        label="submit `cross_entropy_loss`"
    )
    submit_cross_entropy_loss_button
    return (submit_cross_entropy_loss_button,)


@app.cell
def _(submit_cross_entropy_loss_button):
    mugrade.submit_tests(
        cross_entropy_loss
    ) if submit_cross_entropy_loss_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 8 - Pretokenizing data

    Rather than load the text data and tokenize during training, a common optimization (espeically if you want to iterate over different training runs) is to tokenize the data once, write those tokens to a binary file, and then have our DataLoader class operate on the tokenized representation directly.

    Implement the following function, which takes a `tiktoken` tokenizer (we'll decribe how to use that momentarily), a `.txt` filename specifying the tokens, an output filename, and chunk sizes.

    To avoid loading the entire text file into memory, you should instead read the input file file in block of size `chunk_size`.  You will then tokenize each of the blocks, append that to the output file, and continue until you have either tokenized the whole file, or you reach the `max_chunks` argument (we provide this so that you can also create a smaller tokenized version that only has a part of the dataset, for testing).

    We will use the `tiktoken` library to tokenize data.  You can load a tokenizer that uses the `gpt2` encodings (this is just some pre-defined set of tokens, it was built for the GPT2 model, but it doesn't have anything to do with that model as this point), and then tokenize text via the following commands:
    ```python
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text, allowed_special="all")
    ```
    (the `allowed_special` argument ensures that special tokens like the end-of-sequence token are properly tokenized).

    You can convert a list of tokens to a binary bytes sequence (here using 16 bit unsigned integers, because they can represent numbers up to 64K, and the GPT2 tokenizer has only 50K tokens) using the following commands:
    ```python
    np.asarray(tokens, dtype=np.uint16).tobytes()
    ```
    """)
    return


@app.function
def pretokenize_data(tokenizer, in_filename, out_filename, chunk_size=2**20, max_chunks=None):
    """
    Tokenize a text file in chunks and write the token ids to a binary file.

    Inputs:
        tokenizer : object - tokenizer with an encode() method returning token ids
        in_filename : str - input text filename
        out_filename : str - output binary filename for uint16 tokens
        chunk_size : int - number of text characters to read at a time
        max_chunks : int or None - maximum number of chunks to tokenize
    Output:
        None - writes tokenized data to out_filename
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_pretokenize_data_local():
    test_pretokenize_data(pretokenize_data)


@app.cell(hide_code=True)
def _():
    submit_pretokenize_data_button = mo.ui.run_button(
        label="submit `pretokenize_data`"
    )
    submit_pretokenize_data_button
    return (submit_pretokenize_data_button,)


@app.cell
def _(submit_pretokenize_data_button):
    mugrade.submit_tests(
        pretokenize_data
    ) if submit_pretokenize_data_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If you have properly implemented this function, then you should be able to tokenize the start of the Tiny Stories dataset using the following code.
    """)
    return


@app.cell(hide_code=True)
def _():
    pretokenize_small_button = mo.ui.run_button(
        label="Pretokenize a small slice of Tiny Stories"
    )
    pretokenize_small_button
    return (pretokenize_small_button,)


@app.cell
def _(pretokenize_small_button):
    mo.stop(not pretokenize_small_button.value)

    tokenizer = tiktoken.get_encoding("gpt2")
    pretokenize_data(
        tokenizer,
        "TinyStoriesV2-GPT4-train.txt",
        "TinyStoriesV2-GPT4-train.small.bin",
        chunk_size=2**20,
        max_chunks=2,
    )
    return (tokenizer,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 9 - Data Loader

    Now we will implement a DataLoader class that will serve as the may way to feed tokens.  This class will work a lot like the DataLoader from HW3, but with an important difference: because the number of tokens in the entire dataset can be quite large, this setup will _only_ read the tokens from a file as they are needed, rather than read the whole file into memory first (to be fair, in this current version the max size of the tokens in memory is just 1GB, so it would be possible to keep it all in memory, but these are still useful techniques).

    As with the DataLoader in HW3, you should implement this as a [Python iterator](https://www.w3schools.com/python/python_iterators.asp) which means that in addition to the `__init__()` function, you have to implement an `__iter__()` function (which resets the iteration), and a `__next__()` function that return the next element in the dataset.

    The approach should work like this:
    1. On initialization, you should save the filename (or, if you prefer, open a file object, it doesn't really matter here), any other elements you need to store within the class, and compute the number of batches total in the dataset -- this number will be the total size of the file divided by `seq_len * batch_size * 2` where this is due to the fact that each batch has `batch_size` token sequences each of size `seq_len`, and the two factor comes from the fact that the each token is an unsigned 16 bit number (two bytes).
    2. The `__iter__()` function should reset the batch counter and return `self`.
    3. The `__next__()` function should open the file (if not open already, but be sure it close it so you don't open too many files), use `.seek()` to reach the point in the file for the current batch (`current_batch * seq_len * batch_size * 2`), then read `(seq_len+1) * batch_size` 16 bit tokens (one more than `seq_len` for each batch, because we want to load both the current and next token target for the whole sequence).  You can read tokens by reading the proper number of types and then calling the command
        ```python
        torch.tensor(np.frombuffer(bytes,dtype=np.uint16).astype(np.int64))
        ```
        You should resize this tensor to be a `batch_size x seq_len+1`, and call `.to(<device>)` on the tensor, where `<device>` is whatever parameter is passed to the initializer as `device`.  Finally, return the first `seq_len` tokens and the next `seq_len` tokens shifted by 1 as the `X,Y` return values of this loader.  Then increment the current batch_counter (if this number is larger than the number of batches, i.e., there is not enough data left to read then raise `StopIteration`).


    Note that even though the TinyStories dataset has a natural sense of a "boundary" between different stories, we don't try to align batches with the actual stories and take equally-sized chunks ignoring the dataset structure.
    """)
    return


@app.class_definition
class DataLoader:
    def __init__(self, filename, seq_len, batch_size, device="cpu"):
        """
        Initialize a sequential token data loader backed by a binary file.

        Inputs:
            filename : str - binary filename containing uint16 token ids
            seq_len : int - number of tokens per input sequence
            batch_size : int - number of sequences per minibatch
            device : str - device on which to place each minibatch tensor
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def __iter__(self):
        """
        Reset iteration state and return the iterator object.

        Output:
            DataLoader - iterator over token minibatches
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def __next__(self):
        """
        Return the next input-target token minibatch.

        Output:
            tuple(torch.Tensor, torch.Tensor) - current input tokens and next-token targets
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_DataLoader_local():
    test_DataLoader(DataLoader)


@app.cell(hide_code=True)
def _():
    submit_DataLoader_button = mo.ui.run_button(label="submit `DataLoader`")
    submit_DataLoader_button
    return (submit_DataLoader_button,)


@app.cell
def _(submit_DataLoader_button):
    mugrade.submit_tests(DataLoader) if submit_DataLoader_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 10 - Adam

    Implement the Adam Optimizer.  Recall that the Adam updates are given by:
    $$
    \begin{split}
    u & := \beta_1 u + (1-\beta_1) \nabla_W \text{Loss} \\
    v & := \beta_2 v + (1-\beta_2) \nabla_W \text{Loss}^2 \quad \text{(square applied elementwise)} \\
    \hat{u} & := u / (1 - \beta_1^t) \\
    \hat{v} & := v / (1 - \beta_2^t) \\
    w & := w - \eta \frac{\hat{u}}{\sqrt{\hat{v}} + \epsilon}  \quad \text{(division done elementwise)}  \\
    \end{split}
    $$

    You'll implement this in the same fashion on the `SGD` optimization from Homework 3: implement an `__init__()`, `__step__()` (which updates parameters in place), and `zero_grad()` (which clears the gradients of all parameters).

    We are including reminders from HW3 about some common pitfalls when implementing an optimizer in this fashion:
    - In your __init__ function, you should explicitly call `list()` on the `parameters` input to store it in your class (and in the case of Adam, form similar lists for `u` and `v` objectives).  This is because the `model.parameters()` function returns a Python generator, an object that can be iterated over _one_ time to return all its elements.  So if you only store the passed `parameters` variable and then try to iterate over it during your `zero_grad` or `step` functions, you will only iterate over the parameters one time, and thereafter there won't be any elements to iterate over.
    - You need to compute the updates to the parameters within a `torch.no_grad()` block, as shown below.  The reason for this is that otherwise, the gradient update will happen _within a automatic differentiation loop itself_, i.e., you will be computing the gradient of the entire chain of parameter updates you perform with gradient descent.  There are actually some very cool reasons why it's often useful to differentiate through an entire parameter update, but that is definitely not what we want here.
    ```python
    with torch.no_grad():
        ### parameter update here
    ```
    """)
    return


@app.class_definition
class Adam:
    def __init__(self, params, lr=1e-3, betas = (0.9, 0.999), eps=1e-8):
        """
        Initialize Adam optimizer state for a set of parameters.

        Inputs:
            params : iterable[torch.nn.Parameter] - parameters to optimize
            lr : float - learning rate
            betas : tuple(float, float) - decay rates for first and second moments
            eps : float - numerical stability constant
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def step(self):
        """
        Apply one Adam update to all stored parameters.
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def zero_grad(self):
        """
        Zero out gradients for all stored parameters when gradients exist.
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_Adam_local():
    test_Adam(Adam)


@app.cell(hide_code=True)
def _():
    submit_Adam_button = mo.ui.run_button(label="submit `Adam`")
    submit_Adam_button
    return (submit_Adam_button,)


@app.cell
def _(submit_Adam_button):
    mugrade.submit_tests(Adam) if submit_Adam_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 11 - Training Your LLM

    Finally, let's put it all together.  Train your LLM given a model, a data loader, and an optimizer.  This should now be a familiar loop:
    1. Iterate over `x,y` pairs in the data loader (these will be the tokens and shifted token targets).
    2. Apply the model to `x` and compute the cross entropy loss between the predicted outputs and `y`.  One important element to do here is to call `model(x).float()` (convert the output to a 32 bit floating point number, regardless of the model's native output), which will be very useful for accelerated training using lower precision.
    3. Call the normal form of the update:
        ```python
        opt.zero_grad()
        loss.backward()
        opt.step()
        ```

    Repeat this over the whole data loader.  Given that this can take some time for the larger models, it would be good to e.g., print the number of tokens seen and the loss achieved over the iterations.
    """)
    return


@app.function
def train_llm(model, loader, opt):
    """
    Run one full training pass over a token data loader.

    Inputs:
        model : Module - language model mapping token sequences to logits
        loader : iterable - yields minibatches of input tokens and next-token targets
        opt : optimizer - object with zero_grad() and step() methods
    Output:
        None - updates model parameters in place
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_train_llm_local():
    test_train_llm(train_llm)


@app.cell(hide_code=True)
def _():
    submit_train_llm_button = mo.ui.run_button(label="submit `train_llm`")
    submit_train_llm_button
    return (submit_train_llm_button,)


@app.cell
def _(submit_train_llm_button):
    mugrade.submit_tests(train_llm) if submit_train_llm_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 12 - Generation

    Last of all, let's adapt our `generate()` function from the previous homework slightly to the new setting.  All that needs to change here from the solution in HW4 is that:
    1. You have be sure to initailize the input tokens with the `device` keyword set to the same device as the model (e.g., obtained by `next(model.parameters()).device`)
    2. The tiktoken tokenizer has a field `tokenizer.eot_token` that is output when the text ends.  After this happens you should break.
    """)
    return


@app.function
def generate(model, prompt_tokens, tokenizer, temp=0.7, max_tokens=500, verbose=True):
    """
    Autoregressively sample tokens from a language model using its KV cache.

    Inputs:
        model : Module - language model mapping token sequences to logits
        prompt_tokens : list[int] - initial prompt tokens
        tokenizer : object - tokenizer with decode() and `eot_token`
        temp : float - sampling temperature
        max_tokens : int - maximum number of new tokens to generate
        verbose : bool - whether to print each generated token as it is sampled
    Output:
        list[int] - generated tokens, excluding the prompt tokens
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
    ### Question 13 - Training a (very bad) LLM

    Let's train an LLM using a tiny fraction of the dataset.  This won't actually be trained for long enough to output any reasonable text, but you can ensure that loss is going down (with this data loader size, it should run for 63 iterations, and achieve a loss lower than 5).
    """)
    return


@app.cell(hide_code=True)
def _():
    train_small_button = mo.ui.run_button(label="Train a very small LLM")
    train_small_button
    return (train_small_button,)


@app.cell
def _(tokenizer, train_small_button):
    mo.stop(not train_small_button.value)

    # Train a very small LLM (CPU might be slow)
    # replace with device="cuda" for GPU
    loader = DataLoader(
        "TinyStoriesV2-GPT4-train.small.bin", 512, 16, device="cpu"
    )

    model = LLM(
        num_tokens=(tokenizer.n_vocab // 256 + 1) * 256,
        dim=256,
        n_heads=8,
        max_seq_len=512,
        ffn_dim=512,
        num_layers=4,
    )  # .to("cuda") for GPU

    opt = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    train_llm(model, loader, opt)
    return (model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Run the following code to evaluate the trained model.
    """)
    return


@app.cell
def _(model):
    def eval_llm():
        return model

    return (eval_llm,)


@app.cell(hide_code=True)
def _(eval_llm):
    def test_eval_llm_local():
        test_eval_llm(eval_llm)

    return


@app.cell(hide_code=True)
def _():
    submit_eval_llm_button = mo.ui.run_button(label="submit `eval_llm`")
    submit_eval_llm_button
    return (submit_eval_llm_button,)


@app.cell
def _(eval_llm, submit_eval_llm_button):
    mugrade.submit_tests(eval_llm) if submit_eval_llm_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Finally, you can try generating some text with it.
    """)
    return


@app.cell
def _(model, tokenizer):
    small_prompt = tokenizer.encode("Once upon a time,")
    print("Once upon a time,")
    generate(model, small_prompt, tokenizer)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 14 - Training a (better) LLM

    The text generated by the last model didn't look too good.  But this is where the magic of scaling comes in: all you really need to do here is train a model on a lot more data, and it will start to get a lot better.

    This last part of the assigment isn't graded, but I would strongly suggest you try it out.  All it will take is to get a GPU runtime running on Colab (this costs around $1 worth of credits), and about an hour of time depending on which GPU you get (you'll want at least at A100, but ideally an H100 if you can get one).

    The code below will first tokenize the entire dataset (this takes about 4 minutes on colab), and then trains a slightly larger model on the entire dataset.  As it will convert the model to CUDA (Nvidia's GPU runtime), and also converts it to a 16 bit format known as "bfloat16".  Together these run about 100x faster than the CPU version, and make it possible to train the model on the entire dataset.  And this is just with one GPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_large_button = mo.ui.run_button(
        label="Tokenize full dataset and train a 121M parameter model"
    )
    train_large_button
    return (train_large_button,)


@app.cell
def _(train_large_button):
    mo.stop(not train_large_button.value)

    big_tokenizer = tiktoken.get_encoding("gpt2")
    pretokenize_data(
        big_tokenizer, "TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-train.bin"
    )

    # Train a 121M parameter model
    big_loader = DataLoader(
        "TinyStoriesV2-GPT4-train.bin", 1024, 32, device="cuda"
    )

    # models are a bit faster if dimensions are divisible by 256
    big_model = (
        LLM(
            num_tokens=(big_tokenizer.n_vocab // 256 + 1) * 256,
            dim=768,
            n_heads=12,
            max_seq_len=1024,
            ffn_dim=4 * 768,
            num_layers=6,
        )
        .to("cuda")
        .bfloat16()
    )

    big_opt = Adam(big_model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    train_llm(big_model, big_loader, big_opt)
    return big_model, big_tokenizer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Finally, try generating some text from the trained model.
    """)
    return


@app.cell
def _(big_model, big_tokenizer):
    big_prompt = big_tokenizer.encode("Once upon a time,")
    print("Once upon a time,", end="")
    generate(big_model, big_prompt, big_tokenizer, verbose=True)
    return


if __name__ == "__main__":
    app.run()
