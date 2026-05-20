# /// script
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "pytest==9.0.2",
#     "requests==2.32.5",
#     "mugrade @ git+https://github.com/locuslab/mugrade.git",
#     "torch",
#     "tiktoken",
#     "datasets",
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
            "https://raw.githubusercontent.com/modernaicourse/hw7/refs/heads/main/hw7_tests.py",
        ]
    )

    import os
    import math
    import json
    import copy
    import re
    import warnings

    import mugrade
    import torch
    import tiktoken
    from datasets import load_dataset
    from torch.nn import Module, ModuleList, Parameter, Buffer

    from hw7_tests import (
        test_Linear,
        test_Embedding,
        test_silu,
        test_rms_norm,
        test_self_attention,
        test_MLP,
        test_TransformerBlock,
        test_cross_entropy_loss,
        test_Adam,
        test_LLM,
        test_log_probs,
        test_MultiHeadAttentionKVCache,
        submit_MultiHeadAttentionKVCache,
        test_generate_parallel,
        submit_generate_parallel,
        test_gsm8k_to_text,
        submit_gsm8k_to_text,
        test_pretokenize_gsm8k,
        submit_pretokenize_gsm8k,
        test_get_loss_mask,
        submit_get_loss_mask,
        test_DataLoader,
        submit_DataLoader,
        test_train_llm_sft,
        submit_train_llm_sft,
        test_eval_tool,
        submit_eval_tool,
        test_generate_parallel_tool,
        submit_generate_parallel_tool,
        test_extract_answer,
        submit_extract_answer,
        test_grade_responses,
        submit_grade_responses,
        test_eval_model,
        submit_eval_model,
        test_rl_loss,
        submit_rl_loss,
        test_train_llm_rl,
        submit_train_llm_rl,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Homework 7 - Reasoning models and Reinforcement Learning

    In this homework, you will you supervised learning and RL to build a (very minimal, of course) reasoning model that can (sometimes) solve basic math problems and employ simple tool use.  While the actual performance of your model is going to pale in comparison to what even slightly larger LLMs can do, it will be a nice illustration of what is possible using _just_ the code that you have built so far in this course, with the caveat that the pretrained model here is trained for much longer than you were able to on your previous assignments.

    As a preview of what you will implement, you will finetune an LLM that can solve problems like this one, from the GSM8K dataset.
    ```
    Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?
    ```

    The final performance of your model may vary quite a bit (RL training in particular can be noisy), but you should expect to be able to build a system that can solve this kind of problem >3% of the time, and can solve it >30% of the time if given 32 attempts.  What's more, to do so, the generations will explicitly call out to Python, with proper arithmetic formatting, for simple arithmetic operations.
    """)
    return


@app.cell
def _():
    os.environ["MUGRADE_HW"] = "Homework 7"
    os.environ["MUGRADE_KEY"] = ""  ### Your key here
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part 0 - Older implementation

    We're going to start, as we did for the last assignment, by copying your old implementation of your LLM into this notebook.  Since these functions are identical to what we had before, they will only have local tests, and you should _not_ submit them to the assignment in mugrade.  You should use the latest versions of all these models as they were built in the HW6 assigment.

    Note that we are _leaving out_ the `MultiheadAttentionKVCache` layer and `generate` functions from this list, as you will need to modify these portions slightly in the next section to allow for parallel generation.
    """)
    return


@app.class_definition
class Linear(Module):
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_Linear_local():
    test_Linear(Linear)


@app.class_definition
class Embedding(Module):
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_Embedding_local():
    test_Embedding(Embedding)


@app.function
def silu(x):
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_silu_local():
    test_silu(silu)


@app.function
def rms_norm(X, eps=1e-5):
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_rms_norm_local():
    test_rms_norm(rms_norm)


@app.function
def self_attention(Q, K, V, mask=None):
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_self_attention_local():
    test_self_attention(self_attention)


@app.class_definition
class MLP(Module):
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_MLP_local():
    test_MLP(MLP)


@app.class_definition
class TransformerBlock(Module):
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_TransformerBlock_local():
    test_TransformerBlock(TransformerBlock)


@app.function
def cross_entropy_loss(logits, y):
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_cross_entropy_loss_local():
    test_cross_entropy_loss(cross_entropy_loss)


@app.class_definition
class Adam:
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_Adam_local():
    test_Adam(Adam)


@app.class_definition
class LLM(Module):
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_LLM_local():
    test_LLM(LLM)


@app.function
def log_probs(logits, y, mask):
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_log_probs_local():
    test_log_probs(log_probs)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part I - Parallel Sampling

    When we work on RL methods, which generally require generating (many) samples from the underlying model in order to improve performance, it is extremely helpful to be able to be able to simultaneously sample _multiple_ different completions from the same prompt.  Doing so requires that we modify the KV cache mechanism to store caches for a batch size of larger than 1 (which was hardcoded in the previous impelemtation).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 1 - Multi-batch KV Cache

    As a first step, implement a slightly modified version of of the `MultiHeadAttentionKVCache` class, which creates K and V caches of size `(max_cache_batches x max_cache_size x dim)` (instead of the previous `(1 x max_cache_size x dim)`).  The forward call should then store the KV cache for however large the input batch size is.
    """)
    return


@app.class_definition
class MultiHeadAttentionKVCache(Module):
    def __init__(self, dim, n_heads, max_cache_size, max_cache_batches=64):
        """
        Initialize a multi-head self-attention layer with KV cache buffers.

        Inputs:
            dim : int - total embedding dimension
            n_heads : int - number of attention heads
            max_cache_size : int - maximum sequence length stored in the cache
            max_cache_batches : int - maximum batch size for cache
        """
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 2 - Parallel Generation

    Implement the `generate_parallel` function below, which for a _single_ prompt, generates `num_completions` different completions.  A few notes about how this function works differently from the previous `generate()` function you implemented before.
    1. For simplicity, we don't have a `verbose` flag to print generation as it is ongoing, and thus also don't need to pass the tokenizer to the function itself.
    2. We're going to use the convention that we generate up to a _total_ of `max_tokens` (including the prompt) rather than the previous version which would generate up to `max_tokens` _additional_ tokens.
    3. For the `eot_token`, you should end generation only if _all_ the different completions contain the `eot_token`.  You can keep generating tokens as usual even after the `eot_token` while you wait for them all to be generated.

    Note that you can use the `torch.multinomial(probs, 1)` function to generate from _multiple_ different distributions.  That is, if you have something like this:
    ```
    probs = torch.tensor([
        [0.1, 0.2, 0.7],
        [0.6, 0.3, 0.1]
    ])
    torch.multinomial(probs, 1)
    ```
    then this will generate two samples, one from the probability distribution in the first row, and the second from the probability distribution in the second row.
    """)
    return


@app.function
def generate_parallel(model, prompt_tokens, num_completions=1,
                      eot_token=None, temp=0.7, max_tokens=500):
    """
    Autoregressively sample multiple completions from a language model using its KV cache.

    Inputs:
        model : Module - language model mapping token sequences to logits
        prompt_tokens : list[int] - initial prompt tokens
        num_completions : int - number of different completions to generate
        eot_token: int or None - token at which generation may stop once all completions contain it
        temp : float - sampling temperature
        max_tokens : int - maximum total number of tokens per completion, including the prompt
    Output:
        torch.Tensor[int] (num_completions x max_tokens) - prompt plus generated tokens for each completion
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_generate_parallel_local():
    test_generate_parallel(generate_parallel)


@app.cell(hide_code=True)
def _():
    submit_generate_parallel_button = mo.ui.run_button(
        label="submit `generate_parallel`"
    )
    submit_generate_parallel_button
    return (submit_generate_parallel_button,)


@app.cell
def _(submit_generate_parallel_button):
    mugrade.submit_tests(
        generate_parallel
    ) if submit_generate_parallel_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's now evaluate how will this works.  First run this following cell to download the necessary datasets and models for the rest of this assignments.
    """)
    return


@app.cell(hide_code=True)
def _():
    download_data_button = mo.ui.run_button(label="Download models and datasets")
    download_data_button
    return (download_data_button,)


@app.cell
def _(download_data_button):
    mo.stop(not download_data_button.value)

    ### Downlaod the models and datasets for the assignment
    from huggingface_hub import hf_hub_download

    repo = "zkolter/RL-Homework"
    filenames = ["model_base.pth", "model_sft.pth", "params.json"]

    for filename in filenames:
        if not os.path.exists(filename):
            hf_hub_download(
                repo_id=repo, filename=filename, repo_type="model", local_dir="."
            )

    ## download GSM8K
    if not os.path.exists("gsm8k_train.json") or not os.path.exists(
        "gsm8k_test.json"
    ):
        data = load_dataset("gsm8k", "main")
        with open("gsm8k_train.json", "wt") as _f:
            json.dump(list(data["train"]), _f, indent=4)
        with open("gsm8k_test.json", "wt") as _f:
            json.dump(list(data["test"]), _f, indent=4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The following code will download and load the base model we trained on FineWebEDU, then generate multiple completions in parallel.

    **Note:** for the majority of this assignment, we're going to directly use CUDA for all actual model training, etc.  You can still evaluate test cases on a CPU implmementation (to get full credit), but there's not much point to creating the fine-tuned version with only a few steps of optimization, as they will not work for the following portions.  When needed, we will provide variants of the models trained to that point where you can pass all the test case, but we strongly recommend getting the GPU instance of Colab for this assignment (all full training runs will take less than 10 minutes).
    """)
    return


@app.cell(hide_code=True)
def _():
    load_base_button = mo.ui.run_button(label="Load base model")
    load_base_button
    return (load_base_button,)


@app.cell
def _(load_base_button):
    mo.stop(not load_base_button.value)

    ### load the model and test generation
    with open("params.json", "rt") as _f:
        params = json.load(_f)
    model = LLM(
        params["num_tokens"],
        params["dim"],
        params["n_heads"],
        params["max_seq_len"],
        params["ffn_dim"],
        params["n_layers"],
    )
    model.load_weights("model_base.pth")
    model.float().cuda()
    return (model,)


@app.cell
def _(model):
    ### generate some text
    base_tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "In a shocking discovery,"
    generated_tokens = generate_parallel(
        model,
        base_tokenizer.encode(prompt, allowed_special="all"),
        num_completions=4,
        temp=0.4,
        max_tokens=100,
    )
    for t in generated_tokens:
        print(base_tokenizer.decode(t.tolist()), "\n")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part II - Training a reasoning model with SFT

    In this part of the assignment, you will use the GSM8K dataset and supervised finetuning to build a simple reasoning model.  The [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset contains a collection of simple math problems that require a few steps of reasoning in order to arrive at the right answer.  The dataset's entries consist of items like this:
    ```json
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
    }
    ```
    There are a few points to highlight here.  First, as indicated, the objects above have "question" and "answer" as the main keys of the json dictionary, but there is also substantial structure within the "answer" portion in particular.  Specifically, the answer portions are broken down into the following elements:
    1. Within the answer text, there are tool calls indicated by the `<< >>` tags, e.g., `<<48/2=24>>`.  In this format, the left hand side of the equation (less of the = sign) indicates the arithmetic expression to send the tool, and the element to the right represents the result of the tool call.  For instance, if you parsed this text you would input `48/2` into the tool (i.e., into a Python interpreter) the expected result would be `24` (note that when you actually generate these results, your interpreter might generate 24.0 as the answer, or have other floating point issues, but you don't need to worry about this for our purposes).
    2. The answer to the question appears always as an integer and always after the "`#### `" symbol on its own line.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 3 - Coverting GSM8K to our format

    As a first task, you'll convert the GSM8K format to a more explicit format that we can parse with our tokenizer for use with SFT and RL training.  Specifically we will use three different tags: `<QUESTION>` to specify the question, `<THINK>` to detail the solution logic, and `<ANSWER>` to indicate the integer answer alone.  In addition within the `<THINK>` tags you can use the `<TOOL>` and `<RESPONSE>` tags to indicate the call to and return from a tool call.  For example, the above format would be parsed into
    ```
    <QUESTION>Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?</QUESTION><THINK>Natalia sold 48/2 = <TOOL>48/2</TOOL><RESPONSE>24</RESPONSE>24 clips in May.\nNatalia sold 48+24 = <TOOL>48+24</TOOL><RESPONSE>72</RESPONSE>72 clips altogether in April and May.</THINK><ANSWER>72</ANSWER>
    ```
    """)
    return


@app.function
def gsm8k_to_text(message):
    """
    Convert one GSM8K example into the tagged reasoning format used in this homework.

    Input:
        message : dict[str, str] - GSM8K example with "question" and "answer" fields
    Output:
        str - formatted text containing QUESTION, THINK, TOOL, RESPONSE, and ANSWER tags
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_gsm8k_to_text_local():
    test_gsm8k_to_text(gsm8k_to_text)


@app.cell(hide_code=True)
def _():
    submit_gsm8k_to_text_button = mo.ui.run_button(label="submit `gsm8k_to_text`")
    submit_gsm8k_to_text_button
    return (submit_gsm8k_to_text_button,)


@app.cell
def _(submit_gsm8k_to_text_button):
    mugrade.submit_tests(
        gsm8k_to_text
    ) if submit_gsm8k_to_text_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 4 - Pretokenizing dataset

    Next implement the `pretokenize_gsm8k()` function, which takes the original GSM8K json files, converts them to the text format using the previous function, and then saves this as tokens for a data loader.  This should work almost exactly as the `pretokenize_chat()` function from HW6, just converting to gsm8k template instead of chat templates.
    """)
    return


@app.function
def pretokenize_gsm8k(tokenizer, in_filename, out_filename):
    """
    Convert GSM8K json examples into tokenized tagged reasoning traces.

    Inputs:
        tokenizer : object - tokenizer with encode(text, allowed_special="all")
        in_filename : str - input json filename containing GSM8K examples
        out_filename : str - output json filename for tokenized examples
    Output:
        None - writes the tokenized examples to out_filename
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_pretokenize_gsm8k_local():
    test_pretokenize_gsm8k(pretokenize_gsm8k)


@app.cell(hide_code=True)
def _():
    submit_pretokenize_gsm8k_button = mo.ui.run_button(
        label="submit `pretokenize_gsm8k`"
    )
    submit_pretokenize_gsm8k_button
    return (submit_pretokenize_gsm8k_button,)


@app.cell
def _(submit_pretokenize_gsm8k_button):
    mugrade.submit_tests(
        pretokenize_gsm8k
    ) if submit_pretokenize_gsm8k_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 5 - Loss masking

    Impelment the `get_loss_mask()` that returns a binary mask of which tokens to train on for a our GSM8K format.  The convention here is that the mask should start off false (you don't train on the question portion up to the `<THINK>` token), be true after the `<THINK>` token (this is where the LLM starts generating the answer), be false again after any `</TOOL>` token (you don't try to predict tool response), and then be true again after any `</RESPONSE>` token (after the tool call you once again start generating text).  Finally, anything after `</ANSWER>` (if it exists) would also be false.

    For example for the text above,
    ```
    <QUESTION>Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?</QUESTION><THINK>Natalia sold 48/2 = <TOOL>48/2</TOOL><RESPONSE>24</RESPONSE>24 clips in May.\nNatalia sold 48+24 = <TOOL>48+24</TOOL><RESPONSE>72</RESPONSE>72 clips altogether in April and May.</THINK><ANSWER>72</ANSWER>
    ```
    the mask would be true corresponding to the following tokens:
    ```
    Natalia sold 48/2 = <TOOL>48/2</TOOL>24 clips in May.\nNatalia sold 48+24 = <TOOL>48+24</TOOL>72 clips altogether in April and May.</THINK><ANSWER>72</ANSWER>
    ```
    """)
    return


@app.function
def get_loss_mask(tokens, tokenizer):
    """
    Build a boolean mask selecting reasoning and answer tokens to train on for GSM8K.

    Inputs:
        tokens : list[int] - tokenized GSM8K example with QUESTION, THINK, TOOL, RESPONSE, and ANSWER tags
        tokenizer : object - tokenizer with special token ids for the GSM8K tags
    Output:
        list[bool] - True on generated reasoning and answer tokens, but False on questions and tool responses
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_get_loss_mask_local():
    test_get_loss_mask(get_loss_mask)


@app.cell(hide_code=True)
def _():
    submit_get_loss_mask_button = mo.ui.run_button(label="submit `get_loss_mask`")
    submit_get_loss_mask_button
    return (submit_get_loss_mask_button,)


@app.cell
def _(submit_get_loss_mask_button):
    mugrade.submit_tests(
        get_loss_mask
    ) if submit_get_loss_mask_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If you have implemented these functions correctly, then the following code (which defines a custom tokenizer and pretokenizes the GSM8K train and test sets) will generate the files you use in the remainder of this homework.
    """)
    return


@app.cell(hide_code=True)
def _():
    pretokenize_button = mo.ui.run_button(label="Pretokenize GSM8K")
    pretokenize_button
    return (pretokenize_button,)


@app.cell
def _(pretokenize_button):
    mo.stop(not pretokenize_button.value)

    base_tokenizer_gsm = tiktoken.get_encoding("gpt2")
    tokenizer = tiktoken.Encoding(
        name="gpt2_chat",
        pat_str=base_tokenizer_gsm._pat_str,
        mergeable_ranks=base_tokenizer_gsm._mergeable_ranks,
        special_tokens={
            **base_tokenizer_gsm._special_tokens,
            "<QUESTION>": 50257,
            "</QUESTION>": 50258,
            "<THINK>": 50259,
            "</THINK>": 50260,
            "<TOOL>": 50261,
            "</TOOL>": 50262,
            "<RESPONSE>": 50263,
            "</RESPONSE>": 50264,
            "<ANSWER>": 50265,
            "</ANSWER>": 50266,
        },
    )

    pretokenize_gsm8k(
        tokenizer, "gsm8k_train.json", "gsm8k_train_tokenized.json"
    )
    pretokenize_gsm8k(tokenizer, "gsm8k_test.json", "gsm8k_test_tokenized.json")
    return (tokenizer,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 6 - Data loader for GSM8K

    In this next portion, write a DataLoader for GSM8K data.  This will be identical to the `DataLoaderChat` class you implemented in HW6, with one exception: rather than specify a maximum sequence length and always return objects that size, since GSM8K data is often much shorter, you should instead dynamically compute the maximum sequence length over the current batch, and only return `x`, `y`, and `mask` up to that sequence length (really one minus that maximum sequence length, since `x` will be all tokens to the max length minus one, and `y` and `mask` starting at the first token).
    """)
    return


@app.class_definition
class DataLoader:
    def __init__(self, filename, batch_size, tokenizer, device="cpu"):
        """
        Initialize a GSM8K data loader backed by a tokenized json file.

        Inputs:
            filename : str - json filename containing tokenized GSM8K examples
            batch_size : int - number of examples per minibatch
            tokenizer : object - tokenizer used to build reasoning and answer masks
            device : str - device on which to place each minibatch tensor
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def __iter__(self):
        """
        Reset iteration state and return the iterator object.

        Output:
            DataLoader - iterator over GSM8K minibatches
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def __next__(self):
        """
        Return the next GSM8K minibatch padded only to this batch's maximum length.

        Output:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor) - input tokens, next-token targets, and boolean loss mask
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
    ### Question 7 - Supervised finetuning

    Implement the `train_llm_sft()` function to carry out supervised finetuning given a model, a data loader, and an optimizer.  This function will be indentical to the chat training from HW6.
    """)
    return


@app.function
def train_llm_sft(model, loader, opt, max_iter=None):
    """
    Run one pass of supervised finetuning with a masked next-token loss.

    Inputs:
        model : Module - language model mapping token sequences to logits
        loader : iterable - yields minibatches of inputs, targets, and boolean loss masks
        opt : optimizer - object with zero_grad() and step() methods
        max_iter : int or None - maximum number of minibatches to process
    Output:
        None - updates model parameters in place
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_train_llm_sft_local():
    test_train_llm_sft(train_llm_sft)


@app.cell(hide_code=True)
def _():
    submit_train_llm_sft_button = mo.ui.run_button(label="submit `train_llm_sft`")
    submit_train_llm_sft_button
    return (submit_train_llm_sft_button,)


@app.cell
def _(submit_train_llm_sft_button):
    mugrade.submit_tests(
        train_llm_sft
    ) if submit_train_llm_sft_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If you have implemented all the function above, then the following code will train a simple reasoning model based upon the format we built above.  This training runs 5 epochs over the GSM8K training set, and should take 5-10 minutes to run on Colab with a GPU.  If you don't want to do this run, you can use the pre-saved `model_sft.pth` file we included, but we encourage you to run this full training to see how quickly the pretrained model can be tuned to accomplish even these fairly complex tasks, reasoning through the problem and calling tools when needed.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_sft_button = mo.ui.run_button(label="Run SFT training")
    train_sft_button
    return (train_sft_button,)


@app.cell
def _(model, tokenizer, train_sft_button):
    mo.stop(not train_sft_button.value)

    ### SFT Training of the reasoning model
    sft_loader = DataLoader(
        "gsm8k_train_tokenized.json", 16, tokenizer, device="cuda"
    )
    model.load_weights("model_base.pth")
    model.float().cuda()
    sft_opt = Adam(model.parameters(), lr=3e-5, betas=(0.9, 0.95))

    for _epoch in mo.status.progress_bar(range(5), title="SFT Training"):
        train_llm_sft(model, sft_loader, sft_opt)

    model.save_weights("model_sft.pth")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part III - Evaluating tools and reasoning models

    Unlike our previous homeworks, evaluating a reasoning model with tool calls (especially on a concrete task like GSM8K) is not as simple as just generating text from the model.  Instead, we need to generate text, intercept tool call tags as needed, run the actual Python interpreter to generate the output of the tools, insert this a tool response, and when finished, validate the final answer versus the ground truth answer in a test set.  And most of this same logic will be needed to train the RL methods themselves.  This part of the homework will build up these elements bit by bit.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 8 - Tool evaluation

    As a starting point, you'll need to design a tool evaluation function that can evaluate simple arithmetic expressions like `48/2` and `24+28` that are included in the example above.  For this assignment we'll take a (quite risky) approach as just directly evaluate these exprssions using the Python `eval()` function.  However, because GSM8K uses integers as the results of tool calls when the answer is an integer and _also_ has some intermediate calculations that produce decimal results (versus Python, which will always return floating point solutions when e.g., dividing two integers), you also need to convert the final answer to an integer _if it is close to an integer_.  Specifically, your function shoudl work as follows:
    1. Call `eval()` on `tool_call_text` to parse the result.  By the conventions of GSM8K, if the tool call is properly formatted, the result will always be a number (either an integer or a floating point number).
    2. Check if the result is within `1e-4` of an integer, which you can compute using the `math.isclose(..., abs_tol=1e-4)` function along with the built-in Python `round()` function.  If so, then return this integer value instead of the floating point value.
    3. There is, of course, the chance that your LLM will not generate valid code within the tool call.  To account for this, wrap _all_ of this logic within an `try`/`except` block, and set the response to be an error if there is any exception.  In other works this works like the following:
    ```python
    try:
        ### do the evaluation
    except:
        response = "ERROR"
    ```
    """)
    return


@app.function
def eval_tool(tool_call_text):
    """
    Evaluate an arithmetic tool call, rounding near-integer results and catching failures.

    Input:
        tool_call_text : str - arithmetic expression generated between TOOL tags
    Output:
        int, float, or str - evaluated result, rounded integer when appropriate, or "ERROR" on failure
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_eval_tool_local():
    test_eval_tool(eval_tool)


@app.cell(hide_code=True)
def _():
    submit_eval_tool_button = mo.ui.run_button(label="submit `eval_tool`")
    submit_eval_tool_button
    return (submit_eval_tool_button,)


@app.cell
def _(submit_eval_tool_button):
    mugrade.submit_tests(eval_tool) if submit_eval_tool_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 9 - Generation with tool calls

    Next, modify the `generate_parallel()` function you wrote above to include the results of tool calls.  This is a somewhat involved process, so we break down each element that you will need to do:
    1. Copy the prompt to the needed number of completions and start generating text as in the original `generate_parallel` call.
    2. If you encounter a `</TOOL>` token in any of the batches, then extract the tool call text by searching for the most recent `<TOOL>` token, using the tokenizer to decode the text between these tags, and evaluate the tool call (using the `eval_tool()` function).  If no opening `<TOOL>` tag exists then you should generate an `ERROR` response.
    3. Include the tool response inside `<RESPONSE>` `</RESPONSE>` tags, and tokenize the text into a sequence of token (let's say that the total length of these tokens is `n`).  Now, when generating the next `n` tokens for this particular element in the batch, don't include the sample produced by `torch.multinomial()`, but directly overwrite it with the tokenized response (and after you have written these `n` tokens, go back to generating text as normal).
    4. If you ever encounter an `</ANSWER>` tag in one of the batch elements, force all tokens to be zero for this particular element after that point.

    Note that all these tool calls need to be done individually for each generated completion (not all at one over all batch elements), because each generation will likely make tool calls at different points in the generation, etc.
    """)
    return


@app.function
def generate_parallel_tool(model, prompt_tokens, tokenizer, num_completions=1,
                           eot_token=None, temp=0.7, max_tokens=500):
    """
    Autoregressively sample multiple completions and inject tool responses when needed.

    Inputs:
        model : Module - language model mapping token sequences to logits
        prompt_tokens : list[int] - initial prompt tokens
        tokenizer : object - tokenizer with encode() and decode() methods
        num_completions : int - number of different completions to generate
        eot_token: int or None - token at which generation may stop once all completions contain it
        temp : float - sampling temperature
        max_tokens : int - maximum total number of tokens in each prompt and completion
    Output:
        torch.Tensor[int] (num_completions x max_tokens) - prompt plus generated tokens for each completion
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_generate_parallel_tool_local():
    test_generate_parallel_tool(generate_parallel_tool)


@app.cell(hide_code=True)
def _():
    submit_generate_parallel_tool_button = mo.ui.run_button(
        label="submit `generate_parallel_tool`"
    )
    submit_generate_parallel_tool_button
    return (submit_generate_parallel_tool_button,)


@app.cell
def _(submit_generate_parallel_tool_button):
    mugrade.submit_tests(
        generate_parallel_tool
    ) if submit_generate_parallel_tool_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 10 - Extracting and grading answers

    Implement the following two functions below:
    1. `extract_answer()` should search for any text contained between the `<ANSWER>` and `</ANSWER>` and convert this to an _integer_ value.  If any of this process fails (e.g., if there are no answer tags, or if the content between them does not parse to a valid integer), then the function shoudl return `None`.
    2. `grade_responses()` should take a 2D tensor of tokens representing multiple different generations, extract the answer of each, and compare it to the ground truth answer.  The result should be a list of response grades, where each grade is given by `correct_weight` (if the response is correct) added to `format_weight` (if the response is properly formatted, i.e., `extract_answer()` does not return `None`).  These parameters will help us evaluate proper formatting versus correct answers for the generated responses, and also be useful in reinforcement learning for designing reward functions that take these different elements into account.
    """)
    return


@app.function
def extract_answer(tokenizer, tokens):
    """
    Extract the integer answer contained between ANSWER tags.

    Inputs:
        tokenizer : object - tokenizer with decode() and special token ids for the ANSWER tags
        tokens : list[int] - token sequence potentially containing an ANSWER span
    Output:
        int or None - parsed integer answer, or None if extraction fails
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_extract_answer_local():
    test_extract_answer(extract_answer)


@app.cell(hide_code=True)
def _():
    submit_extract_answer_button = mo.ui.run_button(
        label="submit `extract_answer`"
    )
    submit_extract_answer_button
    return (submit_extract_answer_button,)


@app.cell
def _(submit_extract_answer_button):
    mugrade.submit_tests(
        extract_answer
    ) if submit_extract_answer_button.value else None
    return


@app.function
def grade_responses(tokenizer, tokens, ground_truth,
                    correct_weight=1.0, format_weight=0.0):
    """
    Score multiple generated responses by correctness and answer formatting.

    Inputs:
        tokenizer : object - tokenizer used to decode answers from token sequences
        tokens : torch.Tensor[int] (num_completions x seq_len) - generated completions to score
        ground_truth : torch.Tensor[int] (seq_len,) - reference token sequence containing the correct answer
        correct_weight : float - reward added when a completion's answer matches the ground truth
        format_weight : float - reward added when a completion contains a valid integer ANSWER span
    Output:
        list[float] - one score per completion
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_grade_responses_local():
    test_grade_responses(grade_responses)


@app.cell(hide_code=True)
def _():
    submit_grade_responses_button = mo.ui.run_button(
        label="submit `grade_responses`"
    )
    submit_grade_responses_button
    return (submit_grade_responses_button,)


@app.cell
def _(submit_grade_responses_button):
    mugrade.submit_tests(
        grade_responses
    ) if submit_grade_responses_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 11 - Evaluating the reasoning model

    Finally, put all of these elements together to write a function that evaluates the accuracy of a reasoning model on GSM8K tasks.  Your function should take a model and a data loader (with batch size 1), and for each item in the loader, you should extract the text up to and including the `<THINK>` token and treat this as the prompt to the model.  Then, generate `num_completions` different completions to the prompt and for each one, grade the number of correct and properly formatted answers.

    Your function should return three elements:
    1. The fraction of correct answers out of all `max_cases * num_completions` that you generate (this approimates the true probability of the models accuracy, or the Pass@1 rate).
    2. The fraction of all answers, again out of all `max_cases * num_completions` that are properly formatted.
    3. The fraction of all examples that had at least _one_ correct answer.  This represents what's referred to as the Pass@k rate, where here `k` is the number of completions.
    """)
    return


@app.function
def eval_model(loader, model, tokenizer,
               num_completions=32, max_tokens=200, temp=0.6, max_cases=100):
    """
    Evaluate a reasoning model by sampling multiple completions for each GSM8K prompt.

    Inputs:
        loader : iterable - GSM8K data loader with batch_size equal to 1
        model : Module - language model to evaluate
        tokenizer : object - tokenizer with the GSM8K special token ids
        num_completions : int - number of completions to generate per prompt
        max_tokens : int - maximum total number of tokens per completion
        temp : float - sampling temperature
        max_cases : int - maximum number of examples from the loader to evaluate
    Output:
        tuple(float, float, float) - pass@1 accuracy, formatting rate, and pass@k rate
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_eval_model_local():
    test_eval_model(eval_model)


@app.cell(hide_code=True)
def _():
    submit_eval_model_button = mo.ui.run_button(label="submit `eval_model`")
    submit_eval_model_button
    return (submit_eval_model_button,)


@app.cell
def _(submit_eval_model_button):
    mugrade.submit_tests(eval_model) if submit_eval_model_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After implementing all of these functions, you can now evaluate the reasoning model you trained with SFT above.  Given the fact that answers are generated randomly, there is going to be some variance in the numbers you'll see in these evaluations, but to give you a rough range of estimates, in our tests we have an accuracy of around 2-2.5%, a Pass@32 of 25-35% and a correct formatting of 70-80%.
    """)
    return


@app.cell(hide_code=True)
def _():
    eval_sft_button = mo.ui.run_button(label="Evaluate SFT model")
    eval_sft_button
    return (eval_sft_button,)


@app.cell
def _(eval_sft_button, model, tokenizer):
    mo.stop(not eval_sft_button.value)

    model.load_weights("model_sft.pth")
    model.float().cuda()
    sft_test_loader = DataLoader(
        "gsm8k_test_tokenized.json", 1, tokenizer, device="cuda"
    )
    sft_acc, sft_formatting, sft_passk = eval_model(
        sft_test_loader,
        model,
        tokenizer,
        num_completions=32,
        max_tokens=200,
        max_cases=100,
        temp=0.7,
    )
    print(
        f"Accuracy (Pass@1): {sft_acc}\nPass@32: {sft_passk}\n"
        + f"Correct format: {sft_formatting}"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part IV - Reinforcement learning

    Now that all the other components of a reasoning model are in place, let's build an RL training procedure.  The good news here is that the most difficult part of writing an RL method (sampling, appropriate tool use, and grading) are already done, and so the actual RL training component can be extremely short.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 12 - RL Loss

    First implement the loss function for an RL training loop.  Given a prompt $x$ and several different completions $y_1,\ldots,y_N$, we define the RL loss as
    $$
    \mathcal{L}_{RL} = \frac{1}{N_{tok}} \sum_{i=1}^N \log p(y_i | x) \cdot (R(x,y_i) - \bar{R})
    $$
    where $\log p(y_i|x)$ is the sum of log probabilities (on valid predicted tokens, i.e., tokens where the mask is `True`, as implemented by the `log_probs()` function), where $N_{tok}$ is the _total_ number of valid predicted tokens (i.e., the sum of `True` elements in the masks), and where $\bar{R}$ is the mean reward for this batch
    $$
    \bar{R} = \frac{1}{N} \sum_{j=1}^N R(x,y_j).
    $$
    Recall from lecture that the gradient of this loss has the same expected value as the gradient of the true loss we care about (the expected reward under the completions generated by the model).
    """)
    return


@app.function
def rl_loss(model, tokenizer, tokens, rewards):
    """
    Compute the centered policy-gradient loss for multiple sampled completions.

    Inputs:
        model : Module - language model assigning logits to token sequences
        tokenizer : object - tokenizer used to build GSM8K loss masks
        tokens : torch.Tensor[int] (num_completions x seq_len) - prompt plus sampled completion tokens
        rewards : torch.Tensor[float] (num_completions,) - reward assigned to each sampled completion
    Output:
        torch.Tensor[float] () - scalar RL loss
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_rl_loss_local():
    test_rl_loss(rl_loss)


@app.cell(hide_code=True)
def _():
    submit_rl_loss_button = mo.ui.run_button(label="submit `rl_loss`")
    submit_rl_loss_button
    return (submit_rl_loss_button,)


@app.cell
def _(submit_rl_loss_button):
    mugrade.submit_tests(rl_loss) if submit_rl_loss_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 13 - RL training

    Finally, put this all together to built your RL training loop.  Like the evaluation function, `train_llm_rl()` takes a model and a data loader (which must have batch size 1), and then iterates over the whole loader or for a maximum of `max_iter` iterations.  For each example the loader, it will actually _ignore_ the `y` and `masks` parameters returned by the loader, but instead just extract the prompt up to and including the `<THINK>` token, and generate `num_completions` completions.  Although you can experiment with this if you want  should grade these completions with the `grade_completions()` function with the proper `correct_weight` and `format_weight` parameters (this defines the reward to both prioritize correct answers and proper formatting).  Given these rewards it should compute the RL loss, and take an optimization step in the normal fashion.
    """)
    return


@app.function
def train_llm_rl(model, loader, opt, tokenizer,
                 num_completions=32, temp=0.8, max_tokens=200, max_iter=None,
                 correct_weight=1.0, format_weight=0.05):
    """
    Run one pass of reinforcement learning using sampled completions and scalar rewards.

    Inputs:
        model : Module - language model being optimized
        loader : iterable - GSM8K data loader with batch_size equal to 1
        opt : optimizer - object with zero_grad() and step() methods
        tokenizer : object - tokenizer with the GSM8K special token ids
        num_completions : int - number of completions to sample per prompt
        temp : float - sampling temperature
        max_tokens : int - maximum total number of tokens per sampled completion
        max_iter : int or None - maximum number of loader iterations to process
        correct_weight : float - reward added when a completion is correct
        format_weight : float - reward added when a completion has a valid ANSWER span
    Output:
        None - updates model parameters in place
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_train_llm_rl_local():
    test_train_llm_rl(train_llm_rl)


@app.cell(hide_code=True)
def _():
    submit_train_llm_rl_button = mo.ui.run_button(label="submit `train_llm_rl`")
    submit_train_llm_rl_button
    return (submit_train_llm_rl_button,)


@app.cell
def _(submit_train_llm_rl_button):
    mugrade.submit_tests(
        train_llm_rl
    ) if submit_train_llm_rl_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If you have implemented all of this correctly, you can use the following codee to futher train the SFT'd model using RL.  As should be apparently, RL training is _substantially_ slower than supervised finetuning, and so we're only going to train on 200 total iterations here (the additional performance you'll get from this weaker model will plateau pretty quickly anyway).  This process should take between 2-5 minutes on GPUs.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_rl_button = mo.ui.run_button(label="Run RL training")
    train_rl_button
    return (train_rl_button,)


@app.cell
def _(model, tokenizer, train_rl_button):
    mo.stop(not train_rl_button.value)

    rl_train_loader = DataLoader(
        "gsm8k_train_tokenized.json", 1, tokenizer, device="cuda"
    )
    model.load_weights("model_sft.pth")
    model.float().cuda()
    rl_opt = Adam(model.parameters(), lr=2e-6, betas=(0.9, 0.95))
    train_llm_rl(
        model, rl_train_loader, rl_opt, tokenizer, num_completions=32, max_iter=200
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And finally, we can evaluate the performance of the resulting model.  Here there is going to be a _huge_ amount of potential variation, including the possibility that some runs may end up with worse performance than the SFT model, just due to back luck from the sampling process over the course of RL.  However, in general we find that our method here gets 3-3.5% accuracy, 35-40% Pass@32, and 90+% formatting accuracy, which is a substantial relative increase over the SFT model alone (and remember, this is all evaluated on a test set different from those that the model was trained upon).
    """)
    return


@app.cell(hide_code=True)
def _():
    eval_rl_button = mo.ui.run_button(label="Evaluate RL model")
    eval_rl_button
    return (eval_rl_button,)


@app.cell
def _(eval_rl_button, model, tokenizer):
    mo.stop(not eval_rl_button.value)

    rl_test_loader = DataLoader(
        "gsm8k_test_tokenized.json", 1, tokenizer, device="cuda"
    )
    rl_acc, rl_formatting, rl_passk = eval_model(
        rl_test_loader,
        model,
        tokenizer,
        num_completions=32,
        max_tokens=200,
        max_cases=100,
        temp=0.7,
    )
    print(
        f"Accuracy (Pass@1): {rl_acc}\nPass@32: {rl_passk}\n"
        + f"Correct format: {rl_formatting}"
    )
    return


if __name__ == "__main__":
    app.run()
