# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "pytest==9.0.2",
#     "requests==2.32.5",
#     "mugrade @ git+https://github.com/locuslab/mugrade.git",
#     "torch",
#     "torchvision==0.25.0",
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
            "https://raw.githubusercontent.com/modernaicourse/hw3/refs/heads/main/hw3_tests.py",
        ]
    )

    import os
    import math
    import mugrade
    import torch
    from torch.nn import Module, ModuleList, Parameter
    from hw3_tests import (
        test_Linear,
        submit_Linear,
        test_CrossEntropyLoss,
        submit_CrossEntropyLoss,
        test_SGD,
        submit_SGD,
        test_DataLoader,
        submit_DataLoader,
        test_epoch,
        submit_epoch,
        test_eval_linear_model,
        submit_eval_linear_model,
        test_TwoLayerNN,
        submit_TwoLayerNN,
        test_eval_two_layer_nn,
        submit_eval_two_layer_nn,
        test_MultiLayerNN,
        submit_MultiLayerNN,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Homework 3 - Training models in PyTorch

    In this homework, you will implement the basic components of training ML models in PyTorch.  Unlike in the previous homework where you implemented the gradient computation manually and wrote your own training loop "from scratch" (in the sense that you didn't use modules, optimizers, etc), this homework will implement the more "standard" approach to training models in PyTorch.  Specifically, you will implement the following components:

    1. A `Linear` layer as a `Module` subclass
    2. A `CrossEntropyLoss` module
    3. A `SGD` optimizer
    4. A `DataLoader` class
    5. An `epoch` function that runs one pass over the data

    You will then use these components to train a linear model on the MNIST dataset, and then extend this to train a two-layer and multi-layer neural network.

    **Important**: For this assignment, you should use the `Module` and `Parameter` classes from `torch.nn`, but you should _not_ use any of the built-in layer, loss, or optimizer implementations from `torch.nn` or `torch.optim` (i.e., don't use `torch.nn.Linear`, `torch.nn.CrossEntropyLoss`, `torch.optim.SGD`, etc.).  The goal is for you to implement these yourself using only basic PyTorch tensor operations.
    """)
    return


@app.cell
def _():
    os.environ["MUGRADE_HW"] = "Homework 3"
    os.environ["MUGRADE_KEY"] = ""  ### Your key here
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part I. Training a linear model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 1. Linear layer

    Implement a linear layer as a `Module` subclass.  The layer should have a single `.weight` parameter (a `Parameter` object) of shape `out_dim x in_dim`.  The weight should be initialized to random Gaussian values scaled by $\sqrt{2 / \text{in\_dim}}$ (i.e., `torch.randn(out_dim, in_dim) * math.sqrt(2 / in_dim)`).

    The forward pass should compute the matrix multiplication $X W^T$ where $X$ is the input and $W$ is the weight matrix.  Note that this should work for inputs of arbitrary batch dimensions (e.g., both 2D inputs of shape `(batch, in_dim)` and 3D inputs of shape `(batch1, batch2, in_dim)` should be supported).  You can use the `@` operator for matrix multiplication, which handles broadcasting automatically.
    """)
    return


@app.class_definition
class Linear(Module):
    """
    A linear layer module that computes X @ W^T.

    Attributes:
        weight: Parameter of shape (out_dim, in_dim) initialized with
                Kaiming-style scaling sqrt(2/in_dim).
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X):
        """
        Compute the forward pass of the linear layer.

        Input:
            X: torch.Tensor of shape (..., in_dim)
        Output:
            torch.Tensor of shape (..., out_dim)
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
    ### Question 2 - Cross entropy loss

    Implement the cross entropy loss as a PyTorch `Module`.  Recall that the cross entropy loss is defined as

    $$L_{ce}(\hat{y}, y) = \frac{1}{N} \sum_{i=1}^N \left ( -\hat{y}_{i,y_i} + \log \sum_{j=1}^k \exp \hat{y}_{i,j} \right )$$

    where $\hat{y} \in \mathbb{R}^{N \times k}$ are the logits and $y \in \{0,\ldots,k-1\}^N$ are the targets.  You can use `torch.logsumexp` to compute the log-sum-exp term for numerical stability.
    """)
    return


@app.class_definition
class CrossEntropyLoss(Module):
    """
    Cross entropy loss module.
    """

    def forward(self, logits, y):
        """
        Compute the cross entropy loss.

        Input:
            logits: torch.Tensor of shape (N, k) - predicted logits
            y: torch.Tensor of shape (N,) - target class indices
        Output:
            scalar torch.Tensor - average cross entropy loss
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_CrossEntropyLoss_local():
    test_CrossEntropyLoss(CrossEntropyLoss)


@app.cell(hide_code=True)
def _():
    submit_CrossEntropyLoss_button = mo.ui.run_button(
        label="submit `CrossEntropyLoss`"
    )
    submit_CrossEntropyLoss_button
    return (submit_CrossEntropyLoss_button,)


@app.cell
def _(submit_CrossEntropyLoss_button):
    mugrade.submit_tests(
        CrossEntropyLoss
    ) if submit_CrossEntropyLoss_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 3 - Stochastic Gradient Descent

    Implement a simple SGD optimizer.  In PyTorch, the general paradigm for optimizers is as follows:

    1. Create an optimizer object, passing it the parameters of the model you want to optimize and a learning rate.
    2. In the training loop, you call `opt.zero_grad()` to zero out all gradients, compute the loss, call `loss.backward()` to compute gradients, and then call `opt.step()` to update the parameters.

    The `SGD` class should implement this interface.  A few important notes:

    - In `__init__`, you should store the parameters as a **list** (i.e., call `list()` on the parameters generator).  This is because `model.parameters()` returns a generator, and you need to iterate over it multiple times.
    - In `step()`, you should update each parameter's `.data` attribute (not the parameter itself) using `torch.no_grad()` context manager, or alternatively just modify `.data` directly, since modifying `.data` doesn't track gradients.
    - In `zero_grad()`, you should set each parameter's `.grad` to `None`.
    """)
    return


@app.class_definition
class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(self, parameters, learning_rate):
        """
        Initialize SGD optimizer.

        Input:
            parameters: iterable of Parameters to optimize
            learning_rate: float - step size for parameter updates
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def step(self):
        """
        Update all parameters using their gradients.
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def zero_grad(self):
        """
        Zero out gradients for all parameters.
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_SGD_local():
    test_SGD(SGD)


@app.cell(hide_code=True)
def _():
    submit_SGD_button = mo.ui.run_button(label="submit `SGD`")
    submit_SGD_button
    return (submit_SGD_button,)


@app.cell
def _(submit_SGD_button):
    mugrade.submit_tests(SGD) if submit_SGD_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 4 - Data Loader

    Implement a simple data loader class that iterates over a dataset in batches.  The class should implement the Python iterator protocol using `__iter__` and `__next__` methods.

    - `__init__` should store the data, labels, and batch size, and initialize any state needed for iteration.
    - `__iter__` should reset the iteration state and return `self` (so the loader can be used in a `for` loop and can be iterated multiple times).
    - `__next__` should return the next batch of `(X_batch, y_batch)` as a tuple.  When there are no more batches, it should raise `StopIteration`.  The last batch may be smaller than `batch_size` if the dataset size is not evenly divisible.
    """)
    return


@app.class_definition
class DataLoader:
    """
    Simple data loader that iterates over a dataset in batches.
    """

    def __init__(self, X, y, batch_size=100):
        """
        Initialize the data loader.

        Input:
            X: torch.Tensor - input data
            y: torch.Tensor - labels
            batch_size: int - number of examples per batch
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def __iter__(self):
        """
        Reset iteration state and return self.
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def __next__(self):
        """
        Return next batch of (X_batch, y_batch).
        Raises StopIteration when no more batches.
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
    ### Question 5 - Optimization epoch

    Implement an `epoch` function that runs one pass over the data.  The function takes a model, a data loader, a loss function, and an optional optimizer.

    - If `opt` is provided (training mode): for each batch, zero the gradients, compute the model output, compute the loss, call `loss.backward()`, and take an optimizer step.
    - If `opt` is `None` (evaluation mode): for each batch, compute the model output and loss without computing or updating gradients (use `torch.no_grad()`).

    The function should return the average loss and average error rate over the entire dataset.  The error rate is the fraction of examples where the model's prediction (argmax of output) does not match the target.

    **Important**: The average loss should be weighted by batch size (i.e., accumulate `loss * batch_size` and divide by total number of examples), and similarly for error.
    """)
    return


@app.function
def epoch(model, loader, loss, opt=None):
    """
    Run one epoch over the data.

    Input:
        model: Module - the model to train/evaluate
        loader: iterable of (X_batch, y_batch) tuples
        loss: Module - loss function
        opt: SGD or None - optimizer (None for evaluation)
    Output:
        (avg_loss, avg_error): tuple of floats
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_epoch_local():
    test_epoch(epoch)


@app.cell(hide_code=True)
def _():
    submit_epoch_button = mo.ui.run_button(label="submit `epoch`")
    submit_epoch_button
    return (submit_epoch_button,)


@app.cell
def _(submit_epoch_button):
    mugrade.submit_tests(epoch) if submit_epoch_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If you implemented all the problems above correctly, then the following two cells of code will load the data, and train a linear model on the MNIST training set.
    """)
    return


@app.cell
def _():
    from torchvision import datasets

    mnist_train = datasets.MNIST(".", train=True, download=True)
    mnist_test = datasets.MNIST(".", train=False, download=True)
    train_dataloader = DataLoader(
        mnist_train.data.reshape(-1, 784) / 255.0, mnist_train.targets
    )
    test_dataloader = DataLoader(
        mnist_test.data.reshape(-1, 784) / 255.0, mnist_test.targets
    )
    return test_dataloader, train_dataloader


@app.cell
def _(test_dataloader, train_dataloader):
    linear_model = Linear(784, 10)
    linear_opt = SGD(linear_model.parameters(), learning_rate=0.2)
    linear_loss = CrossEntropyLoss()

    for _i in range(20):
        _train_loss, _train_err = epoch(
            linear_model, train_dataloader, linear_loss, linear_opt
        )
        _test_loss, _test_err = epoch(linear_model, test_dataloader, linear_loss)
        print(
            f"Train Loss: {_train_loss:.4f}, Train Error: {_train_err:.4f}, "
            + f"Test Loss: {_test_loss:.4f}, Test Error: {_test_err:.4f}"
        )
    return (linear_model,)


@app.cell
def _(linear_model):
    def eval_linear_model():
        """Return the trained linear model."""
        return linear_model

    return (eval_linear_model,)


@app.cell(hide_code=True)
def _(eval_linear_model):
    def test_eval_linear_model_local():
        test_eval_linear_model(eval_linear_model)

    return


@app.cell(hide_code=True)
def _():
    submit_eval_linear_model_button = mo.ui.run_button(
        label="submit `eval_linear_model`"
    )
    submit_eval_linear_model_button
    return (submit_eval_linear_model_button,)


@app.cell
def _(eval_linear_model, submit_eval_linear_model_button):
    mugrade.submit_tests(
        eval_linear_model
    ) if submit_eval_linear_model_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part II - Training Neural Networks
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 6 - Two-layer neural network

    Implement a two-layer neural network as a `Module` subclass.  The network computes

    $$h(x) = W_2 \sigma(W_1 x)$$

    where $\sigma$ is the ReLU activation function, $W_1 \in \mathbb{R}^{\text{hidden\_dim} \times \text{in\_dim}}$ and $W_2 \in \mathbb{R}^{\text{out\_dim} \times \text{hidden\_dim}}$.

    You should use the `Linear` class you implemented above as the building block.  The network should have two attributes `self.linear1` and `self.linear2` that are `Linear` layers.  Use `torch.relu` (or equivalently `torch.clamp(x, min=0)`) for the ReLU activation.
    """)
    return


@app.class_definition
class TwoLayerNN(Module):
    """
    Two-layer neural network: h(x) = W2 * relu(W1 * x)
    """

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X):
        """
        Compute the forward pass of the two-layer network.

        Input:
            X: torch.Tensor of shape (..., in_dim)
        Output:
            torch.Tensor of shape (..., out_dim)
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_TwoLayerNN_local():
    test_TwoLayerNN(TwoLayerNN)


@app.cell(hide_code=True)
def _():
    submit_TwoLayerNN_button = mo.ui.run_button(label="submit `TwoLayerNN`")
    submit_TwoLayerNN_button
    return (submit_TwoLayerNN_button,)


@app.cell
def _(submit_TwoLayerNN_button):
    mugrade.submit_tests(TwoLayerNN) if submit_TwoLayerNN_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After you have implemented the layer, the following code will train this model.
    """)
    return


@app.cell
def _(test_dataloader, train_dataloader):
    nn_model = TwoLayerNN(784, 300, 10)
    nn_opt = SGD(nn_model.parameters(), learning_rate=0.3)
    nn_loss = CrossEntropyLoss()

    for _i in range(20):
        _train_loss, _train_err = epoch(
            nn_model, train_dataloader, nn_loss, nn_opt
        )
        _test_loss, _test_err = epoch(nn_model, test_dataloader, nn_loss)
        print(
            f"Train Loss: {_train_loss:.4f}, Train Error: {_train_err:.4f}, "
            + f"Test Loss: {_test_loss:.4f}, Test Error: {_test_err:.4f}"
        )
    return (nn_model,)


@app.cell
def _(nn_model):
    def eval_two_layer_nn():
        """Return the trained two-layer neural network model."""
        return nn_model

    return (eval_two_layer_nn,)


@app.cell(hide_code=True)
def _(eval_two_layer_nn):
    def test_eval_two_layer_nn_local():
        test_eval_two_layer_nn(eval_two_layer_nn)

    return


@app.cell(hide_code=True)
def _():
    submit_eval_two_layer_nn_button = mo.ui.run_button(
        label="submit `eval_two_layer_nn`"
    )
    submit_eval_two_layer_nn_button
    return (submit_eval_two_layer_nn_button,)


@app.cell
def _(eval_two_layer_nn, submit_eval_two_layer_nn_button):
    mugrade.submit_tests(
        eval_two_layer_nn
    ) if submit_eval_two_layer_nn_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 7 - Multi-layer neural network

    Implement an arbitrary-depth multi-layer neural network with ReLU activations.  The network computes

    $$h(x) = W_L \sigma(W_{L-1} \sigma(\cdots \sigma(W_1 x) \cdots))$$

    where $L$ is the number of layers and $\sigma$ is ReLU.  You should use `ModuleList` to store the list of `Linear` layers (this ensures that PyTorch can find all the parameters when you call `model.parameters()`).

    The `hidden_dims` argument is a list of integers specifying the sizes of the hidden layers.  For example, `MultiLayerNN(784, 10, [300, 200])` would create a network with layers of sizes `784 -> 300 -> 200 -> 10`.

    The network should have an attribute `self.linears` that is a `ModuleList` of `Linear` layers.
    """)
    return


@app.class_definition
class MultiLayerNN(Module):
    """
    Multi-layer neural network with ReLU activations.
    """

    def __init__(self, in_dim, out_dim, hidden_dims):
        """
        Initialize the multi-layer network.

        Input:
            in_dim: int - input dimension
            out_dim: int - output dimension
            hidden_dims: list[int] - sizes of hidden layers
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X):
        """
        Compute the forward pass of the multi-layer network.

        Input:
            X: torch.Tensor of shape (..., in_dim)
        Output:
            torch.Tensor of shape (..., out_dim)
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE


@app.function(hide_code=True)
def test_MultiLayerNN_local():
    test_MultiLayerNN(MultiLayerNN)


@app.cell(hide_code=True)
def _():
    submit_MultiLayerNN_button = mo.ui.run_button(label="submit `MultiLayerNN`")
    submit_MultiLayerNN_button
    return (submit_MultiLayerNN_button,)


@app.cell
def _(submit_MultiLayerNN_button):
    mugrade.submit_tests(
        MultiLayerNN
    ) if submit_MultiLayerNN_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We won't require any further tests, but play around with a few different networks to see how low you can get the loss.  For example, you might try training a `MultiLayerNN` with different hidden layer configurations and learning rates.
    """)
    return


if __name__ == "__main__":
    app.run()
