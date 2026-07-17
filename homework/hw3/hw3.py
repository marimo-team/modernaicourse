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

    In this homework, we'll start to build and train machine learning models (both a linear model a neural network) using PyTorch.  While a lot of the code you will develop here corresponds to existing implementations in the PyTorch `nn` module, you will implement almost everything from scratch in these assignments, rather than use pre-built layers.  Specifically you will use only the `Module`, and `Parameter` classes from PyTorch (later assignments will also use `ModuleList` and `Buffer`), and everything else should be implemented just with the calls included in the base `torch` library.

    ***Important:*** **To be very explicit, you solutions in this, and all later problem sets, should _not_ use any classes or functions from within the `torch.nn` module, nor function calls from the `torch.nn.functional` module.  You should only use calls available in the base `torch.` module.**

    If you are curious, you can look at the `hw3_tests.py` file to see how we evaluate these tests, where (in the `test_` functions), we essentially are comparing the methods to the equivalent PyTorch operations.  This illustrates how your own methods implementations are exactly mirroring those in PyTorch.
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
    ## Part I - Training a linear model

    To begin, we'll implement a linear model trained via (stochastic) gradient descent, and then use it to train a classifier for the same MNIST digit prediction task you completed in the last homework.  The only significant difference is that we are going to implement all of this in an idiomatic PyTorch fashion.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 1 - Linear layer

    As a first exercise, implement a linear layer in PyTorch.  We in fact did this in class, so you can largely use the example there, but it might help to have a review of the important elements here.  Here are the key points about implementing such a linear layer.

    - All PyTorch layers are implemented as a subclasses of the `Module` class.  This class implements a few things: 1) it lets you instatiate layers as class instances, and apply these layers to inputs or intermediate units in the network; 2) it encapsulates the parameters of that layer (e.g., the weights that you will be training), and recursively tracks parameters of any module included in the class.
    - The `Parameter` class is a simple wrapper you can provide apply to a tensor within that class, such that the layer will track these parameters (in any layer that includes it), and compute gradients of the parameters by default.
    - Generally, you need to implement two functions in a module class: the `__init__()` function and the `forward()` function.  The former is called to initialize your layer, set up the parameters, etc, and the latter is called when you apply the layer to some input.

    For the linear layer in particular:

    - You should store the weights is a `.weight` Parameter in the class, which should be an `out_dim x in_dim` dimensional tensor.  You should initialize with the $\sqrt{2/\text{in\_dim}}$ scaling of random Gaussian weights that we discussed in class.
    - The forward call always takes a batch of examples, i.e. a `batch_size x in_dim` tensor, and should return a `batch_size x out_dim` tensor.

    Remember, do _not_ use the `nn.Linear` layer in PyTorch, but rather you should implement this just using functions available in the top-level `torch` library.
    """)
    return


@app.class_definition
class Linear(Module):
    def __init__(self, in_dim, out_dim):
        """
        Initialize a linear layer with Gaussian weights scaled by sqrt(2/in_dim).

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
            X : torch.Tensor[float] (batch_size x in_dim) - input tensor
        Output:
            torch.Tensor[float] (batch_size x out_dim) - transformed tensor
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

    Implement cross entropy loss as a PyTorch module.  This function likely doesn't really need to be a layer (a function would be fine), but it's a common enough to also use a module (e.g., there is the `nn.CrossEntropyLoss` module in PyTorch, though of course you should not use this).  Given a `batch_size x k` real-valued tensor `logits`, where the predicted outputs for each example are stored in the rows, and a `batch_size` dimensional tensor of integer values `y` denoted the desired discrete outputs, the `forward()` method of the function should return the average cross entropy loss.
    """)
    return


@app.class_definition
class CrossEntropyLoss(Module):
    def forward(self, logits, y):
        """
        Compute average cross entropy loss over a minibatch.

        Inputs:
            logits : 2D torch.Tensor[float] (N x k) - predicted logits for each example
            y : 1D torch.Tensor[int] (N) - desired class for each example
        Output:
            scalar torch.Tensor[float] - average cross entropy loss
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
    ### Stochastic Gradient Descent

    Next, you'll implement stochastic gradient descent as a class similar to the analogous class in the PyTorch.  This is not at `Module` class, and in fact rather than subclass the analogous `Optimizer` class in PyTorch, we'll just define the class directly, and use a similar interface to what PyTorch uses in its optimizers.

    In the standard optimizer paradigm of PyTorch, an optimizer is used like the following:
    ```python
    ### initialization
    opt = Optimizer(model.parameters(), *optimizer_settings)

    ### in your training loop:
    # compute loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    ```
    When initializing the optimizer, you pass the model parameters it should be optimizing, usually from the `.parameters()` call of your final model `Module`, plus any other settings like step size for the optimizer.  Then, during optimization, you first call `.zero_grad()`, which zeros out all the `.grad` variables (if they exist) of all the parameters, then compute the gradients using PyTorch's automatic differentiation (called via the `.backward()` ), and call the optimizer's `.step()` function, which modifies the parameters with the optimization update, e.g. a gradient descent step.

    Implement these operations in the class below.  There are a few pitfalls to keep in mind:

    - In your `__init__` function, you should explicitly call `list()` on the `parameters` input to store it in your class.  This is because the `model.parameters()` function returns a Python generator, an object that can be iterated over _one_ time to return all its elements.  So if you only store the passed `parameters` variable and then try to iterate over it during your `zero_grad` or `step` functions, you will only iterate over the parameters one time, and thereafter there won't be any elements to iterate over.
    - You need to compute the updates to the parameters within a `torch.no_grad()` block, as shown below.  The reason for this is that otherwise, the gradient update will happen _within a automatic differentiation loop itself_, i.e., you will be computing the gradient of the entire chain of parameter updates you perform with gradient descent.  There are actually some very cool reasons why it's often useful to differentiate through an entire parameter update, but that is definitely not what we want here.
    ```python
    with torch.no_grad():
        ### parameter update here
    ```
    """)
    return


@app.class_definition
class SGD:
    def __init__(self, parameters, learning_rate):
        """
        Initialize an SGD optimizer over a set of model parameters.

        Inputs:
            parameters : iterable[torch.nn.Parameter] - parameters to optimize
            learning_rate : float - gradient descent step size
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def step(self):
        """
        Apply one SGD update to all stored parameters.
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

    Finally, you'll implement what's known as a DataLoader for your problem.  In idiomatic PyTorch, a data loader is a class that you can iterate over to get all the minibatches of a dataset.  You can use one with code that looks something like that following (there's slight differences when it comes to how to initialize the data loader from a dataset, and you'll have to write another once when considering LLMs, but this is the basic approach).

    ```python
    ### initialize data loader, where X_full and y_full are complete dataset
    loader = DataLoader(X_full, y_full, batch_size=100)

    ### to iterate over the dataset
    for X,y in loader:
        ### X,y contain each sequential sequential from X_full,y_full
    ```

    You have to implement this using what's known as a Python iterator.  This is a somewhat complex topic, and we won't cover iterators in generality at all, you just have to know the following.  In addition to the `__init__` routine, you need to implement two functions:

    - `__iter__()` resets the iteration (i.e., somehow indicates that we are in minibatch number 0), and returns the class object `self`
    - `__next__()` returns the current minibatch and increments the minibatch counter.  If there are no minibatches left, it calls `raise StopIteration`

    You can read more about Python iterators [here](https://www.w3schools.com/python/python_iterators.asp).
    """)
    return


@app.class_definition
class DataLoader:
    def __init__(self, X, y, batch_size=100):
        """
        Initialize a simple sequential minibatch data loader.

        Inputs:
            X : 2D torch.Tensor[float] - (N x n) full input dataset
            y : 1D torch.Tensor[int] - (N elements) full set of desired outputs
            batch_size : int - number of examples per minibatch
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def __iter__(self):
        """
        Reset iteration state and return the iterator object.

        Output:
            DataLoader - iterator over minibatches
        """
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def __next__(self):
        """
        Return the next minibatch or raise StopIteration when exhausted.

        Output:
            tuple(torch.Tensor, torch.Tensor) - next (X_batch, y_batch)
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
    ### Optimization epoch

    Finally, let's implement a routine that actually runs an epoch of optimization on a given dataset.  It's convenient (though we'll change this a bit when we do LLM training, since that it typically run single-epoch) to implement an `epoch()` function that carries out a single pass over the data, because this can be used both perform one epoch of optimization on the training set _and_ to compute test loss/error on a held out test set (making sure we don't actually run the optimization then).

    Implement the function below.  This function takes three arguments a model, a loader, a loss, and an (optional) optimizer.  An example usage would be:
    ```python
    model = Linear(n,k)
    loader = DataLoader(X_full,y_full)
    loss = CrossEntropyLoss()
    opt = SGD(model.parameters(), learning_rate=0.1)
    avg_loss, avg_error = epoch(model, loader, loss, opt) # or without opt if just evaluating
    ```
    The basic approach to implement is as follows:

    - Iterate over all minibatches in the data loader
    - For each minibatch, compute the model's predicted outputs and the loss between the predicted and desired outputs.
    - If `opt` is not none, update the model parameters using the optimization
    - Over the entire data loader, capture a running total of the total loss and total error over all samples, then return the average loss and average error.

    Note that epoch needs to return the average loss and average error as _floats_ not as torch tensors.  You can use the `.item()` call on a scalar torch tensor to return the floating point value.
    """)
    return


@app.function
def epoch(model, loader, loss, opt=None):
    """
    Run one full pass through a dataset, with optional optimization.

    Inputs:
        model : Module - model mapping inputs to logits
        loader : iterable - yields minibatches (X, y)
        loss : Module - loss function taking (logits, y)
        opt : optimizer or None - if provided, run gradient updates each minibatch
    Output:
        tuple(float, float) - average loss and average error over the epoch
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


@app.function(hide_code=True)
def train_model(model, train_dataloader, test_dataloader, lr, n_epochs=20):
    """Train a model and return it along with final metrics."""
    opt = SGD(model.parameters(), learning_rate=lr)
    loss_fn = CrossEntropyLoss()
    for _i in mo.status.progress_bar(range(n_epochs), title="Training"):
        train_loss, train_err = epoch(model, train_dataloader, loss_fn, opt)
        test_loss, test_err = epoch(model, test_dataloader, loss_fn)
        print(
            f"Train Loss: {train_loss:.4f}, Train Error: {train_err:.4f}, "
            + f"Test Loss: {test_loss:.4f}, Test Error: {test_err:.4f}"
        )
    return model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If you implemented all the problems above correctly, then the following cells of code will load the data, and train a linear model on the MNIST training set (click the button to start training).  You can then use the submit button below to evaluate the learned model.
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


@app.cell(hide_code=True)
def _():
    train_linear_button = mo.ui.run_button(label="Train linear model")
    train_linear_button
    return (train_linear_button,)


@app.cell
def _(train_linear_button, test_dataloader, train_dataloader):
    mo.stop(not train_linear_button.value)
    linear_model = train_model(
        Linear(784, 10), train_dataloader, test_dataloader, lr=0.2
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

    Now that you have the basic scaffolding for training a linear model in PyTorch, one of the nice features of this kind of modular framework, is that it is quite easy to extend this to train other models, just by swapping in a different model (i.e., a different Module subclass instance) as the `model` parameter in `epoch()`.  As an illustration, implement next a two layer neural network.  This should consist of two linear layers, stored as members `.linear1` and `.linear2` in the class, of the appropriate dimensions.  The network should implement the model

    $$h(x) = W_2 \sigma(W_1x)$$

    where $\sigma$ is the ReLU nonlinearity, i.e., the two linear layers with a ReLU nonlinearity.  Be sure to make `.linear1` and `.linear2` instances of the `Linear` class you built above, rather than just storing their weights.  This is an important illustration of the value in using sublayers (if you stored the weights directly in the class, you would need, for instance, to rewrite the logic for initializing weights).
    """)
    return


@app.class_definition
class TwoLayerNN(Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        """
        Initialize a two-layer ReLU neural network.

        Inputs:
            in_dim : int - input feature dimension
            hidden_dim : int - hidden layer dimension
            out_dim : int - output feature dimension
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X):
        """
        Apply two linear layers with ReLU between them.

        Input:
            X : torch.Tensor[float] (batch_size x in_dim) - input tensor
        Output:
            torch.Tensor[float] (batch_size x out_dim) - output logits
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
    After you have implemented the layer, click the button below to train this model.  You should be able to achieve a test error less than 2% on MNIST.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_nn_button = mo.ui.run_button(label="Train two-layer neural network")
    train_nn_button
    return (train_nn_button,)


@app.cell
def _(train_nn_button, test_dataloader, train_dataloader):
    mo.stop(not train_nn_button.value)
    nn_model = train_model(
        TwoLayerNN(784, 300, 10), train_dataloader, test_dataloader, lr=0.3
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
    Finally, implement an arbitrary multi-layer neural network.  This would represent the multi-layer deep ReLU network

    $$h(x) = W_{L} \sigma(W_{L-1} \sigma ( \ldots W_2 \sigma(W_1 x) \ldots))$$

    where $\sigma$ again is the ReLU nonlinearity.

    This class is initialized by passing the input and output dimensions, along with list of all hidden dimensions (i.e., the dimensionality of the inner activations in the network).  Your class should create a single `.linears` element which is a `ModuleList` of each `Linear` module of the appropriate size.
    """)
    return


@app.class_definition
class MultiLayerNN(Module):
    def __init__(self, in_dim, out_dim, hidden_dims):
        """
        Initialize a deep ReLU network with arbitrary hidden dimensions.

        Inputs:
            in_dim : int - input feature dimension
            out_dim : int - output feature dimension
            hidden_dims : list[int] - hidden layer widths in order
        """
        super().__init__()
        ### BEGIN YOUR CODE
        pass
        ### END YOUR CODE

    def forward(self, X):
        """
        Apply all hidden linear layers with ReLU, then final linear output layer.

        Input:
            X : torch.Tensor[float] (batch_size x in_dim) - input tensor
        Output:
            torch.Tensor[float] (batch_size x out_dim) - output logits
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
    We won't require any further tests, but play around with a few different networks to see how low you can get the loss (the lowest we have gotten is a test error of around 1.5%, but this winds up actually depending quite a bit on the random initialization).  Let us know what you get!
    """)
    return


if __name__ == "__main__":
    app.run()
