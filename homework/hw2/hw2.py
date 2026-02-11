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

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo

    import pytest
    import subprocess

    # Run this cell to download and install the necessary modules for the homework
    subprocess.call(
        [
            "wget",
            "-nc",
            "https://raw.githubusercontent.com/modernaicourse/hw2/refs/heads/main/hw2_tests.py",
        ]
    )

    import os
    import mugrade
    import math
    import torch
    from torchvision import datasets
    from hw2_tests import (
        test_Add,
        submit_Add,
        test_Subtract,
        submit_Subtract,
        test_Divide,
        submit_Divide,
        test_Power,
        submit_Power,
        test_Log,
        submit_Log,
        test_Exp,
        submit_Exp,
        test_compute_gradients,
        submit_compute_gradients,
        test_cross_entropy_loss,
        submit_cross_entropy_loss,
        test_error,
        submit_error,
        test_train_sgd,
        submit_train_sgd,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Homework 2 - Automatic differentiation and linear models

    This homework contains two main portion.  In part one, you will implement an extremely minimal automatic differentiation module.  This is the same technique that underlies PyTorch, and while you will not implement anything close to the complexity of a library like PyTorch, it will give you a basic understanding of the basic principles of the approach, giving you some insight into how the nuts and bolts of PyTorch do work under the hood.  In the second part, you will use the (built in) automatic differentiation tooling of PyTorch to train a simple linear model; note that for this assignment you won't do this in the "normal" PyTorch way of defining PyTorch Module subclasses, optimizer subclasses, and this sort of thing (that will be done in the next homework, on neural networks), but you will use the basic gradient descent approach to build a simple linear model.
    """)
    return


@app.cell
def _():
    os.environ["MUGRADE_HW"] = "Homework 2"
    os.environ["MUGRADE_KEY"] = ""  ### Your key here
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part I - Automatic differentiation

    As the core of automatic differentiation is a technique that builds a _compute graph_, which constructs a graph out of a series of functions applied to variables.  In our setting, we will implement this functionality with two simple classes: a `Variable` class that represents the variables will will differentiate with respect to and a `Function` class that contains the logic to both implement the function itself and compute its gradient.
    """)
    return


@app.class_definition
class Function:
    """
    Base class for automatic differentiation functions. Subclasses must
    implement forward() and backward().
    """

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad, *args):
        raise NotImplementedError


@app.class_definition
class Variable:
    def __init__(self, value, function=None, parents=None):
        """
        Initialize the variable with its needed properties.
        """
        self.value = value
        self.grad = None
        self.function = function
        self.parents = parents
        self.num_children = 0

    @staticmethod
    def _apply(fn, *args):
        """Construct a node in the computation graph by applying fn to args."""
        value = fn.forward(*[a.value for a in args])
        for p in args:
            p.num_children += 1
        return Variable(value, function=fn, parents=args)

    ### these functions will call later implementations you develop
    def __repr__(self):
        return f"Variable({self.value}, grad={self.grad})"

    def __add__(self, other):
        return Variable._apply(Add(), self, other)

    def __sub__(self, other):
        return Variable._apply(Subtract(), self, other)

    def __mul__(self, other):
        return Variable._apply(Multiply(), self, other)

    def __truediv__(self, other):
        return Variable._apply(Divide(), self, other)

    def __neg__(self):
        return Variable._apply(Negate(), self)

    def __pow__(self, d):
        return Variable._apply(Power(d), self)

    def log(self):
        return Variable._apply(Log(), self)

    def exp(self):
        return Variable._apply(Exp(), self)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's discuss in some detail how the automatic differentiation works, which is best understood by an example.  Let's consider the following operation for two variables `x` and `y`:
    ```
    (x*y + x**2)/y
    ```
    To be even more explicit, let's assign names to all the intermediate terms generated by this computation
    ```
    a = x*y
    b = x**2
    c = a + b
    d = c / y
    ```
    Such a operation would correspond to the following computation graph:

    ![computation graph](https://raw.githubusercontent.com/modernaicourse/hw2/refs/heads/main/computation_graph.svg)

    In this graph, the original variables and intermediate terms are represented as nodes, and the parents of a node represent the variables that were used to compute that term.  Although not depicted in the graph, each variable in the graph also stores a link to the function that created that variable (as a function of its parents).

    In our code, these computations graphs are modeled implicitly via the `Variable` class.  Specifically, the class contains the following items:

    - `.value` : a `float` value that contain the numerical value of the variable
    - `.grad` : a `float` value (or `None`) that will be populated with the variable's derivative with respect to a final function
    - `.parents` : the parents of the node in the graph or `None` if the node has no parents
    - `.function` : a reference to the function that was used to create the note from its parents (which will be a reference to an instance of the `Function` class)
    - `.num_children` : the number of children that each node has (this will be needed for counting whether all children have already computed their gradient, we could compute it online, but this would make the code more complex)

    In addition to the `Variable` class, there is also a `Function` class that creates variables in a fashion that builds the graph.  Specifically, the `__call__` method of the class implements the routine that constructs the graph. This lets you call a function like multiplication in the following manner:
    ```python
    Multiply()(x,y)
    ```
    which initializes the `Multiply()` class and then calls the resulting with arguments `x` and `y`, which invokes the `__call__()` function.

    Subclasses of `Function` need to implement two methods:

    1. The `.forward()` method actually computes the function.  For instance, the forward method of a `Multiply` class would multiply two numbers together, the forward pass of the `Log` class would take the log of a variable, etc.  As you see from the implementation above, this forward call is called by the `__call__()` class, but with additional code that constructions the graph.
    2. The `.backward()` function computes the _product_ of what's referred to as an "incoming derivative" term (this will correspond to the already-computed derivative of nodes later in the graph), and the _partial derivatives_ of this function.  In general, the arguments to the backward function will always be both this incoming derivative and the arguments to the original function.  For example, if we consider some function of two variables $f(x,y)$, and incoming derivative $g \in \mathbb{R}$, then the `.backward()` function would compute two separate product of partial derivatives, which are returned as a list.

    $$ \frac{\partial f(x,y)}{\partial x} \cdot g, \;\; \frac{\partial f(x,y)}{\partial y} \cdot g $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Example: multiplication
    Let's see how these look in a few examples.  First let's show the implementation of a `Multiply` function:
    """)
    return


@app.class_definition
class Multiply(Function):
    def forward(self, x, y):
        """
        Compute the forward pass, in this case multiplying x and y
        Input:
            x: float - first argument
            y: float - second argument
        Output:
            float - equal to x * y
        """
        return x * y

    def backward(self, grad, x, y):
        """
        Compute the product of grad and each partial derivative of the function.
        Input:
            grad: float - incoming derivative
            x: float - first argument (to the original forward call)
            y: float - second argument (to the original forward call)
        Output:
            list[float] - list of floats for the products of grad and each
                          partial derivative of the function
        """
        return [y * grad, x * grad]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In this case, we are defining the function

    $$f(x,y) = x y.$$

    The `.forward()` function simply implements this multiplication, `x*y`.  Furthermore, the partial derivatives of this particular function are given by

    $$\frac{\partial f(x,y)}{\partial x} = y, \;\; \frac{\partial f(x,y)}{\partial y} = x.$$

    Thus the `.backward()` function computes the product of the incoming derivative `grad` and each of these partial derivatives, and return them as a list.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Example: Negation

    Let's look at one more example, this time a function with a single argument, given by $f(x) = -x$.  In this case, the partial derivative is given by

    $$\frac{\partial f(x)}{\partial x} = -1$$

    so the `.backward()` function returns `[-grad]` (note that the backward function _always_ returns a list, in this case of just one element, even if the function has only one argument).
    """)
    return


@app.class_definition
class Negate(Function):
    def forward(self, x):
        """
        Compute the forward pass, in this case negating x
        Input:
            x : float - argument to function
        Output:
            return float - negation of x
        """
        return -x

    def backward(self, grad, x):
        """
        Compute product of incoming derivative grad and partial derivative
        Input:
            grad: float - incoming derivative
            x: float - argument (to the original forward call)
        Output:
            list[float] - list of a single float for the product of grad and
                          partial derivative of the function
        """
        return [-grad]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 1 - Function implementations

    Implement the following `Function` classes to complete the implementation of the operators listed in the `Variable` class.  We are just providing the class definition itself: you'll need to define and implement `.forward()` and `.backward()` functions in each of these.  Remember that `.backward()` needs to always return a _list_ of products between the incoming derivative and each partial derivative, even if there is only a single argument.

    The one slightly-less straightforward implementation here is the `Power` class, which computes the operation `x**d`.  In this case, we won't actually enable differentation with respect to the `d` variable (we certainly could, it's just a slightly more involved implementation function, so we don't do it in this assignment).  Thus, for this implementation, we'll store the `d` value in the class itself, and pass it to the `__init__()` operation of the function.  This is what's done in the `Variable` class above, i.e., whereas we call the `Mulitply` class like the following:
    ```python
    Multiply()(x,y)
    ```
    (i.e., we initialize the class, then call it with the `x` and `y` arguments).  You would call `Power` via the following:
    ```python
    Power(d)(x)
    ```
    """)
    return


@app.class_definition
class Add(Function):
    """
    Implements addition between two variables f(x,y) = x + y
    """

    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_Add_local():
    test_Add(Add)


@app.cell(hide_code=True)
def _():
    submit_Add_button = mo.ui.run_button(label="submit `Add`")
    submit_Add_button
    return (submit_Add_button,)


@app.cell
def _(submit_Add_button):
    mugrade.submit_tests(Add) if submit_Add_button.value else None
    return


@app.class_definition
class Subtract(Function):
    """
    Implements subtraction between two variables f(x,y) = x - y
    """

    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_Subtract_local():
    test_Subtract(Subtract)


@app.cell(hide_code=True)
def _():
    submit_Subtract_button = mo.ui.run_button(label="submit `Subtract`")
    submit_Subtract_button
    return (submit_Subtract_button,)


@app.cell
def _(submit_Subtract_button):
    mugrade.submit_tests(Subtract) if submit_Subtract_button.value else None
    return


@app.class_definition
class Divide(Function):
    """
    Implements division between two variables f(x,y) = x / y
    """

    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_Divide_local():
    test_Divide(Divide)


@app.cell(hide_code=True)
def _():
    submit_Divide_button = mo.ui.run_button(label="submit `Divide`")
    submit_Divide_button
    return (submit_Divide_button,)


@app.cell
def _(submit_Divide_button):
    mugrade.submit_tests(Divide) if submit_Divide_button.value else None
    return


@app.class_definition
class Power(Function):
    """
    Implements the power function between two variables f(x) = x^d.  Since the
    function does _not_ need to provide a derivative with respect to the the
    d argument, you should instead implement an __init__ function that stores
    the d variable as a member, and use this in the forward pass.  The final
    usage of the class will then be what is done by our `Variable`
    implementation above.

    Be sure to handle the case where d is zero (i.e., the function is equal to
    one).
    """

    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_Power_local():
    test_Power(Power)


@app.cell(hide_code=True)
def _():
    submit_Power_button = mo.ui.run_button(label="submit `Power`")
    submit_Power_button
    return (submit_Power_button,)


@app.cell
def _(submit_Power_button):
    mugrade.submit_tests(Power) if submit_Power_button.value else None
    return


@app.class_definition
class Log(Function):
    """
    Implements the (natural) logarithm of a function f(x) = log(x).  You can
    use calls from the math package to implement this.
    """

    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_Log_local():
    test_Log(Log)


@app.cell(hide_code=True)
def _():
    submit_Log_button = mo.ui.run_button(label="submit `Log`")
    submit_Log_button
    return (submit_Log_button,)


@app.cell
def _(submit_Log_button):
    mugrade.submit_tests(Log) if submit_Log_button.value else None
    return


@app.class_definition
class Exp(Function):
    """
    Implements the exponential (with base e) of x, f(x) = e^x.  You can use
    calls from the math package to implement this.
    """

    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_Exp_local():
    test_Exp(Exp)


@app.cell(hide_code=True)
def _():
    submit_Exp_button = mo.ui.run_button(label="submit `Exp`")
    submit_Exp_button
    return (submit_Exp_button,)


@app.cell
def _(submit_Exp_button):
    mugrade.submit_tests(Exp) if submit_Exp_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If you implement all the following arguments above, you should be able to run the following code that will implicitly build a compute graph of the following expression we described above.
    """)
    return


@app.cell
def _():
    x = Variable(3.0)
    y = Variable(5.0)
    d = (x * y + x**2) / y
    print(d)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 2 - Implementing the full backward pass

    Given the functions defined above, now you'll implement the actual automatic differentiation pass that will extend the Variable class to compute gradients in a computation graph.  This is done with a `compute_gradients()` function called on the `Variable` instance [Note: in PyTorch this function is also called `.backward()`, but that can be a bit confusing since the method in `Function` is called the same thing, so I want to differentiate.]

    The basic algorithm is quite simple, and we'll outline it here, then explain some of the intuition behind it.  Note that this implementation makes a very simplifying assumptions (for instance, it requires that variables not be used anywhere _except_ in the computation of the final output we are differentiating, or their `num_children` could will never reach zero), and it doesn't e.g., let you compute gradients multiple times (because it actively manipulates the `num_children` counters) but it nonetheless works for our simple cases.  The algorithm is as follows:

    #### Backward pass `compute_gradients()` called on `Variable` instance:
    1. If the `grad` variable of node is `None` (i.e., this is the final function being differentiated), set to 1.0.
    2. If the node has `parents` and `function` values (i.e., it is a computed node):
        1. Call the function's `backward()` implementation passing the node's `grad` value and the parent's values. This returns a list of products `grad` and the partial derivatives of the function.
        2. For each of the node's parents:
            - Add the corresponding grad/partial derivative product to the parent's `.grad` property (or set it if the `grad` is currently None).
            - Decrease the `num_children` parameter of the parent
            - If the parent's `num_children` is zero, call it's `.compute_gradient()` method recursively.


    ### Worked-through example

    Let's see how this works in a slightly different version of our example above (we'll ignore that last computation step, just to keep things simpler).  Note that going through all of this at first may not be needed, but it can be helpful to go through as you debug your implementation.
    ```
    a = x*y
    b = x**2
    c = a + b
    ```
    After we construct the computation graph, say with values `x=3.0`, `y=4.0`, it would have the following values
    ```
    x = Variable(value = 3.0, grad=None, parents = None, function = None, num_children=2)
    y = Variable(value = 4.0, grad=None, parents = None, function = None, num_children=1)
    a = Variable(value = 12.0, grad=None, parents = [x,y], function = Multiply, num_children=1)
    b = Variable(value = 9.0, grad=None, parents = [x], function = Power, num_children=1)
    c = Variable(value = 21.0, grad = None, parents = [a,b], function = Add, num_children=0)
    ```
    We would call `compute_gradients()` on `c`, which would first set `c.grad=1.0`, corresponding to the simple fact that

    $$\frac{\partial c}{\partial c} = 1.$$

    This would then call:
    ```
    grad_partials_products = c.function.backward(c.grad, a.value, b.value) # = [1, 1]
    ```
    and set `a.grad` and `b.grad` to each of these values, which represents the fact that

    $$\frac{\partial c}{\partial a} = 1, \frac{\partial c}{\partial b} = 1$$

    and decrease the `num_children` parameter of `a` and `b`. These last three nodes would now take on the values
    ```
    a = Variable(value = 12.0, grad=1.0, parents = [x,y], function = Multiply, num_children=0)
    b = Variable(value = 9.0, grad=1.0, parents = [x], function = Power, num_children=0)
    c = Variable(value = 21.0, grad = 1.0, parents = [a,b], function = Add, num_children=0)
    ```
    Since the parents of `c` both have `num_children=0`, the function would then call `.compute_gradients()` recursively on each of these nodes.  Let's consider calling `a.compute_gradients()` first.  This would compute
    ```
    grad_partials_products = a.function.backward(a.grad, x.value, y.value) # = [1*4.0, 1*3.0]
    ```
    which after similar updates would result in the values
    ```
    x = Variable(value = 3.0, grad=4.0, parents = None, function = None, num_children=1)
    y = Variable(value = 4.0, grad=3.0, parents = None, function = None, num_children=0)
    ```
    Finally, calling `b.compute_gradients()`, would compute
    ```
    grad_partials_products = b.function.backward(b.grad, x.value) # = [2*3.0]
    ```
    and result in the final `x` variable
    ```
    x = Variable(value = 3.0, grad=10.0, parents = None, function = None, num_children=0)
    y = Variable(value = 4.0, grad=3.0, parents = None, function = None, num_children=0)
    ```
    which contains all the correct gradients.


    ### Your implementation

    Implement the `compute_gradients` method of the following `Variable` class as described above.
    """)
    return


@app.function
def compute_gradients(self):
    """
    Recursively compute derivatives in a computation graph.  This method
    iteratively computes the gradients for a node and all it's parents.
    It has no input or output arguments, but instead directly modifies
    the `Variable` objects, populating the `.grad` variables as needed
    and calling the function recursively on its parents as needed.
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.cell
def _():
    Variable.compute_gradients = compute_gradients
    return


@app.function(hide_code=True)
def test_compute_gradients_local():
    test_compute_gradients(compute_gradients)


@app.cell(hide_code=True)
def _():
    submit_compute_gradients_button = mo.ui.run_button(
        label="submit `compute_gradients`"
    )
    submit_compute_gradients_button
    return (submit_compute_gradients_button,)


@app.cell
def _(submit_compute_gradients_button):
    mugrade.submit_tests(
        compute_gradients
    ) if submit_compute_gradients_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If your implementation is correct, you can now run commands like the following:
    """)
    return


@app.cell
def _():
    _x = Variable(3.0)
    _y = Variable(4.0)
    ((_x * _y + _x**2) / _y).compute_gradients()
    print(_x, _y)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Or more complex examples:
    """)
    return


@app.cell
def _():
    _x = Variable(3.0)
    _y = Variable(4.0)
    _z = Variable(1.2)
    ((_x * _y + _x**2 + _z).log() / (_z**3).exp()).compute_gradients()
    print(_x, _y, _z)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Part II - Training a digit classifier
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In this second part of this lecture, you'll train a linear classifier using PyTorch.  While we showed an example of this process in class, we computed the gradients manually there (without much intuition on how that gradient was derived), and here you'll rely on PyTorch's automatic differentiation to actually compute the gradients.  Note that your implementation here will just be based upon functions, not PyTorch `Module` classes that are the more standardized way of building models in PyTorch (in the next assignment, we will use these module classes)

    Let's begin by first loading the required data.
    """)
    return


@app.cell
def _():
    mnist_train = datasets.MNIST(".", train=True, download=True)
    mnist_test = datasets.MNIST(".", train=False, download=True)
    X, y_data = mnist_train.data.reshape(60000, 784) / 255, mnist_train.targets
    X_test, y_test = mnist_test.data.reshape(10000, 784) / 255, mnist_test.targets
    return X, X_test, y_data, y_test


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 3 - Cross entropy loss and error

    Implement the following functions, which compute the cross entropy loss and the error between a set of predictions.  Recall that the cross entropy loss is defined, for $\hat{y} \in \mathbb{R}^k$ and $y \in \{1,\ldots,k\}$ as

    $$L_{ce}(\hat{y}, y) = -\log \left ( \frac{\exp \hat{y}_y}{\sum_{j=1}^k \exp \hat{y}_j} \right ) = -\hat{y}_y + \log \sum_{j=1}^k \exp \hat{y}_j$$

    You can use the PyTorch function `torch.logsumexp` to compute the last term (this will be more numerically stable for large/small prediction values than individually calling `log` and `exp`).

    While you could use e.g. a for loop to compute cross entropy loss, this will be fairly inefficient later on.  Instead, you should use the fact that if you index a 2D tensor with two lists of indexes, it will select the elements corresponding to each of these indexes.  For instead given

    ```python
    A = torch.tensor([[1,2,3], [4,5,6]])
    i = torch.tensor([0,1,0,1])
    j = torch.tensor([0,1,2,1])
    ```
    Then
    ```python
    A[i,j] = tensor([1, 5, 3, 5]) # elements [A[0,0], A[1,1], A[0,2], A[1,1]]
    ```
    """)
    return


@app.function
def cross_entropy_loss(y_pred, y):
    """
    Compute the average cross entropy loss between predictions and desired
    outputs.

    Input:
        y_pred: 2D torch.Tensor[float] (N x k) - each row represents predicted
                                                 outputs for the ith example
        y : 1D torch.Tensor[int] (N) - each element represents desired output
                                       of ith example
    Output:
        scalar torch.Tensor[float] - average cross entropy loss of the predicted
                                     outputs
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


@app.function
def error(y_pred, y):
    """
    Compute the average error between predictions and desired outputs, assuming
    we make a "hard" prediction of whichever class has the highest predicted
    value.

    Input:
        y_pred: 2D torch.Tensor[float] (N x k) - each row represents predicted
                                                 outputs for the ith example
        y : 1D torch.Tensor[int] (N) - each element represents desired output
                                       of ith example
    Output:
        scalar torch.Tensor[float] - average error of the predicted outputs
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_error_local():
    test_error(error)


@app.cell(hide_code=True)
def _():
    submit_error_button = mo.ui.run_button(label="submit `error`")
    submit_error_button
    return (submit_error_button,)


@app.cell
def _(submit_error_button):
    mugrade.submit_tests(error) if submit_error_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question 4 - (Minibatch) Stochastic Gradient descent

    Finally, implement a minibatch version of the stochastic gradient descent method to optimize a linear classifier, using PyTorch.  Form a linear classifier specified by a matrix $W \in \mathbb{R}^{k \times n}$ (make sure to set `requires_grad=True` for this tensor, so you can compute gradients).

    Your function should iterate over the dataset `epochs` times, each time spliting the data into chunks of size `batch_size` (you can use the `torch.split()` function for this). For each of these chunks, compute the predictions of the linear classifier on this batch, compute the gradient of the cross entropy loss between these predictions and the desired outputs, and update $W$ by taking a step (scaled by `step_size`) in the direction of the negative gradient.  Note that after you've taken this step, you'll want to zero out the gradients of $W$ (otherwise, new gradients will be _added_ on top of the older existing gradients in the `.grad` variable, which is not what you want).
    """)
    return


@app.function
def train_sgd(X, y, epochs, step_size, batch_size):
    """
    Run minibatch stochastic gradient descent on the dataset X,y to minimize
    cross entropy loss.

    Inputs:
        X : 2D torch.Tensor[float] (N x n) - each row represents the ith input
                                             of the training set
        y : 1D torch.Tensor[int] (N) - each element represents desired output
                                       of ith example, in 0,...,k-1
        epochs : int - number of passes to make over the training set
        step_size : float - step size with which to update parameters
        batch_size : int - number of examples in a minibatch
    Output:
        2D torch.tensor[float] (k x n) - trained linear classifier
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_train_sgd_local():
    test_train_sgd(train_sgd)


@app.cell(hide_code=True)
def _():
    submit_train_sgd_button = mo.ui.run_button(label="submit `train_sgd`")
    submit_train_sgd_button
    return (submit_train_sgd_button,)


@app.cell
def _(submit_train_sgd_button):
    mugrade.submit_tests(train_sgd) if submit_train_sgd_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If you implemented this correctly, you should be able to train a classifier using code like the following.
    """)
    return


@app.cell(hide_code=True)
def _():
    step_size_slider = mo.ui.slider(
        start=0.01, stop=1.0, step=0.01, value=0.1, label="Step size", show_value=True, debounce=True
    )
    epochs_slider = mo.ui.slider(
        start=1, stop=50, step=1, value=5, label="Epochs", show_value=True, debounce=True
    )
    batch_size_slider = mo.ui.slider(
        start=10, stop=1000, step=10, value=100, label="Batch size", show_value=True, debounce=True
    )
    return batch_size_slider, epochs_slider, step_size_slider


@app.cell
def _(batch_size_slider, epochs_slider, step_size_slider):
    [
        step_size_slider,
        epochs_slider,
        batch_size_slider,
    ]
    return


@app.cell
def _(
    X,
    X_test,
    batch_size_slider,
    epochs_slider,
    step_size_slider,
    y_data,
    y_test,
):
    W = train_sgd(
        X, y_data,
        step_size=step_size_slider.value,
        epochs=epochs_slider.value,
        batch_size=batch_size_slider.value,
    )
    mo.md(f"**Test error: {error(X_test @ W.T, y_test):.4f}**")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Try to play around with different step sizes, epochs, and batch sizes until you can get a classifier with error under 8% (then best we've managed is slightly less than 7.5%).
    """)
    return


if __name__ == "__main__":
    app.run()
