# /// script
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

__generated_with = "0.19.7"
app = marimo.App()

with app.setup(hide_code=True):
    import marimo as mo

    import pytest
    import subprocess

    # Run this cell to download and install the necessary modules for the homework
    subprocess.call(
        [
            "wget",
            "-nc",
            "https://raw.githubusercontent.com/modernaicourse/hw1/refs/heads/main/hw1_tests.py",
        ]
    )

    import os
    import mugrade
    import torch
    from hw1_tests import (
        images,
        test_classify_zero_one,
        submit_classify_zero_one,
        test_vector_add,
        submit_vector_add,
        test_vector_inner_product,
        submit_vector_inner_product,
        test_matrix_vector_product_1,
        submit_matrix_vector_product_1,
        test_matrix_vector_product_2,
        submit_matrix_vector_product_2,
        test_vector_matrix_product_2,
        submit_vector_matrix_product_2,
        test_matmul_1,
        submit_matmul_1,
        test_matmul_2,
        submit_matmul_2,
        test_matmul_3,
        submit_matmul_3,
        test_batch_matmul,
        submit_batch_matmul,
        test_block_matmul,
        submit_block_matmul,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Homework 1 - Introduction to Linear Algebra + PyTorch

    This homework is aimed to familiarize you with some of the basic linear algebra operations we covered in class, as well as how to implement these functions and more in PyTorch.

    As before, make a copy of the assignment to your drive, add your mugrade key, and then run the cells below to get started.
    """)
    return


@app.cell
def _():
    os.environ["MUGRADE_HW"] = "Homework 1"
    os.environ["MUGRADE_KEY"] = ""  ### Your key here
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problem 1: ``Classical'' programming for digit classification

    This course deals primarily with machine learning approaches, but it's worth emphasizing that you _can_ try to approach many of the problems you'll want to solve with machine learning with traditional programming approaches as well.  In this problem, you should experiment with developing a "manual" classifier between images of digits in the MNIST dataset, which will be the first machine learning mode you'll develop during the later assignments.  Specifically, you'll want to implement the following function `classify_zero_one` to classify between images of zeros and ones in the MNIST dataset.  Try to think intuitively about features that might distinguish between zeros and ones, and if possible, try not to look at any statistics from the actual dataset (i.e., average values of the images, or anything like that).

    You can use the `images` dataset loaded above from the `hw1_tests.py` function (specifically the `images.data` and `images.targets` fields, which have been limited to just include the 0/1 images) to help you develop your code.
    """)
    return


@app.function
def classify_zero_one(image):
    """
    Classify a 28x28 pixel image as either a zero or one.

    Input:
        image : Tensor - 2D tensor storing grayscale pixel values of the image,
                         with each element a real-valued number in [0,1]
    Output:
        integer : 0 or 1
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The next cell runs _unit tests_ for the `classify_zero_one` function. Use these local tests to guide your implementation until all the tests pass.
    """)
    return


@app.function(hide_code=True)
def test_classify_zero_one_local():
    test_classify_zero_one(classify_zero_one)


@app.cell(hide_code=True)
def _():
    submit_classify_zero_one_button = mo.ui.run_button(
        label="submit `classify_zero_one`"
    )
    submit_classify_zero_one_button
    return (submit_classify_zero_one_button,)


@app.cell
def _(submit_classify_zero_one_button):
    mugrade.submit_tests(
        classify_zero_one
    ) if submit_classify_zero_one_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problem 2: Vector Addition

    In the remainder of this assignment, you're going to implement a wide variety of simple linear algebra operators, _without_ using any of the build-in tensor addition or matrix multiplication operators.  Your code should also throw assertion errors if any of the sizes do not match was it allowed for the given operation (i.e., you should be calling assert() to check that the sizes are correct).  Instead, you should use explicit for loops and element-by-element assignment/operations to implement your function.  You can also create new vectors of the right size as your return variable, etc.

    First implement a simple vector addition function that adds two vectors together, $x,y \in \mathbb{R}^n$.  Note that it is ok if this only works when provided with vectors, i.e., 1D tensors.
    """)
    return


@app.function
def vector_add(x, y):
    """
    Add two vectors x and y, _without_ using the built-in addition of torch.
    Instead, you need to manually iterate through the elements of x and y and
    add them together.  The function should throw an AssertionError, via
    calling assert(), if the vectors are not the proper size to add together.

    Input:
        x : 1D torch.Tensor - first term to add
        y : 1D torch.Tensor - second term to add

    Output:
        return 1D torch.Tensor - sum of x + y

    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_vector_add_local():
    test_vector_add(vector_add)


@app.cell(hide_code=True)
def _():
    submit_vector_add_button = mo.ui.run_button(label="submit `vector_add`")
    submit_vector_add_button
    return (submit_vector_add_button,)


@app.cell
def _(submit_vector_add_button):
    mugrade.submit_tests(vector_add) if submit_vector_add_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problem 3: Vector inner product

    Now implement the vector inner product.  I.e., for two vectors $x, y \in \mathbb{R}^n$, return the inner product
    $$\langle x,y \rangle \equiv x^T y = \sum_{i=1}^n x_i y_i.$$

    As before, don't use any PyTorch functions that compute a matrix multiplication or inner product directly, but do it all with for loops.
    """)
    return


@app.function
def vector_inner_product(x, y):
    """
    Compute the inner product between two vectors x and y, _without_ using the
    matrix multiplication operator '@' (or any similar PyTorch function). The
    function should throw an AssertionError if the vectors are not the proper
    size.

    Input:
        x : 1D torch.Tensor - first term to add
        y : 1D torch.Tensor - second term to add

    Output:
        return float - inner product <x,y>
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_vector_inner_product_local():
    test_vector_inner_product(vector_inner_product)


@app.cell(hide_code=True)
def _():
    submit_vector_inner_product_button = mo.ui.run_button(
        label="submit `vector_inner_product`"
    )
    submit_vector_inner_product_button
    return (submit_vector_inner_product_button,)


@app.cell
def _(submit_vector_inner_product_button):
    mugrade.submit_tests(
        vector_inner_product
    ) if submit_vector_inner_product_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problem 4: Matrix-vector product approach #1

    Write a routine that function that computes the matrix-vector product $Ax$ for $A \in \mathbb{R}^{m \times n}$ and $x \in \mathbb{R}^n$.  This version should compute each entry of the resuting vector using the inner product between rows of $A$ and the vector $x$, i.e., shown graphically this would be

    $$
    Ax = \left [ \begin{array}{ccc}
    \;\text{—} & a^T_1 & \text{—}\; \\
    \;\text{—} & a^T_2 & \text{—}\; \\
    & \vdots & \\
    \;\text{—} & a^T_m & \text{—}\;
    \end{array} \right ] \left [ \begin{array}{c}\mid \\ x \\ \mid \end{array}  \right ] = \left [ \begin{array}{c} a^T_1 x \\ a^T_2 x \\ \vdots \\ a^T_m x \end{array} \right].
    $$

    Only make use of the above-implemented `vector_inner_product()` function you implemetned above for this routine, i.e., no other operations on the tensors.
    """)
    return


@app.function
def matrix_vector_product_1(A, x):
    """
    Compute the matrix vector product Ax _without_ using the matrix
    multiplication operator @ or any related function.  In this variant
    implement the output as the inner product of each row of A with
    the vector x (i.e., only make use of the vector_inner_product function).
    Be sure to throw AssertionErrors if the product is not valid.

    Input:
        A : 2D torch.Tensor - m x n matrix A
        x : 1D torch.Tensor - vector x with n elements

    Output:
        return 1D torch.Tensor - vector Ax with m elements
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_matrix_vector_product_1_local():
    test_matrix_vector_product_1(matrix_vector_product_1)


@app.cell(hide_code=True)
def _():
    submit_matrix_vector_product_1_button = mo.ui.run_button(
        label="submit `matrix_vector_product_1`"
    )
    submit_matrix_vector_product_1_button
    return (submit_matrix_vector_product_1_button,)


@app.cell
def _(submit_matrix_vector_product_1_button):
    mugrade.submit_tests(
        matrix_vector_product_1
    ) if submit_matrix_vector_product_1_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problem 5: Matrix-vector product approach #2

    Write a routine that function that computes the matrix-vector product $Ax$ for $A \in \mathbb{R}^{m \times n}$ and $x \in \mathbb{R}^n$.  This version should compute the result as a linear combination of the columns of $A$ with coefficients given by the entries of $x_i$, i.e., shows graphically this would be

    $$
    Ax = \left [ \begin{array}{cccc} \mid & \mid & & \mid \\
    a_1 & a_2 & \cdots & a_n \\
    \mid & \mid & & \mid \end{array} \right ]
    \left [ \begin{array}{c} x_1 \\ x_2 \\ \vdots \\ x_n \end{array}\right ] =
    \left [ \begin{array}{c} \mid \\ a_1 \\ \mid \end{array} \right ] x_1 +
    \left [ \begin{array}{c} \mid \\ a_2 \\ \mid \end{array} \right ] x_2 + \ldots +
    \left [ \begin{array}{c} \mid \\ a_n \\ \mid \end{array} \right ] x_n
    $$

    Only make use of the above-implemented `vector_add()` function to implement your solution (plus of course creating vectors to return, etc).  It is also ok to multiply a vector by a scalar, i.e., the code `c*y` where `c` is a vector and `y` is a real-valued scalar.
    """)
    return


@app.function
def matrix_vector_product_2(A, x):
    """
    Compute the matrix vector product Ax _without_ using the matrix
    multiplication operator @ or any related function.  In this variant
    implement the output as a linear combination of the columns of A with
    coefficients given by the entries of x (and only make use of the
    vector_add function).  Be sure to throw AssertionErrors if the sizes do
    not allow for a valid product

    Input:
        A : 2D torch.Tensor - m x n matrix A
        x : 1D torch.Tensor - vector x with n elements

    Output:
        return 1D torch.Tensor - vector Ax with m elements
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_matrix_vector_product_2_local():
    test_matrix_vector_product_2(matrix_vector_product_2)


@app.cell(hide_code=True)
def _():
    submit_matrix_vector_product_2_button = mo.ui.run_button(
        label="submit `matrix_vector_product_2`"
    )
    submit_matrix_vector_product_2_button
    return (submit_matrix_vector_product_2_button,)


@app.cell
def _(submit_matrix_vector_product_2_button):
    mugrade.submit_tests(
        matrix_vector_product_2
    ) if submit_matrix_vector_product_2_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problem 6: Vector-matrix product approach #2

    Write a routine that function that computes the vector-Matrix product $x^TA$ for $A \in \mathbb{R}^{m \times n}$ and $x \in \mathbb{R}^m$.  In keeping with PyTorch convention (i.e., not differentiating between column and row vectors), this should return a 1D tensor representing the resulting row vector.

    This version should compute the result as a linear combination of the rows of $A$ with coefficients given by the entries of $x_i$, i.e., shows graphically this would be

    $$
    \begin{split}
    x^T A & =
    \left [ \begin{array}{cccc} x_1 & x_2 & \ldots & x_m \end{array} \right ]
    \left [ \begin{array}{ccc}
    \;\text{—} & a^T_1 & \text{—}\; \\
    \;\text{—} & a^T_2 & \text{—}\; \\
    & \vdots & \\
    \;\text{—} & a^T_m & \text{—}\;
    \end{array} \right ] \\ & =
    x_1 \left [ \begin{array}{ccc} \;\text{—} & a^T_1 & \text{—}\; \end{array} \right ] +
    x_2 \left [ \begin{array}{ccc} \;\text{—} & a^T_2 & \text{—}\; \end{array} \right ] + \ldots +
    x_m \left [ \begin{array}{ccc} \;\text{—} & a^T_m & \text{—}\; \end{array} \right ]
    \end{split}
    $$

    Only make use of the above-implemented `vector_add()` function to implement your solution, with the same caveats as in the previous problem.
    """)
    return


@app.function
def vector_matrix_product_2(x, A):
    """
    Compute the vector Matrix product x^T A _without_ using the matrix
    multiplication operator @ or any related function.  In this variant
    implement the output as a linear combination of the rows of A with
    coefficients given by the entries of x (and only make use of the
    vector_add function).  Note that, in keeping with PyTorch convention (of
    not differentiating between row and column vectors), x will just be an
    vector (1D tensor) with m elements, and the output should be a vector (1D
    Tensor) with n elements. Be sure to throw AssertionErrors if the sizes do
    not allow for a valid product.

    Input:
        A : 2D torch.Tensor - m x n matrix A
        x : 1D torch.Tensor - vector x with m elements

    Output:
        return 1D torch.Tensor - vector x^T A with n elements
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_vector_matrix_product_2_local():
    test_vector_matrix_product_2(vector_matrix_product_2)


@app.cell(hide_code=True)
def _():
    submit_vector_matrix_product_2_button = mo.ui.run_button(
        label="submit `vector_matrix_product_2`"
    )
    submit_vector_matrix_product_2_button
    return (submit_vector_matrix_product_2_button,)


@app.cell
def _(submit_vector_matrix_product_2_button):
    mugrade.submit_tests(
        vector_matrix_product_2
    ) if submit_vector_matrix_product_2_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problem 7: Matrix-matrix multiplication approach #1

    Write a matrix-matrix multiplication function, again without using any built-in operators.  For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, this version should compute each element $(AB)_{ij}$ as the inner product of the $i$th row of $A$ and the $j$th column of $B$.  Depicted graphically, this would be the breakdown

    $$
    AB =
    \left [ \begin{array}{ccc}
    \;\text{—} & a^T_1 & \text{—}\; \\
    \;\text{—} & a^T_2 & \text{—}\; \\
    & \vdots & \\
    \;\text{—} & a^T_m & \text{—}\;
    \end{array} \right ]
    \left [ \begin{array}{cccc} \mid & \mid & & \mid \\
    b_1 & b_2 & \cdots & b_p \\
    \mid & \mid & & \mid \end{array} \right ]
    =
    \left [ \begin{array}{cccc} a_1^T b_1 & a_1^T b_2 & \cdots & a_1^T b_p \\
    a_2^T b_1 & a_1^T b_2 & \cdots & a_2^T b_p \\
    \vdots & \vdots & \ddots & \vdots \\
    a_m^T b_1 & a_m^T b_2 & \cdots & a_m^T b_p \end{array} \right ]
    $$

    With all the same caveats as before, this implementation should only use the function `vector_inner_product()` that you implemented above.
    """)
    return


@app.function
def matmul_1(A, B):
    """
    Compute the matrix matrix multiplication AB without using the @ operator.
    In this variant, compute each entry of the matrix product as the inner
    product of a row of A and a column of B (i.e., using the
    vector_inner_product function).  Be sure to throw AssertionErrors if the
    sizes of the matrices do not make for a valid product.


    Input:
        A : 2D torch.Tensor - m x n matrix A
        B : 2D torch.Tensor - n x p matrix B

    Output:
        return 2D torch.Tensor - m x p matrix equal to the product AB
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_matmul_1_local():
    test_matmul_1(matmul_1)


@app.cell(hide_code=True)
def _():
    submit_matmul_1_button = mo.ui.run_button(label="submit `matmul_1`")
    submit_matmul_1_button
    return (submit_matmul_1_button,)


@app.cell
def _(submit_matmul_1_button):
    mugrade.submit_tests(matmul_1) if submit_matmul_1_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problem 8: Matrix-matrix multiplication approach #2

    Write another matrix multiplication implemention. For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, this version should compute the $i$th column of $AB$ as the matrix-vector product between $A$ and $i$th column of $B$. Depicted graphically, this would be the breakdown

    $$
    AB =
    A
    \left [ \begin{array}{cccc} \mid & \mid & & \mid \\
    b_1 & b_2 & \cdots & b_p \\
    \mid & \mid & & \mid \end{array} \right ]
    =
    \left [ \begin{array}{cccc} \mid & \mid & & \mid \\
    A b_1 & A b_2 & \cdots & A b_p \\
    \mid & \mid & & \mid \end{array} \right ]
    $$

    With all the same caveats as before, this implementation should only use the function `matrix_vector_product_1()` (or `matrix_vector_product_2()`) that you implemented above.
    """)
    return


@app.function
def matmul_2(A, B):
    """
    Compute the matrix matrix multiplication AB without using the @ operator.
    In this variant, compute the ith _column_ of the matrix product as the
    matrix-vector product of A and the ith column of B (i.e., using only the
    function matrix_vector_product_1 or matrix_vector_product_2). Be sure to
    throw AssertionErrors if the sizes of the matrices do not make for a valid
    product.

    Input:
        A : 2D torch.Tensor - m x n matrix A
        B : 2D torch.Tensor - n x p matrix B

    Output:
        return 2D torch.Tensor - m x p matrix equal to the product AB
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_matmul_2_local():
    test_matmul_2(matmul_2)


@app.cell(hide_code=True)
def _():
    submit_matmul_2_button = mo.ui.run_button(label="submit `matmul_2`")
    submit_matmul_2_button
    return (submit_matmul_2_button,)


@app.cell
def _(submit_matmul_2_button):
    mugrade.submit_tests(matmul_2) if submit_matmul_2_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problem 9: Matrix-matrix multiplication approach #3

    Finally, write one last matrix multiplication implementation. For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, this version should compute the $i$th row of $AB$ as the vector-matrix product between the $i$th row of $A$ and $B$. This would be the breakdown

    $$
    AB =
    \left [ \begin{array}{ccc}
    \;\text{—} & a^T_1 & \text{—}\; \\
    \;\text{—} & a^T_2 & \text{—}\; \\
    & \vdots & \\
    \;\text{—} & a^T_m & \text{—}\;
    \end{array} \right ] B =
    \left [ \begin{array}{ccc}
    \;\text{—} & a^T_1 B & \text{—}\; \\
    \;\text{—} & a^T_2 B & \text{—}\; \\
    & \vdots & \\
    \;\text{—} & a^T_m B & \text{—}\;
    \end{array} \right ]
    $$


    With all the same caveats as before, this implementation should only use the function `vector_matrix_product_2()` that you implemented above.
    """)
    return


@app.function
def matmul_3(A, B):
    """
    Compute the matrix matrix multiplication AB without using the @ operator.
    In this variant, compute the ith _row_ of the matrix product as the
    vector-matrix product of the ith row A and B (i.e., using only the
    function vector_matrix_product_2). Be sure to throw AssertionErrors if the
    sizes of the matrices do not make for a valid product.

    Input:
        A : 2D torch.Tensor - m x n matrix A
        B : 2D torch.Tensor - n x p matrix B

    Output:
        return 2D torch.Tensor - m x p matrix equal to the product AB
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_matmul_3_local():
    test_matmul_3(matmul_3)


@app.cell(hide_code=True)
def _():
    submit_matmul_3_button = mo.ui.run_button(label="submit `matmul_3`")
    submit_matmul_3_button
    return (submit_matmul_3_button,)


@app.cell
def _(submit_matmul_3_button):
    mugrade.submit_tests(matmul_3) if submit_matmul_3_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problem 10: Batch matrix multiplication

    In this problem, you will implement batch matrix multiplication.  Consider two ND PyTorch tensors of the dimensions $A \in \mathbb{R}^{n_1 \times n_2 \times \ldots \times n_d}$ $B \in \mathbb{R}^{m_1 \times m_2 \times \ldots \times m_d}$ with the same sizes on all but the last two dimensions

    $$ n_i = m_i, \; i=1,\ldots,d-2$$

    and the last two dimensions properly sized for a matrix multiplication

    $$n_i = m_{i-1}.$$

    In this case implement a batched version of matrix multiplication that iterates over all the leading $d-2$ dimensions and performs a matrix multiplication of the corresponding entries.  The function should throw an AssertionError if any of the sizes do not match.

    You should still not use the PyTorch matrix multiplication operator, but instead call one of the `matmul()` functions you implemented above (it doesn't really matter which one).
    """)
    return


@app.function
def batch_matmul(A, B):
    """
    Implement batch matrix multiplication between 2 tensors A and B by
    iterating over all the leading dimensions of A and B (all dimensions other
    than the last two), and performing a matrix multiplication over the last
    two dimensions. A and B must be sized so that their leading dimensions are
    all the same, and the last two dimensions are sized for a valid matrix
    multiplication.

    Inputs:
        A : torch.Tensor - ND tensor with trailing dimensions (..., m, n)
        B : torch.Tensor - ND tensor with trailing dimensions (..., n, p)

    Output:
        return torch.Tensor - ND tensor with tailing dimensions (..., m, p)
                              containing all matrix multiplications of the
                              corresponding entries.
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_batch_matmul_local():
    test_batch_matmul(batch_matmul)


@app.cell(hide_code=True)
def _():
    submit_batch_matmul_button = mo.ui.run_button(label="submit `batch_matmul`")
    submit_batch_matmul_button
    return (submit_batch_matmul_button,)


@app.cell
def _(submit_batch_matmul_button):
    mugrade.submit_tests(
        batch_matmul
    ) if submit_batch_matmul_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problem 11: Block matrix multiplication

    In this last question, you'll implement a "blocked" form of matrix multiplication.  Although we defined matrix multiplication in terms of the individual scalar entries of a matrix, it can also be defined by operating on subblocks of the matrices.  Specifically for an matrix $A \in \mathbb{R}^{4m \times 4n}$ we can define $A_{ij} \in \mathbb{R}^{4 \times 4}$ to be a _subblock_ of the matrix, and similarly for the matrix $B \in \mathbb{R}^{4n \times 4p}.  Then the corresponding $4 \times 4$ subblock of the matrix product $AB$ can be computed as
    $$ (AB)_ij = \sum_{k=1}^n A_{ik} B_{kj} $$
    analogous to the usual definition of matrix multiplication, but with $A_{ik} B_{kj}$ now being a matrix product.

    In practice, techniques like this (with proper memory layouts, which we don't cover here) are how write fast matrix multiplication primitives on GPUs (where e.g., so-called "tensor cores" actually exactly perform 4x4 matrix multiplication).

    Implement the `block_matmul` function below.  You should _only_ call the `add_matmul_44()` function in your implementation.  You should check to ensure that the matrices form a valid matrix multiplication, and that they are all divisible by 4.
    """)
    return


@app.function
def add_matmul_44(Z, A, B):
    """
    Simulate a "fast" 4x4 matrix multiplication and in-place addition to Z:
        Z += AB
    """
    assert Z.shape == (4, 4) and A.shape == (4, 4) and B.shape == (4, 4)
    for i in range(4):
        for j in range(4):
            Z[i, j] += (
                A[i, 0] * B[0, j]
                + A[i, 1] * B[1, j]
                + A[i, 2] * B[2, j]
                + A[i, 3] * B[3, j]
            )


@app.function
def block_matmul(A, B):
    """
    Implement a block matrix multiplication to compute the matrix-matrix
    product AB.  You should use the formula above, and also assert that that
    matrices are the proper shapes (and have dimensions that are multiples of
    4).  Use only the matmul_44 call.
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_block_matmul_local():
    test_block_matmul(block_matmul)


@app.cell(hide_code=True)
def _():
    submit_block_matmul_button = mo.ui.run_button(label="submit `block_matmul`")
    submit_block_matmul_button
    return (submit_block_matmul_button,)


@app.cell
def _(submit_block_matmul_button):
    mugrade.submit_tests(
        block_matmul
    ) if submit_block_matmul_button.value else None
    return


if __name__ == "__main__":
    app.run()
