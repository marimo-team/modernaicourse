# /// script
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "pytest==9.0.2",
#     "requests==2.32.5",
#     "mugrade @ git+https://github.com/locuslab/mugrade.git",
# ]
# ///

import marimo

__generated_with = "0.19.2"
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
            "https://raw.githubusercontent.com/modernaicourse/hw0/refs/heads/main/hw0_tests.py",
        ]
    )

    import os
    import mugrade
    import math
    from hw0_tests import (
        submit_add,
        submit_primes,
        submit_poly_add,
        submit_poly_mul,
        submit_poly_derivative,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Homework 0

    This homework is different from the other assignments in this course, in that we don't actually build anything related to AI or chatbots.  Instead, this is a basic assignment meant to test two things:
    1. Getting set up with our mugrade autograding systems, and
    2. Evaluating some basic background in programming and math proficiency, of the kind you'll be expected to need for this course.

    If you can complete this assignment relatively easily, it is a good sign that your background will be sufficient for the course.  If the assignment is particularly challenging, then this might be a good indication that you would benefit from additional background before taking the course.  It's definitely not impossible to take the course while simultaneously building up this background, but it's likely going to be quite a bit harder.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Basic notebook setup

    This assignment uses the [marimo notebook](https://docs.marimo.io), an alternative to the Jupyter notebook
    that was developed in 2022 in collaboration with scientists at Stanford. Like
    Jupyter, marimo is open-source, but unlike Jupyter is is a *reactive* notebook,
    keeping code and outputs in sync.

    Using marimo is optional. You can run marimo in molab, an online service
    similar to Google Colab, or locally. You will implement homework functions
    in this notebook, then test your implementations, first locally and then by
    submitting to the autograder.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problem 1: Mugrade setup (via simple addition)

    The first problem in this homework mainly serves to introduce you to the [mugrade](https://mugrade.org) grading system.  To access the mugrade system:
    1. Visit the mugrade enrollment link shared in Ed (or shared in the email we will send to the online course participants).
    2. Sign in using the Google account associated with your _CMU Andrew email_ (for the CMU in-person course, for the online course you can use any email).

    After you have set up your mugrade account and enrolled in the class, you should see a page for the course and the available assignments.  There is also a "Show API Key" button at the top of the page.  To submit assignments, copy this key and enter it into the cell of the notebook.

    [Note: Please don't share your API key with others.  If any of you are (properly) horrified at the idea of storing an API key in the plain text of a notebook, you're welcome to add a `MUGRADE_KEY` environmental variable using any technique that will properly set the environment of the notebook.  For simplicity, and because you can't do much other than submit assignments with these keys, we're taking the simple (insecure) approach are just storing it in the notebook.]
    """)
    return


@app.cell
def _():
    os.environ["MUGRADE_HW"] = "Homework 0"
    os.environ["MUGRADE_KEY"] = ""
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Once this account is set up, let's see how to submit an assignment.  Consider implementing the following function, which adds together its two arguments.  In general, the functions you implement in homework will be provided like this, giving you a signature of the function (or Python class).  Your job will be to implement the block within the comments:
    ```python
    ### BEGIN YOUR CODE

    ### END YOUR CODE
    ```
    This is not too challenging an implementation, and in fact we'll tell you that the correct solution to this assignment is simply to replace the `pass` code with `return x+y`.  You can fill this out in the implementation below.
    """)
    return


@app.function
def add(x, y):
    """
    Add x and y

    Input:
        x: integer or float
        y: integer or float

    Output:
        integer or float, addition of x and y
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The next cell runs _unit tests_ for the `add` function, checking the correctness
    of the function you just implemented. You can take a look at these unit tests by
    expanding the code of the next cell.

    By default, changing the code of the `add` function will automatically run these local tests.
    Because this function starts without an implementation, the tests all fail initially.
    Use these local tests to guide your implementation until all the tests pass.
    Note that passing the local test cases definitely doesn't guarantee correctness
    of the function, but it should hopefully provide a sanity check.
    """)
    return


@app.cell(hide_code=True)
def _():
    @pytest.mark.parametrize(
        ("x", "y", "z"),
        [
            (5, 6, 11),
            (2.1, 2.3, 4.4),
        ],
    )
    def test_add(x, y, z):
        assert add(x, y) == z

    def test_add_type():
        assert isinstance(add(4, 2.1), float)

    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    _After_ you pass all the local tests succesfully, you should submit your solution to mugrade. You can do this by simply clicking the `submit` button below.
    """)
    return


@app.cell(hide_code=True)
def _():
    submit_add_button = mo.ui.run_button(label="submit `add`")
    submit_add_button
    return (submit_add_button,)


@app.cell
def _(submit_add_button):
    mugrade.submit_tests(add) if submit_add_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If all goes correctly, you should see feedback that the grader tests
    passed.  And when you view the mugrade site (you may need to refresh), you
    should see the assignment as being submitted correctly.  If anything goes
    wrong, you should see an error code that indicates if there is any obvious
    problem (for instance, if you didn't set the `MUGRADE_KEY` value above).
    However, in general when you fail a submission we _don't_ provide very much
    feedback, because the goal is for you to use the local tests to debug your
    assignment, and the submission just to submit the final version.

    **For the CMU course only (not needed for online course):** After you have
    submitted _all_ the assignments for the course, you should additionally
    upload the notebook itself: download the notebook from molab, and upload it
    to mugrade using the "Upload Code" link at the bottom of the page
    assignment page.""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <details>
    <summary>How mugrade works</summary>

    <p>Note: for those who want a bit more information about how mugrade works, and how it differs from other autograding systems, we'll provide a bit of info here, but understanding this is not required in anyway way for the course.  Most autograding systems work by uploading your code and running it on the grading server; this ensures that your code runs independently and functions as desired, but has the notable downside that, for complex assignments, it can be difficult to ensure that the environment of the autograder is sufficiently similar to your own environment.  Countless hours have been wasted by students trying to trace the exact way in which the autograding environment differs from their own.  This is especially a problem for machine learning assignments where GPUs are required to run the code, and we know of very few systems that handle this setup in a seamless fashion.</p>

    <p>
    This has led us to develop mugrade, which takes the different philosophy, runs
    all the autograding code _on your local system_ (or on molab), and then just
    sends the _answer_ to mugrade to check against a reference solution.  The upside
    is that we can support large courses with relatively little overhead.  The
    downside is that it does allow cheating of some type: you absolutely could
    compute the solutions manually for each mugrade test case and then submit those
    answers to get full points.  And this would almost certainly be more
    time-consuming than just solving the problem to begin with.  So in a world where
    most assignments can be solved by AI anyway, it seems a very reasonable
    trade-off to give a bit more control to students of their coding environment, at
    the cost of possibly introducing slightly more ability to game the system.  This
    is why, as a final measure, we ask for all students to upload a copy of their
    assignment, so TA's can check it as needed.
    </p>

    </details>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problem 2: Computing primes

    As the first "real" problem of this assignment, you'll have to write a simple routine to compute prime numbers.  You will specifically do this using a method called the [Sieve of Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes), which computes _all_ prime numbers up to some number $n$ by iteratively setting elements of an array to `False` if they cannot be a prime because they are a multiple of another number.  The [pseudocode](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes#Pseudocode) of the algorithm on the Wikipedia link above provides a pretty reasonable description of the algorithm.  As mentioned above, this particular function has nothing to do with machine learning or the rest of the course, but it is a nice demonstration of an algorithm to implement in Python, that is a good test of your general basic proficiency.

    Using this pseudocode as a reference point (but note that you _don't_ have to follow it exactly), plus the function signature below (which you _do_ need to follow exactly), implement a method for computing primes using this method.  Note that even though the algorithm computes a binary array of which numbers are prime, the function as defined below needs to return a list of just the prime numbers, so you'll need to convert to that format.

    For this function, you'll want to first run the `@mugrade.local_tests` test cases, then change this to `@mugrade.submit` once you have finished and debugged your implementation.
    """)
    return


@app.function
def primes(n):
    """
    Compute all the primes up to (but not including) n via sieve of Eratosthenes.

    Input:
        n: integer
    Output:
        list of primes up to (not including) n
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_primes():
    p = primes(10)
    assert isinstance(p, list)
    assert p == [2, 3, 5, 7]


@app.cell(hide_code=True)
def _():
    submit_primes_button = mo.ui.run_button(label="submit `primes`")
    submit_primes_button
    return (submit_primes_button,)


@app.cell
def _(submit_primes_button):
    mugrade.submit_tests(primes) if submit_primes_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Problem 3: Operations on polynomials

    As a last problem in this assignment, you will write code to manipulate polynomials represented by a Python class.  Again, this assignment is not directly relevant to much AI work, but it _does_ serve as a useful evaluation of your familiarity with some basic mathematical concepts, how you translate these into Python work, and in working with existing Python classes.

    The `Polynomial` class implemented below contains a basic representation of polynomials via a list of their coefficients.  Each class contains a `.coefficients` member, which is a list where `coefficients[i]` represents the coefficient on the $i$th degree term, $x^i$.  In other words, the list
    ```python
    coefficients = [1, 0, 4, 3]
    ```
    would represent the polynomial

    $$3x^3 + 4x^2 + 1,$$

    the list
    ```python
    coefficients = [4, 3, 5]
    ```
    would represent the polynomial

    $$5x^2 + 3x + 4,$$

    and so on.  Any term of degree beyond the length of the list implicitly has coefficient zero.

    The class below contains the basic implementation (which is quite sparse, other than a few helper functions and a function that returns a string representation of the polynomial).  Note that you should _not_ change any of this code in the assignment.
    """)
    return


@app.class_definition
class Polynomial:
    """
    This class represents a polynomial as a list of coefficients.  Each item in list
    at position i (zero-indexed), represents the coefficient corresponding to the x^i
    term of the polynomial.  For instance, the list:

    [1, 0, 4, 3]
    would represent the polynomial
    3x^3 + 4x^2 + 1
    """

    def __init__(self, coefficients):
        """Initialize the coefficients, and make the largest degree coefficient is not zero"""
        self.coefficients = coefficients
        while self.coefficients[-1] == 0 and len(self.coefficients) > 1:
            self.coefficients.pop()

    def __eq__(self, value):
        """Check if two polynomials are equal"""
        return self.coefficients == value.coefficients

    def degree(self):
        return len(self.coefficients) - 1

    def __repr__(self):
        """Returns a string representation of the polynomial"""
        if len(self.coefficients) == 0:
            return "0"
        terms = []
        for i, c in enumerate(self.coefficients):
            if c != 0:
                if i == 0:
                    terms.append(f"{c}")
                elif i == 1:
                    terms.append(f"{c}x")
                else:
                    terms.append(f"{c}x^{i}")
        if len(terms) == 0:
            terms.append("0")
        return " + ".join(reversed(terms))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To get a sense of how this code works, you can test out forming some basic polynomials as is done in the cell below:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Problem 3a: Polynomial additiona

    Write the code to add together two polynomials.  Adding polynomials just involves adding the respective coefficients of the same degree.  For example, if you had two polynomials

    $$p_1(x) = 3x^3 + 4x^2 + 3, \quad p_2(x) = x^2 + 5x + 5$$

    then

    $$p_1(x) + p_2(x) = 3x^3 + 5x^2 + 5x + 8.$$

    Implement this logic in the function below.  Note that the "only" thing that's required to implement this is to construct a new set off polynomial coefficients corresponding to the addition, and return a new polynomial constructed from these coefficients.
    """)
    return


@app.function
def poly_add(p1, p2):
    """
    Add two polynomials together.  For instance, if the polynomials represent
    p1 = 3x^3 + 4x^2 + 3
    p2 = x^2 + 5x + 5
    Then:
    p1 + p2 =  3x^3 + 5x^2 + 5x + 8


    Input:
        p1 : Polynomial
        p2 : Polynomial

    Output:
        Polynomial corresponding to the addition of p1 and p2
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_poly_add():
    p1 = Polynomial([1, 5, 0, 5])
    p2 = Polynomial([0, 2])
    p3 = Polynomial([-1, 6, 7, -5])
    p4 = Polynomial([0.3, 0.4, 1.6, 1.9])
    assert poly_add(p1, p2) == Polynomial([1, 7, 0, 5])
    assert poly_add(p1, p3) == Polynomial([0, 11, 7])
    assert poly_add(p1, Polynomial([0])) == p1
    assert poly_add(p2, p4) == Polynomial([0.3, 2.4, 1.6, 1.9])


@app.cell(hide_code=True)
def _():
    submit_poly_add_button = mo.ui.run_button(label="submit `poly_add`")
    submit_poly_add_button
    return (submit_poly_add_button,)


@app.cell
def _(submit_poly_add_button):
    mugrade.submit_tests(poly_add) if submit_poly_add_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Problem 3b: Polynomial multiplication

    Next implement a function that multiplies together two polynomials.  Multiplying polynomials involves multipling every term in the first polynomial with every term in the second, and adding together the results.  For example, for the polynomials

    $$p_1(x) = 3x^3 + 2x + 3, \quad p_2(x) = 2x^2 + 5$$

    their multiplication is given by

    $$\begin{split} p_1(x) \cdot p_2(x) & = (3x^3 + 2x + 3) \cdot 2x^2 + (3x^3 + 2x + 3) \cdot 5 \\
    & = (6x^5 + 4x^3 + 6x^2) + (15x^3 + 10x + 15) \\ & = 6x^5 + 19x^3 + 6x^2 + 10x + 15\end{split}.$$

    Implement this logic in the function below.
    """)
    return


@app.function
def poly_mul(p1, p2):
    """
    Multiply two polynomials together and return the result as a new Polynomial. For example,
    if the two polynomials represent
    p1 = 3x^3 + 2x + 3
    p2 = 2x^2 + 5

    Then:
    p1 + p2 = 6x^5 + 19x^3 + 6x^2 + 10x + 15

    Input:
        p1 : Polynomial
        p2 : Polynomial

    Output:
        Polynomial corresponding to the multiplication of p1 and p2
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_poly_mul():
    p1 = Polynomial([1, 5, 0, 5])
    p2 = Polynomial([0, 2])
    p3 = Polynomial([-1, 6, 7, -5])
    p4 = Polynomial([0.3, 0.4, 1.6, 1.9])
    assert poly_mul(p1, p2) == Polynomial([0, 2, 10, 0, 10])
    assert poly_mul(p1, p3) == Polynomial([-1, 1, 37, 25, 5, 35, -25])
    assert poly_mul(p1, Polynomial([1])) == p1
    assert poly_mul(p1, p4) == Polynomial([0.3, 1.9, 3.6, 11.4, 11.5, 8.0, 9.5])


@app.cell(hide_code=True)
def _():
    submit_poly_mul_button = mo.ui.run_button(label="submit `poly_mul`")
    submit_poly_mul_button
    return (submit_poly_mul_button,)


@app.cell
def _(submit_poly_mul_button):
    mugrade.submit_tests(poly_mul) if submit_poly_mul_button.value else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Problem 3c: Polynomial differentiation

    For the final question, you should implement a function the computes the derivative of the polynomial with respect to $x$.  You should be familiar with the basic rules of derivatives in order to compute this, but as an example, if we have the polynomial

    $$p(x) = 4x^3 + 3x + x$$

    then the derivative of $p(x)$ with respect to $x$, which we denote as $p'(x)$, is given by

    $$p'(x) = 12x^2 + 3$$

    where in general the derivative of any term $c x^n$ is given by $ncx^{n-1}$ and the of any constant term $c$ is zero.

    Implement this logic in the function below.
    """)
    return


@app.function
def poly_derivative(p):
    """
    Compute the derivative of the polynomial with respect to x.  For instance, if
    p = 4x^3 + 3x + 3

    Then the derivative would be given by
    12x^2 + 3

    Input:
        p : Polynomial

    Output:
        Polynomial corresponding to the derivative of p with respect to x
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE


@app.function(hide_code=True)
def test_poly_derivative():
    p1 = Polynomial([1, 5, 0, 5])
    p2 = Polynomial([0.3, 0.4, 1.6])
    assert poly_derivative(p1) == Polynomial([5, 0, 15])
    assert poly_derivative(p2) == Polynomial([0.4, 3.2])
    assert poly_derivative(Polynomial([0])) == Polynomial([0])


@app.cell(hide_code=True)
def _():
    submit_poly_derivative_button = mo.ui.run_button(label="submit `poly_derivative`")
    submit_poly_derivative_button
    return (submit_poly_derivative_button,)


@app.cell
def _(submit_poly_derivative_button):
    mugrade.submit_tests(
        poly_derivative
    ) if submit_poly_derivative_button.value else None
    return


if __name__ == "__main__":
    app.run()
