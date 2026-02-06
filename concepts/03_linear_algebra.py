# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.1",
#     "torch==2.10.0",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from wigglystuff import Matrix


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Linear Algebra for Machine Learning

    Linear algebra is the workhorse of machine learning. Nearly every
    machine learning algorithm — from linear regression to deep neural networks — relies on
    vectors, matrices, and the operations between them.

    This notebook covers the essentials, supplementing lecture 3 of the [Modern AI Course](https://github.com/marimo-team/modernaicourse):

    | Topic | What you'll learn |
    |-------|------------------|
    | **Vectors** | Creation, addition, subtraction, scalar multiplication |
    | **Matrices** | Arithmetic, transpose, multiplication (three views) |
    | **Linear maps** | How matrices transform space, Gram matrices |
    | **Interactive explorer** | Drag a 2x2 matrix and watch geometry change |

    All computations use **PyTorch tensors**, the same objects you'll use to
    build neural networks.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Vectors

    A **vector** is an ordered list of numbers. In machine learning we usually
    think of vectors as column vectors:

    $$
    \mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n
    $$

    **Two views:**

    - *Algebraic:* a vector is a tuple of coordinates.
    - *Geometric:* a vector is an arrow from the origin to a point in space.

    In PyTorch, vectors are **1-D tensors**.
    """)
    return


@app.cell
def _():
    v = torch.tensor([2.0, 1.0])
    w = torch.tensor([1.0, 3.0])
    v, w
    return v, w


@app.function(hide_code=True)
def plot_vectors(vectors, colors, labels, title="", ax=None):
    """Plot 2D vectors as arrows from the origin."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(5, 5))
    for vec, color, label in zip(vectors, colors, labels):
        vx, vy = float(vec[0]), float(vec[1])
        ax.annotate(
            "",
            xy=(vx, vy),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2", color=color, lw=2),
        )
        # Offset label away from origin so it doesn't overlap the arrow
        _norm = (vx**2 + vy**2) ** 0.5 or 1.0
        _ox, _oy = 0.25 * vx / _norm, 0.25 * vy / _norm
        ax.text(vx + _ox, vy + _oy, label, fontsize=12, color=color, fontweight="bold", ha="center", va="center")
    all_coords = [float(c) for vec in vectors for c in vec]
    bound = max(abs(c) for c in all_coords) + 1.5
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=13)
    return ax


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Vector addition & subtraction

    $$
    \mathbf{v} + \mathbf{w} = \begin{bmatrix} v_1 + w_1 \\ v_2 + w_2 \end{bmatrix}, \qquad
    \mathbf{v} - \mathbf{w} = \begin{bmatrix} v_1 - w_1 \\ v_2 - w_2 \end{bmatrix}
    $$

    Geometrically, $\mathbf{v} + \mathbf{w}$ is the diagonal of the
    **parallelogram** formed by $\mathbf{v}$ and $\mathbf{w}$.
    """)
    return


@app.cell(hide_code=True)
def _(v, w):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Addition
    plot_vectors([v, w, v + w], ["#4c78a8", "#e45756", "#54a24b"], ["v", "w", "v+w"], title="Addition", ax=_ax1)
    # Parallelogram dashed lines
    _ax1.plot([float(v[0]), float((v + w)[0])], [float(v[1]), float((v + w)[1])], "--", color="#999", lw=1)
    _ax1.plot([float(w[0]), float((v + w)[0])], [float(w[1]), float((v + w)[1])], "--", color="#999", lw=1)

    # Subtraction
    plot_vectors([v, w, v - w], ["#4c78a8", "#e45756", "#f58518"], ["v", "w", "v-w"], title="Subtraction", ax=_ax2)
    # Dashed line from w to v (v - w points from w to v)
    _ax2.plot([float(w[0]), float(v[0])], [float(w[1]), float(v[1])], "--", color="#999", lw=1)

    _fig.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Scalar multiplication

    $$
    c \cdot \mathbf{v} = \begin{bmatrix} c \, v_1 \\ c \, v_2 \end{bmatrix}
    $$

    - $|c| > 1$: stretches the vector
    - $|c| < 1$: shrinks the vector
    - $c < 0$: reverses direction
    """)
    return


@app.cell(hide_code=True)
def _(v):
    _scalars = [0.5, 1.0, -1.0, 2.0]
    _colors = ["#54a24b", "#4c78a8", "#e45756", "#f58518"]
    _labels = ["0.5v", "v", "-v", "2v"]

    _fig, _axes = plt.subplots(1, 4, figsize=(12, 3))
    _bound = float(2.0 * v.abs().max()) + 1.0
    for _ax, _s, _c, _l in zip(_axes, _scalars, _colors, _labels):
        _sv = _s * v
        plot_vectors([_sv], [_c], [_l], title=f"c = {_s}", ax=_ax)
        _ax.set_xlim(-_bound, _bound)
        _ax.set_ylim(-_bound, _bound)
    _fig.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Matrices

    A **matrix** is a rectangular array of numbers — a 2-D tensor in PyTorch:

    $$
    A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \in \mathbb{R}^{m \times n}
    $$
    """)
    return


@app.cell
def _():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    A, B
    return A, B


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Matrix arithmetic

    | Operation | Formula | PyTorch |
    |-----------|---------|---------|
    | Addition | $A + B$ | `A + B` |
    | Scalar multiplication | $cA$ | `c * A` |
    | Transpose | $A^\top$ | `A.T` |
    """)
    return


@app.cell
def _(A, B):
    A + B, 3 * A, A.T
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Dot product & outer product

    The **dot product** (inner product) of two vectors $\mathbf{v}, \mathbf{w} \in \mathbb{R}^n$:

    $$
    \mathbf{v}^\top \mathbf{w} = \sum_{i=1}^{n} v_i \, w_i \in \mathbb{R}
    $$

    The **outer product** produces a matrix:

    $$
    \mathbf{v} \, \mathbf{w}^\top = \begin{bmatrix} v_1 w_1 & v_1 w_2 \\ v_2 w_1 & v_2 w_2 \end{bmatrix} \in \mathbb{R}^{m \times n}
    $$
    """)
    return


@app.cell
def _(v, w):
    torch.dot(v, w), torch.outer(v, w)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Matrix multiplication

    For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, the product $C = AB \in \mathbb{R}^{m \times p}$.

    **Three equivalent views:**

    1. **Entry-wise:** $\;C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$ — each entry is a dot product of a row of $A$ with a column of $B$.

    2. **Column view:** Each column of $C$ is a linear combination of columns of $A$, with coefficients from the corresponding column of $B$.

    3. **Row view:** Each row of $C$ is a linear combination of rows of $B$, with coefficients from the corresponding row of $A$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    In PyTorch, the `@` operator is used for matrix multiplication:
    """)
    return


@app.cell
def _(A, B):
    C = A @ B
    C
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Matrix as a linear map

    Multiplying a matrix by a vector **applies a linear transformation**:

    $$
    \mathbf{y} = A\mathbf{x} = x_1 \begin{bmatrix} a_{11} \\ a_{21} \end{bmatrix} + x_2 \begin{bmatrix} a_{12} \\ a_{22} \end{bmatrix}
    $$

    The output is a **linear combination of the columns** of $A$, weighted by
    the entries of $\mathbf{x}$. This means:

    - The **columns of $A$** are where the standard basis vectors $\mathbf{e}_1, \mathbf{e}_2$ get mapped.
    - **Lines through the origin stay lines** (linear maps preserve linearity).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Gram matrix

    The **Gram matrix** $G = A^\top A$ captures the inner products between
    columns of $A$. It is always **symmetric** ($G = G^\top$) and **positive
    semi-definite**.

    The Gram matrix appears in PCA, kernel methods, and the normal equations
    for least squares.
    """)
    return


@app.cell
def _(A):
    G = A.T @ A
    G, G == G.T
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Interactive: explore 2D transformations

    Drag the matrix entries below to see how different matrices transform the
    plane. Try these classic transformations:

    | Transformation | Matrix |
    |---------------|--------|
    | Identity | $\begin{bmatrix}1&0\\0&1\end{bmatrix}$ |
    | Scaling | $\begin{bmatrix}s_x&0\\0&s_y\end{bmatrix}$ |
    | Rotation by $\theta$ | $\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}$ |
    | Reflection (y-axis) | $\begin{bmatrix}-1&0\\0&1\end{bmatrix}$ |
    | Shear | $\begin{bmatrix}1&k\\0&1\end{bmatrix}$ |
    """)
    return


@app.cell
def _():
    matrix_widget = mo.ui.anywidget(
        Matrix(matrix=np.eye(2).tolist(), step=1, min_value=-3.0, max_value=3.0)
    )
    return (matrix_widget,)


@app.cell
def _(matrix_widget):
    matrix_widget
    return


@app.cell(hide_code=True)
def _(matrix_widget):
    # Read the current matrix from the widget
    M = torch.tensor(matrix_widget.matrix, dtype=torch.float32)

    # Unit circle points
    _theta = torch.linspace(0, 2 * torch.pi, 200)
    _circle = torch.stack([torch.cos(_theta), torch.sin(_theta)])  # (2, 200)
    _transformed = M @ _circle  # (2, 200)

    # Basis vectors
    _e1 = torch.tensor([1.0, 0.0])
    _e2 = torch.tensor([0.0, 1.0])
    _Me1 = M @ _e1
    _Me2 = M @ _e2

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Original
    _ax1.plot(_circle[0].numpy(), _circle[1].numpy(), color="#4c78a8", lw=2)
    plot_vectors([_e1, _e2], ["#e45756", "#54a24b"], ["e₁", "e₂"], ax=_ax1)
    _ax1.set_title("Original", fontsize=13)

    # Transformed
    _ax2.plot(_transformed[0].numpy(), _transformed[1].numpy(), color="#4c78a8", lw=2)
    plot_vectors([_Me1, _Me2], ["#e45756", "#54a24b"], ["Me₁", "Me₂"], ax=_ax2)
    _ax2.set_title("Transformed", fontsize=13)

    # Match axis limits
    _all_pts = torch.cat([_circle, _transformed], dim=1)
    _bound = max(float(_all_pts.abs().max()) + 0.5, 2.0)
    for _ax in [_ax1, _ax2]:
        _ax.set_xlim(-_bound, _bound)
        _ax.set_ylim(-_bound, _bound)

    _fig.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Geometric formulas

    **Rotation** by angle $\theta$ (counter-clockwise):

    $$
    R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
    $$

    **Reflection** across the $x$-axis: $\begin{bmatrix}1&0\\0&-1\end{bmatrix}$, across the $y$-axis: $\begin{bmatrix}-1&0\\0&1\end{bmatrix}$.

    **Dilation** (uniform scaling): $\begin{bmatrix}c&0\\0&c\end{bmatrix}$ stretches ($c>1$) or shrinks ($0<c<1$) all directions equally.
    """)
    return


@app.cell
def _(v):
    # Rotate v by 45 degrees
    _angle = torch.tensor(torch.pi / 4)
    R = torch.tensor([
        [torch.cos(_angle), -torch.sin(_angle)],
        [torch.sin(_angle),  torch.cos(_angle)],
    ])
    v_rotated = R @ v

    _fig, _ax = plt.subplots(figsize=(5, 5))
    plot_vectors(
        [v, v_rotated],
        ["#4c78a8", "#e45756"],
        ["v", "Rv (45°)"],
        title="Rotation by 45°",
        ax=_ax,
    )
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary

    | Concept | Math | PyTorch |
    |---------|------|---------|
    | Vector addition | $\mathbf{v} + \mathbf{w}$ | `v + w` |
    | Scalar multiplication | $c\mathbf{v}$ | `c * v` |
    | Dot product | $\mathbf{v}^\top\mathbf{w}$ | `torch.dot(v, w)` |
    | Outer product | $\mathbf{v}\mathbf{w}^\top$ | `torch.outer(v, w)` |
    | Matrix multiply | $AB$ | `A @ B` |
    | Transpose | $A^\top$ | `A.T` |
    | Gram matrix | $A^\top A$ | `A.T @ A` |

    **Key takeaway:** Every matrix encodes a linear map. Understanding how
    matrices act on vectors — stretching, rotating, reflecting — gives you
    geometric intuition for the transformations at the heart of machine learning.
    """)
    return


if __name__ == "__main__":
    app.run()
