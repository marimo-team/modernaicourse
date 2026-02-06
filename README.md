<p align="center"><img src="/assets/marimo-modernaicourse-companion-logo.png"></p>

<p align="center">
  <a href="https://discord.gg/djbyJ9SZrC" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://www.youtube.com/@modernaicourse" target="_blank"><strong>Lectures</strong></a> ·
  <a href="https://modernaicourse.org" target="_blank"><strong>Course Website</strong></a>
</p>

_This repo will be updated weekly, as the online version of CMU's Intro to Modern AI progresses. [Join the discussion forum](https://discord.gg/djbyJ9SZrC) to learn with peers._

This is a companion repo to [Intro to Modern AI](https://modernaicourse.org), a
class by [CMU professor Zico Kolter](https://zicokolter.com/) that
teaches you how to train your own LLM from scratch using PyTorch — no prior
experience with machine learning or AI required.

In this repo, you will find:

- [`concepts/`](/concepts/): [marimo](https://marimo.io) notebooks that illustrate key concepts from
lecture videos, with code and UI elements
- [`homework/`](/homework/): notebooks that test your understanding of
the material and work up to training your own LLM

**Running notebooks online.**
The easiest way to run the notebooks in this repo is with [molab](https://molab.marimo.io/notebooks);
just click the links in this README.

**Running notebooks locally.** To run locally with the [marimo library](https://docs.marimo.io), [install
`uv`](https://docs.astral.sh/uv/getting-started/installation/), download the notebook you want to run from [`concepts/`](/concepts/) or [`homework/`](/homework/), then run

```bash
uvx marimo edit --sandbox <notebook>
```

For example, `uvx marimo edit --sandbox hw0.py`.

## Concepts

Each concept notebook is associated with a lecture.

| Lecture | Notebook |
| ------- | ---------- |
| [Lecture 2: Intro to supervised machine learning](https://www.youtube.com/watch?v=xIQkf7ZGQhM) | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/modernaicourse/blob/main/concepts/02_supervised_learning.py) |
| [Lecture 3: Linear Algebra](https://youtu.be/tlbLH77GFLQ?si=3DSmmVXzqxIJrHUP) | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marimo-team/modernaicourse/blob/main/concepts/03_linear_algebra.py) |

## Homework

- [HW 0: Autograding and programming basics](https://molab.marimo.io/github.com/marimo-team/modernaicourse/blob/main/homework/hw0/hw0.py)
- [HW 1: Introduction to linear algebra and PyTorch](https://molab.marimo.io/github.com/marimo-team/modernaicourse/blob/main/homework/hw1/hw1.py)

## About the course

Intro to Modern AI is a course by [CMU professor Zico Kolter](https://zicokolter.com/). From the [course
website](https://modernaicourse.org/):

> This course provides an introduction to how modern AI systems work. By
> “modern AI”, we specifically mean the machine learning methods and large
> language models (LLMs) behind systems like ChatGPT, Gemini, and Claude.
> Despite their seemingly amazing generality, the basic techniques that
> underlie these AI models are surprisingly simple: a minimal LLM
> implementation leverages a fairly small set of machine learning methods and
> architectures, and can be written in a few hundred lines of code.
>
> This course will guide you through the basic methods that will let you
> implement a basic AI chatbot. You will learn the basics of supervised machine
> learning, large language models, and post-training. By the end of the course
> you will be able to write the code that runs an open source LLM from scratch,
> as well as train these models based upon a corpus of data.
