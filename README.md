# "How to build a viewer for (almost) anything" tutorial 🖥️🚀

## Installation

Start by creating a conda environment and installing all the dependencies
```bash
conda create -n viewer-tutorial python=3.11
conda activate viewer-tutorial
pip install -r requirements.txt
```

In the background, this will install a [custom fork](git@github.com:clementjambon/ps-py-plus.git) of Polyscope and [ps-utils](https://github.com/clementjambon/ps-utils), a library with a ton of additional utils.

## 00: A bunny!

To check that everything is running properly, run the following minimal example:
```bash
python 00_bunny.py
```

## 01: Laplacian smoothing 🏄
