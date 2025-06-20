# "How to build a viewer for (almost) anything" tutorial 🖥️🚀

## Installation

Start by cloning this repo
```bash
git clone https://github.com/clementjambon/viewer-tutorial.git
cd viewer-tutorial
```

Then, create a conda environment and install all the dependencies:
```bash
conda create -n viewer-tutorial python=3.11
conda activate viewer-tutorial
pip install -r requirements.txt
```

In the background, this will install a [custom fork](https://github.com/clementjambon/ps-py-plus) of Polyscope and [ps-utils](https://github.com/clementjambon/ps-utils), a library with a ton of additional utils.

To test your installation, please run:
```bash
python test_installation.py
```

If a window opens and stays open, then you have successfully installed everything. Congrats! 🥳

## 00: A bunny! 🐰

To render our favorite bunny, let's try:
```bash
python 00_bunny.py
```

## 01: Laplacian smoothing 🏄

TODO

## 02: Cornell Box 📦

TODO

## 03: Springy Simulation 🔗

TODO