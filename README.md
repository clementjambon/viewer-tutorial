# "How to build a viewer for (almost) anything" tutorial 🖥️🚀

> [!WARNING]
> Please do not look at the [solutions](https://github.com/clementjambon/viewer-tutorial/tree/main/solutions) before we go through the tutorial together!

## Table of Contents

  - [Installation](#installation)
  - [00: A bunny! 🐰](#00-a-bunny-)
  - [01: Slicing SDFs 🔪](#01-slicing-sdfs-)
  - [02: Laplacian smoothing 🏄](#02-laplacian-smoothing-)
  - [03: Cornell Box 📦](#03-cornell-box-)
  - [04: Springy Simulation 🔗](#04-springy-simulation-)
  - [05: Neural Fields 🧠](#05-neural-fields-)

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

In this example, we'll see how we can display [our favorite bunny](https://en.wikipedia.org/wiki/Stanford_bunny)!

Objectives:
* Understand Polyscope's "structures"
* Play around with the viewer options
* Learn how to take screenshot

## 01: Slicing SDFs 🔪

In this example, we'll learn how to slice our favorite bunny 😱

Objectives:
* Understand `volume_grid`s
* Perform isosurface extraction
* Play with isolines
* Add a slicer

## 02: Laplacian smoothing 🏄

Time to start our first viewer by using the [BaseViewer](https://github.com/clementjambon/ps-utils/blob/main/src/ps_utils/viewer/base_viewer.py) template from [ps-utils](https://github.com/clementjambon/ps-utils).

Objectives:
* Get familiar with the [BaseViewer](https://github.com/clementjambon/ps-utils/blob/main/src/ps_utils/viewer/base_viewer.py) abstractions.
* Add a mesh drag-n-drop callback.
* Add UI components to control smoothing parameters.

For more information on mesh smoothing, I recommend to consult these either [this](https://graphics.stanford.edu/courses/cs468-12-spring/LectureSlides/06_smoothing.pdf) or [this](https://crl.ethz.ch/teaching/shape-modeling-18/lectures/07_RemeshingSmoothing.pdf).

## 03: Cornell Box 📦

With this example, I want to prove you that Polyscope isn't just for geometry processing.
We'll see how we can use it to render a scene with ray tracing.
To do this, we'll use [Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/) as an intuitive and self-contained rendering backend.

Objectives:
* Learn about Polyscope's [render buffers](https://polyscope.run/py/structures/floating_quantities/render_images/#render-image-options) and how to update them.
* Edit the scene interactively.
* Use Colorpickers (NOTE: I often use them for more than just color 😜).

> [!NOTE]
> In this example, we'll only cover how to update them on the CPU. If you want to update them directly on the GPU (without a host-device copy), check out the solutions of [Example 05](#05-neural-fields-).

If you want to learn more about Mitsuba, I recommend to follow the very well-designed [tutorials](https://mitsuba.readthedocs.io/en/stable/src/rendering_tutorials.html).

## 04: Springy Simulation 🔗

Let's make something a bit more dynamic! 🤸
In this example, we'll see how to run and control a simulation in real-time.

Objectives:
* Run a simulation in real-time.
* Introduce the [VoxelSet](https://github.com/clementjambon/ps-utils/blob/main/src/ps_utils/structures/voxel_set.py) structure and use it to select boundary conditions on-the-fly.

## 05: Neural Fields 🧠

In this example, we'll see how we can optimize a simple MLP-based [neural field](https://arxiv.org/abs/2111.11426) directly in the viewer.

Objectives:
* Run torch/gradient-based optimization directly in the viewer.
* Learn how to use [Thumbnails](https://github.com/clementjambon/ps-utils/blob/main/src/ps_utils/ui/image_utils.py).
* (Optional) See how we can update render buffers directly on the GPU.