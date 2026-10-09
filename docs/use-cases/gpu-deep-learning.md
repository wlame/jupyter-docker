---
tags:
  - use-cases
  - gpu
  - deep-learning
---

# GPU deep learning

## Scenario

You're training neural networks and want the container to use your machine's
NVIDIA GPU instead of the CPU. The `deeplearn` target carries PyTorch,
TorchVision, TorchAudio, TensorFlow, and Keras; `full` carries the same stack
alongside everything else. PyTorch in either image can hand work to a GPU once
Docker is wired to the NVIDIA runtime; TensorFlow runs on the CPU (see
[TensorFlow runs on the CPU](#tensorflow-runs-on-the-cpu)).

## Prerequisites

- An NVIDIA GPU with a host driver from the **R580 branch or newer**. The PyPI
  torch wheels bundle the CUDA 13.0 runtime libraries (the `nvidia-*` packages in
  the lockfile), and CUDA 13.x applications run on drivers 580 and later. Neither
  the host nor the image needs a CUDA toolkit install.
- The **NVIDIA Container Toolkit** installed and configured on the host — this is
  what lets `docker run --gpus` expose the GPU to the container.
- Docker installed and running.

## Complete example

Run the prebuilt `deeplearn` image with the GPU attached:

```bash
docker run --rm --gpus all -p 8888:8888 \
  ghcr.io/wlame/jupyter-docker:deeplearn
```

Open the tokened `http://localhost:8888/lab?token=...` URL from the terminal (see
[Run JupyterLab](run-jupyterlab.md) for the token flow), then confirm the GPU is
visible from inside a notebook:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
```

## Walkthrough

### Step 1: Attach the GPU

`--gpus all` is the flag that exposes every host GPU to the container. It works
only when the NVIDIA Container Toolkit is installed on the host — without it,
Docker rejects the flag or the container falls back to CPU. To expose a specific
device instead of all of them, use `--gpus '"device=0"'`.

### Step 2: Mount your work (optional but recommended)

Add the standard mounts so notebooks and data persist on the host, exactly as in
[Run JupyterLab](run-jupyterlab.md):

```bash
docker run --rm --gpus all -p 8888:8888 \
  -v "$(pwd)/notebooks:/home/jupyter/notebooks" \
  -v "$(pwd)/data:/home/jupyter/data" \
  ghcr.io/wlame/jupyter-docker:deeplearn
```

### Step 3: Verify the GPU from Python

The `torch.cuda.is_available()` check above returns `True` when the driver,
toolkit, and container runtime line up. If it returns `False`, the notebook is
running on CPU — see [Troubleshooting](../troubleshooting.md).

!!! note "No torch.compile in the TensorFlow images"
    `deeplearn` and `full` bundle both PyTorch and TensorFlow, and triton (torch's
    GPU kernel compiler) segfaults when TensorFlow is already loaded, so these
    images do not install it. torch.compile also needs a C/C++ compiler that the
    runtime images don't ship. Eager PyTorch on the GPU is unaffected. See
    [Troubleshooting](../troubleshooting.md).

## Expected output

`torch.cuda.is_available()` prints `True` and the second line prints your GPU's
name, for example:

```
True
NVIDIA GeForce RTX 4090
```

If the GPU isn't wired up, the first line prints `False` and the second prints
`no GPU` — the container still runs, just on CPU.

## Variations

### Use the `full` image

`full` carries the same deep-learning stack plus every other library. Swap the
tag when you need GPU deep learning alongside other domains:

```bash
docker run --rm --gpus all -p 8888:8888 \
  ghcr.io/wlame/jupyter-docker:full
```

### Build locally and run with `just`

To run a locally built image with the standard mounts, build first, then attach
the GPU by hand (the [`just run`](../reference/cli.md) recipe does not pass `--gpus`):

```bash
just build deeplearn
docker run --rm --gpus all -p 8888:8888 \
  -v "$(pwd)/notebooks:/home/jupyter/notebooks" \
  -v "$(pwd)/data:/home/jupyter/data" \
  ds-deeplearn
```

### TensorFlow runs on the CPU

TensorFlow is installed from PyPI without its `and-cuda` extra, and the image ships
no CUDA 12 libraries for it, so TensorFlow ignores the GPU even with `--gpus all`:

```python
import torch
import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))  # [] in these images
```

Use PyTorch for GPU training in these images.

## Related pages

- [Run JupyterLab](run-jupyterlab.md)
- [Pick the right target](pick-a-target.md)
- [Troubleshooting](../troubleshooting.md)
- [Image targets](../reference/targets.md)
