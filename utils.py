import math
import random

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
from torchvision.utils import make_grid


def plot_1d_dataset(data):
    samples = random.sample(data, 25)
    fig = plt.figure(figsize=(15, 6))
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

    for ax, example in zip(grid, samples):
        ax.imshow(example.T)

    fig.supxlabel("step")
    fig.supylabel("fov")
    fig.tight_layout()


def plot_2d_dataset(data):
    samples = random.sample(data, 25)
    for i in range(samples[0].shape[0]):
        grid = make_grid([torch.from_numpy(sam[i : i + 1]) for sam in samples], nrow=5, padding=5)
        cv2.namedWindow("samples", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("samples", 800, 800)
        cv2.imshow("samples", grid.permute(1, 2, 0).numpy())
        cv2.waitKey(100)
