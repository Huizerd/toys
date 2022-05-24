import random

import numpy as np


def fov_1d_speed_wrap_sample(fov, length, max_seg_length, pattern, pattern_speeds):
    # random background
    background = np.random.randn(length, fov).astype(np.float32) * 0.1

    # cut up length into random-length segments
    # TODO: how to get this properly uniformly distributed?
    def segmentation(i):
        while i > 0:
            n = random.randint(1, min(i, max_seg_length))
            yield n
            i -= n

    seg_lengths = list(segmentation(length))
    random.shuffle(seg_lengths)

    # random speeds
    speeds = np.random.choice(pattern_speeds, size=len(seg_lengths))
    speeds = np.repeat(speeds, seg_lengths)

    # integrate to positions
    start = np.random.randint(0, fov)
    pos = start + np.cumsum(speeds)
    count = np.arange(length)

    # go over pattern, determine index + wrap, fill
    # TODO: can we get rid of this loop?
    # loop is faster than roll, fill, roll back
    for i, patt in enumerate(pattern):
        background[count, (pos + i) % fov] = patt

    return background, speeds.astype(np.float32), seg_lengths


def fov_1d_speed_wrap(
    fov,
    length,
    size=100,
    max_seg_length=20,
    pattern_widths=[1, 3, 5],
    pattern_speeds=[-2, -1, 0, 1, 2],
    pattern_speed_scale=10,
):
    """
    toy problem with a single pattern moving around in a 1D field of view, of which the speed (px/step) should be estimated

    details:
    - wraps around the edges
    - longer constant-speed segments

    randomization:
    - speed
    - direction
    - pattern starting position
    - pattern appearance TODO: what's the impact and can we get rid of this?
    - background noise
    """
    data, labels = [], []
    for _ in range(size):
        # select pattern
        # TODO: improve -> prevent if statement, or look at impact of different widths and remove
        width = random.choice(pattern_widths)
        high = random.uniform(0.8, 1.2)
        medium = random.uniform(0.5, 0.8)
        low = random.uniform(0.2, 0.5)
        if width == 1:
            pattern = np.array([high], dtype=np.float32)
        elif width == 3:
            pattern = np.array([low, high, low], dtype=np.float32)
        elif width == 5:
            pattern = np.array([low, medium, high, medium, low], dtype=np.float32)

        # generate sample
        x, y, _ = fov_1d_speed_wrap_sample(fov, length, max_seg_length, np.array(pattern), pattern_speeds)
        data.append(x)
        labels.append(y / pattern_speed_scale)

    return data, labels


def fov_1d_speed_bounce(
    fov, length, size=100, pattern_widths=[1, 3, 5], pattern_speeds=[-2, -1, 0, 1, 2], pattern_speed_scale=10
):
    """
    toy problem with a single pattern moving around in a 1D field of view, of which the speed (px/step) should be estimated

    details:
    - bounces on the edges
    - more direction and speed changes

    randomization:
    - speed
    - direction
    - pattern starting position
    - pattern appearance
    - background noise
    """
    data, labels = [], []

    for _ in range(size):
        width = random.choice(pattern_widths)
        high = random.uniform(0.8, 1.2)
        medium = random.uniform(0.5, 0.8)
        low = random.uniform(0.2, 0.5)

        if width == 1:
            pattern = np.array([high], dtype=np.float32)
        elif width == 3:
            pattern = np.array([low, high, low], dtype=np.float32)
        elif width == 5:
            pattern = np.array([low, medium, high, medium, low], dtype=np.float32)

        example = np.random.randn(length, fov).astype(np.float32) * 0.1

        start = random.randint(0, fov - width)
        speed = random.choice(pattern_speeds)
        label = []
        for i in range(length):
            # check if going outside of fov
            if start + width + speed > fov or start + speed < 0:
                speed = -speed
            # fill in pattern
            example[i, start : start + width] = pattern
            start += speed
            label.append(speed / pattern_speed_scale)
            # change speed
            if random.random() < 0.2:
                speed = random.choice(pattern_speeds)

        data.append(example)
        labels.append(np.array(label, dtype=np.float32))

    return data, labels


def fov_2d_speed_circle_sample(height, width, length, max_seg_length, pattern_speeds):
    # random background
    background = np.random.randn(length, height, width).astype(np.float32) * 0.1

    # random-length segments
    def segmentation(i):
        while i > 0:
            n = random.randint(1, min(i, max_seg_length))
            yield n
            i -= n

    seg_lengths = list(segmentation(length))
    random.shuffle(seg_lengths)

    # random speeds
    speeds_h = np.random.choice(pattern_speeds, size=len(seg_lengths))
    speeds_w = np.random.choice(pattern_speeds, size=len(seg_lengths))
    speeds_h = np.repeat(speeds_h, seg_lengths)
    speeds_w = np.repeat(speeds_w, seg_lengths)

    # integrate to positions
    start_h = np.random.randint(height // 4, height + height // 4)
    start_w = np.random.randint(width // 4, width + width // 4)
    pos_h = (start_h + np.cumsum(speeds_h)) % height
    pos_w = (start_w + np.cumsum(speeds_w)) % width

    # draw circle with grid method and wrap around to other side
    # such that it is always in view
    # loop is faster than vectorized 3D mgrid
    # TODO: still, is there a faster non-loop way?
    xx, yy = np.mgrid[-10 : height + 10, -10 : width + 10]  # padding to allow wrap
    for i, h, w in zip(range(length), pos_h, pos_w):
        circle = (xx - h) ** 2 + (yy - w) ** 2
        donut = (circle < 70) & (circle > (70 - 40))  # radius of sqrt(70)
        donut = np.where(donut)
        background[i, donut[0] % 50, donut[1] % 50] = 1  # wrap to other side

    return background, np.stack([speeds_h, speeds_w], axis=-1).astype(np.float32)


def fov_2d_speed_circle(
    height, width, length, size=100, max_seg_length=20, pattern_speeds=[-2, -1, 0, 1, 2], pattern_speed_scale=10
):
    """
    toy problem with a circle moving around in a 2D field of view, of which the vertical and horizontal speed (px/step) should be estimated

    details:
    - wraps around the edges
    - circle always has same size and appearance

    randomization:
    - speed
    - direction
    - circle starting position
    - background noise
    """
    data, labels = [], []
    for _ in range(size):
        x, y = fov_2d_speed_circle_sample(height, width, length, max_seg_length, pattern_speeds)
        data.append(x)
        labels.append(y / pattern_speed_scale)
    return data, labels
