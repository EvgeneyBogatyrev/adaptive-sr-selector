from functools import partial

import cv2
import numpy as np


def make_slice(img, left, right, axis):
    sl = [slice(None)] * img.ndim
    sl[axis] = slice(left, right)

    return img[tuple(sl)]


def shift1d(img, gt, shift=1, axis=0):
    if shift > 0:
        x1, x2 = shift, img.shape[axis]
        x3, x4 = 0, -shift  # gt
    elif shift == 0:
        x1, x2, x3, x4 = 0, img.shape[axis], 0, img.shape[axis]
    else:
        x1, x2 = 0, shift
        x3, x4 = -shift, img.shape[axis]

    img = make_slice(img, x1, x2, axis=axis)
    gt = make_slice(gt, x3, x4, axis=axis)

    return img, gt


def shift2d(img, gt, a=1, b=1):
    img, gt = shift1d(img, gt, a, axis=0)
    img, gt = shift1d(img, gt, b, axis=1)

    return img, gt


class ERQA:
    def __init__(self, shift_compensation=True, penalize_wider_edges=None, global_compensation=True, version='1.0'):
        """
        shift_compensation - if one-pixel shifts of edges are compensated
        """
        # Set global defaults
        self.global_compensation = global_compensation
        self.shift_compensation = shift_compensation

        # Set version defaults
        if version == '1.0':
            self.penalize_wider_edges = False
        elif version == '1.1':
            self.penalize_wider_edges = True
        else:
            raise ValueError('There is no version {} for ERQA'.format(version))

        # Override version defaults
        if penalize_wider_edges is not None:
            self.penalize_wider_edges = penalize_wider_edges

        # Set detector
        self.edge_detector = partial(cv2.Canny, threshold1=100, threshold2=200)

    def __call__(self, img):
        assert img.shape[2] == 3, 'Compared images should be in BGR format'

        edge = self.edge_detector(img) // 255

        return np.linalg.norm(edge)
