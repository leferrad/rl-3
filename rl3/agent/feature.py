#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'

import numpy as np


class LBPFeatureTransformer(object):
    def __init__(self):
        self.dimensions = 6
        self.n_points = 9

    @staticmethod
    def transform(state, normalize=False):
        state = np.array(state).reshape(6, 9).astype(np.int32)
        codes = []
        for face in state:
            center = face[4]
            borders = np.concatenate([face[:4], face[5:]])
            code = [b == center for b in borders]  # list of booleans
            code = int(''.join(map(lambda c: str(int(c)), code)), 2)  # convert it to an integer number
            #code = int(''.join(reversed(sorted(map(lambda c: str(int(c)), code)))), 2)  # convert it to an integer number
            if normalize is True:
                code /= 255.0  # Normalize to [0,1] range

            codes.append(code)
        return codes

    @staticmethod
    def hist_lbp_code(code):
        bins = [0, 32, 64, 96, 128, 160, 192, 224, 256]
        #bins = []
        hist, _ = np.histogram(code, bins=bins)
        return list(hist)