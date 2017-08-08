#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'

import argparse
import sys
import time



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='source', type=int,
                        default=0, help='Device index of the camera.')
    args = parser.parse_args()



