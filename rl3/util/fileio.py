#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

__author__ = 'leferrad'

import logging
import logging.handlers


def get_logger(name='pliforto', level=logging.DEBUG):
    """
    Function to obtain a normal logger

    :param name: string
    :param level: instances of Python's *logging* (e.g. logging.INFO, logging.DEBUG)
    :return: logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger