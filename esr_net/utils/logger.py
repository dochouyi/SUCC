# Copyright (c) OpenMMLab. All rights reserved.
import logging

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = get_logger(name='crowd', log_file=log_file, log_level=log_level)
    return logger
