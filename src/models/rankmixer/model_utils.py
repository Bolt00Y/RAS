# !/usr/bin/env python3
# encoding=utf-8

"""Small RankMixer-local helpers.

The original v1 model imports ``.model_utils`` but the module was missing from
this package.  Keeping the helper here makes both v1 and v2 self-contained and
preserves the optimized training/export split used by the surrounding stack.
"""

import logging

import tensorflow as tf

try:
    from cayman.python import layer_norm_for_train
except ImportError:
    layer_norm_for_train = None
    logging.warning("cayman layer_norm_for_train is unavailable; using tf.contrib LayerNorm")


def layer_norm(input_tensor, name=None, export=False):
    """LayerNorm on the final dimension with a safe TensorFlow fallback."""
    if not export and layer_norm_for_train is not None:
        return layer_norm_for_train(
            input_tensor,
            begin_norm_axis=-1,
            begin_params_axis=-1,
            scope=name,
        )
    return tf.contrib.layers.layer_norm(
        input_tensor,
        begin_norm_axis=-1,
        begin_params_axis=-1,
        scope=name,
    )
