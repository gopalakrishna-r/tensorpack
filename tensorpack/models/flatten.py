
import numpy as np
from ..compat import tfv1 as tf  # this should be avoided first in model code

from .common import layer_register

__all__ = ['Flatten']



@layer_register(log_shape=True)
def Flatten(
        inputs,
        data_format='channels_last'):
    """
    Same as `tf.keras.layers.Flatten`. Default strides is equal to pool_size.
    """
    layer = tf.keras.layers.Flatten(data_format=data_format)
    ret = layer.apply(inputs)
    return tf.identity(ret, name='output')
