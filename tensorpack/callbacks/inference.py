# -*- coding: utf-8 -*-
# File: inference.py


import numpy as np
from abc import ABCMeta
import six

from ..tfutils.common import get_op_tensor_name
from ..utils import logger
from ..utils.stats import BinaryStatistics, RatioCounter
from .base import Callback

__all__ = ['ScalarStats', 'Inferencer',
           'ClassificationError', 'BinaryClassificationStats']


@six.add_metaclass(ABCMeta)
class Inferencer(Callback):
    """ Base class of Inferencer.
    Inferencer is a special kind of callback that should be called by :class:`InferenceRunner`.
    It has the methods ``_get_fetches`` and ``_on_fetches`` which are like
    :class:`SessionRunHooks`, except that they will be used only by :class:`InferenceRunner`.

    .. document private functions
    .. automethod:: _before_inference
    .. automethod:: _after_inference
    .. automethod:: _get_fetches
    .. automethod:: _on_fetches
    """

    def _before_epoch(self):
        self._before_inference()

    def _before_inference(self):
        """
        Called before a new round of inference starts.
        """
        pass

    def _trigger_epoch(self):
        ret = self._after_inference()
        if ret is None:
            return
        for k, v in six.iteritems(ret):
            try:
                v = float(v)
            except ValueError:
                logger.warn("{} returns a non-scalar statistics!".format(type(self).__name__))
                continue
            else:
                self.trainer.monitors.put_scalar(k, v)

    def _after_inference(self):
        """
        Called after a round of inference ends.
        Returns a dict of scalar statistics which will be logged to monitors.
        """
        pass

    def get_fetches(self):
        """
        Return a list of tensor names (guaranteed not op name) this inferencer needs.
        """
        ret = self._get_fetches()
        return [get_op_tensor_name(n)[1] for n in ret]

    def _get_fetches(self):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError()

    def on_fetches(self, results):
        """
        Called after each new datapoint finished the forward inference.

        Args:
            results(list): list of results this inferencer fetched. Has the same
                length as ``self._get_fetches()``.
        """
        self._on_fetches(results)

    def _on_fetches(self, results):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError()


class ScalarStats(Inferencer):
    """
    Statistics of some scalar tensor.
    The value will be averaged over all given datapoints.

    Note that the average of accuracy over all batches is not necessarily the
    accuracy of the whole dataset. See :class:`ClassificationError` for details.
    """

    def __init__(self, names, prefix='validation'):
        """
        Args:
            names(list or str): list of names or just one name. The
                corresponding tensors have to be scalar.
            prefix(str): a prefix for logging
        """
        self.names = [names] if not isinstance(names, list) else names
        self.prefix = prefix

    def _before_inference(self):
        self.stats = []

    def _get_fetches(self):
        return self.names

    def _on_fetches(self, output):
        self.stats.append(output)

    def _after_inference(self):
        if len(self.stats):
            self.stats = np.mean(self.stats, axis=0)
            assert len(self.stats) == len(self.names)

        ret = {}
        for stat, name in zip(self.stats, self.names):
            opname, _ = get_op_tensor_name(name)
            name = '{}_{}'.format(self.prefix, opname) if self.prefix else opname
            ret[name] = stat
        return ret


class ClassificationError(Inferencer):
    """
    Compute **true** classification error in batch mode, from a ``wrong`` tensor.

    The ``wrong`` tensor is supposed to be an binary vector containing
    whether each sample in the batch is *incorrectly* classified.
    You can use ``tf.nn.in_top_k`` to produce this vector.

    This Inferencer produces the "true" error, which could be different from
    ``ScalarStats('error_rate')``.
    It takes account of the fact that batches might not have the same size in
    testing (because the size of test set might not be a multiple of batch size).
    Therefore the result can be different from averaging the error rate of each batch.

    You can also use the "correct prediction" tensor, then this inferencer will
    give you "classification accuracy" instead of error.
    """

    def __init__(self, wrong_tensor_name='incorrect_vector', summary_name='validation_error'):
        """
        Args:
            wrong_tensor_name(str): name of the ``wrong`` binary vector tensor.
            summary_name(str): the name to log the error with.
        """
        self.wrong_tensor_name = wrong_tensor_name
        self.summary_name = summary_name

    def _before_inference(self):
        self.err_stat = RatioCounter()

    def _get_fetches(self):
        return [self.wrong_tensor_name]

    def _on_fetches(self, outputs):
        vec = outputs[0]
        # TODO put shape assertion into inference-runner
        assert vec.ndim == 1, "{} is not a vector!".format(self.wrong_tensor_name)
        batch_size = len(vec)
        wrong = np.sum(vec)
        self.err_stat.feed(wrong, batch_size)

    def _after_inference(self):
        return {self.summary_name: self.err_stat.ratio}


class BinaryClassificationStats(Inferencer):
    """
    Compute precision / recall in binary classification, given the
    prediction vector and the label vector.
    """

    def __init__(self, pred_tensor_name, label_tensor_name, prefix='val'):
        """
        Args:
            pred_tensor_name(str): name of the 0/1 prediction tensor.
            label_tensor_name(str): name of the 0/1 label tensor.
        """
        self.pred_tensor_name = pred_tensor_name
        self.label_tensor_name = label_tensor_name
        self.prefix = prefix

    def _before_inference(self):
        self.stat = BinaryStatistics()

    def _get_fetches(self):
        return [self.pred_tensor_name, self.label_tensor_name]

    def _on_fetches(self, outputs):
        pred, label = outputs
        self.stat.feed(pred, label)

    def _after_inference(self):
        return {
            f'{self.prefix}_precision': self.stat.precision,
            f'{self.prefix}_recall': self.stat.recall,
        }
