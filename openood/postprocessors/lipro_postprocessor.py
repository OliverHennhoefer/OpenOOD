import torch
import torch.nn as nn

from typing import Any

from openood.postprocessors import BasePostprocessor
from openood.utils.scanner import NetworkScanner


class LikelihoodProfilingPostprocessor(BasePostprocessor):

    def __init__(self, config):
        super(LikelihoodProfilingPostprocessor, self).__init__(config)

    def setup(self, net: nn.Module, data: Any):
        """
        During the setup, inference on the training data is performed.
        During inference, (pre-)activations will be hooked and stored.
        For every in-distribution class in every hooked entity, fit a
        univariate KDE.
        The KDEs are stored for later reference.
        """
        scanner = NetworkScanner(net, target_layer_names=[])

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """
        During the postprocessing, compute the likelihood of each instance's


        """
