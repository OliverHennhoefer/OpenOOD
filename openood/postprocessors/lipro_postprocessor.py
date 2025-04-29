import polars
import torch
import torch.nn as nn

from typing import Any

from openood.postprocessors import BasePostprocessor
from openood.utils.col_div import ColumnarDistributionDivergence
from openood.utils.scanner import NetworkScanner


class LikelihoodProfilingPostprocessor(BasePostprocessor):

    def __init__(self, config):
        super(LikelihoodProfilingPostprocessor, self).__init__(config)

        self.target_layer_names: list = ["layer4.1.conv1", "layer4.1.conv2", "fc"]
        self.reference_activations: polars.DataFrame | None = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        scanner = NetworkScanner(net, target_layer_names=self.target_layer_names)
        scan = scanner.predict(id_loader_dict["train"])
        self.reference_activations = scan

        discriminator = ColumnarDistributionDivergence(self.reference_activations)
        col_dis_discr = discriminator.transform(
            like=self.target_layer_names,
            label_col="label",
            method="hellinger",
            bin_selection_method="fd"
        )

    # @torch.no_grad()
    # def postprocess(self, net: nn.Module, data: Any):
