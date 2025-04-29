import itertools

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

        # original_train_loader = id_loader_dict["train"]
        # one_batch_train_iterator = itertools.islice(original_train_loader, 1)
        # id_loader_dict = {"train": one_batch_train_iterator}

        scan = scanner.predict(id_loader_dict["train"])
        scan.write_csv("scan.csv")
        print(scan)
        self.reference_activations = scan

        discriminator = ColumnarDistributionDivergence(self.reference_activations)
        col_dis_discr = discriminator.transform(
            like=self.target_layer_names,
            label_col="label",
            method="hellinger",
            bin_selection_method="fd",
        )

        col_dis_discr.write_csv('discr.csv')

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        return 1, 1, 1

