import itertools

import polars
import torch
import torch.nn as nn

from typing import Any

from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.postprocessors import BasePostprocessor
from openood.utils import comm
from openood.utils.cc_kde import DimensionWiseKdeOOD
from openood.utils.col_div import ColumnarDistributionDivergence
from openood.utils.cut_off import ScreeCutoffSelector
from openood.utils.scanner import NetworkScanner


class LikelihoodProfilingPostprocessor(BasePostprocessor):

    def __init__(self, config):
        super(LikelihoodProfilingPostprocessor, self).__init__(config)

        self.target_layer_names: list = ["layer4.1.conv1", "layer4.1.conv2", "fc"]
        self.inference_layer_names: list | None = None

        self.APS_mode: bool = False

        self.dw_kde = DimensionWiseKdeOOD()
        self.reference_activations: polars.DataFrame | None = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        scanner = NetworkScanner(net, target_layer_names=self.target_layer_names)

        # original_train_loader = id_loader_dict["train"]
        # one_batch_train_iterator = itertools.islice(original_train_loader, 1)
        # id_loader_dict = {"train": one_batch_train_iterator}

        scan, _ = scanner.predict(id_loader_dict["train"])
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

        selector = ScreeCutoffSelector(sensitivity=0.1)
        cutoff_id = selector.propose_cutoff(
            col_dis_discr, value_col="average_divergence", method="kneedle"
        )

        first_n_cols = col_dis_discr.head(cutoff_id)
        first_n_cols = first_n_cols.select("column")
        self.inference_layer_names = first_n_cols.get_column("column").to_list()

        scan_subset = scan.select(self.inference_layer_names)
        scan_subset = scan_subset.with_columns(scan.get_column('label'))
        self.dw_kde.fit(data=scan_subset, like=self.inference_layer_names, label_col="label")

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        scanner = NetworkScanner(net, target_layer_names=self.target_layer_names)
        scan, output = scanner.predict(data)

        pp_data = scan.select(self.inference_layer_names)
        class_ll = self.dw_kde.transform(pp_data)

        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)

        return pred, class_ll

    def inference(self, net: nn.Module, data_loader: DataLoader, progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(
                data_loader, disable=not progress or not comm.is_main_process()
        ):
            data = batch["data"].cuda()
            label = batch["label"].cuda()
            pred, conf = self.postprocess(net, data)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())