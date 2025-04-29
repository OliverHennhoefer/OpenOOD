import polars
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from openood.postprocessors import BasePostprocessor
from openood.utils.cc_kde import DimensionWiseKdeOOD
from openood.utils.col_div import ColumnarDistributionDivergence
from openood.utils.cut_off import ScreeCutoffSelector
from openood.utils.scanner import NetworkScanner


class LikelihoodProfilingPostprocessor(BasePostprocessor):

    def __init__(self, config):
        super(LikelihoodProfilingPostprocessor, self).__init__(config)

        self.target_layer_names: list = ["layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"]
        self.inference_layer_names: list | None = None

        self.APS_mode: bool = False

        self.dw_kde = DimensionWiseKdeOOD()
        self.reference_activations: polars.DataFrame | None = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        scanner = NetworkScanner(net, target_layer_names=self.target_layer_names)
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

        #selector = ScreeCutoffSelector(sensitivity=0.1)
        #cutoff_id = selector.propose_cutoff(
        #    col_dis_discr, value_col="average_divergence", method="kneedle"
        #)

        first_n_cols = col_dis_discr.head(100).select("column")
        self.inference_layer_names = first_n_cols.get_column("column").to_list()

        scan_subset = scan.select(self.inference_layer_names)
        scan_subset = scan_subset.with_columns(scan.get_column('label'))
        self.dw_kde.fit(data=scan_subset, like=self.inference_layer_names, label_col="label")

    def inference(self, net: nn.Module, data_loader: DataLoader, progress: bool = True):
        scanner = NetworkScanner(net, target_layer_names=self.target_layer_names)
        scan, output = scanner.predict(data_loader)

        pp_data = scan.select(self.inference_layer_names)
        class_ll = self.dw_kde.transform(pp_data)
        class_ll = class_ll['ood_score_likelihood_difference'].to_list()

        all_labels = [
            batch["label"].cpu()
            for batch in data_loader
            if isinstance(batch, dict) and "label" in batch and isinstance(batch["label"], torch.Tensor)
        ]

        logits_tensor = torch.tensor(output)
        prob = torch.softmax(logits_tensor, dim=1)
        pred = torch.argmax(prob, dim=1).tolist()

        return pred, class_ll, torch.cat(all_labels, dim=0).tolist()