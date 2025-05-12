from typing import Any

import polars
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import openood.utils.comm as comm
from openood.postprocessors import BasePostprocessor
from openood.utils.cc_kde import DimensionWiseKdeOOD
from openood.utils.col_div import ColumnarDistributionDivergence
from openood.utils.sampling import stratified_subset_dataloader
from openood.utils.scanner import NetworkScanner
from sklearn.metrics import roc_auc_score, average_precision_score


class LikelihoodPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(LikelihoodPostprocessor, self).__init__(config)
        self.config = config

        self.target_layer_names: list = ["layer4.1.conv2", "fc"]
        self.tuned_depth: int | None = None

        self.reference_activations: polars.DataFrame | None = None
        self.inference_layer_names: list | None = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        # Hook activations from defined layers
        scanner = NetworkScanner(net, target_layer_names=self.target_layer_names)
        #id_loader_dict["train"] = stratified_subset_dataloader(id_loader_dict["train"], 10,1)
        self.reference_activations, _ = scanner.predict(id_loader_dict["train"])

        # Compute discriminatory power of every activation
        discriminator = ColumnarDistributionDivergence(self.reference_activations)
        col_dis_discr = discriminator.transform(
            like=self.target_layer_names,
            label_col="label",
            method="hellinger",
            bin_selection_method="fd",
        )
        cols = col_dis_discr.sort("average_divergence", descending=True)

        # Hook activations from ID and OOD validation data
        id_scan, _ = scanner.predict(id_loader_dict["val"])
        ood_scan, _ = scanner.predict(ood_loader_dict["val"])

        # Tuning: Divergence Depth
        depths = list(range(5, 201 + 1, 5))
        for depth in depths:

            self.inference_layer_names = cols.head(depth).get_column("column").to_list()

            scan_subset = self.reference_activations.select(self.inference_layer_names)
            scan_subset = scan_subset.with_columns(self.reference_activations.get_column('label'))

            likelihood_ood = DimensionWiseKdeOOD()
            likelihood_ood.fit(scan_subset, like=self.inference_layer_names, label_col="label")

            id_scan_subset = id_scan.select(self.inference_layer_names)
            ood_scan_subset = ood_scan.select(self.inference_layer_names)

            pred_id = likelihood_ood.transform(id_scan_subset, metrics_to_compute=['max_log_likelihood'])
            pred_ood = likelihood_ood.transform(ood_scan_subset, metrics_to_compute=['max_log_likelihood'])

            score_id = pred_id['max_log_likelihood'].to_list()
            score_ood = pred_ood['max_log_likelihood'].to_list()

            neg_pred_id = [-score for score in score_id]
            neg_pred_ood = [-score for score in score_ood]

            labels = [0] * len(neg_pred_id) + [1] * len(neg_pred_ood)
            scores = neg_pred_id + neg_pred_ood

            print(f"Tuning Run with depth: {depth}")
            print("AUROC:", roc_auc_score(labels, scores))
            print("AUPR:", average_precision_score(labels, scores))
            print(f"Balanced: {(roc_auc_score(labels, scores) + average_precision_score(labels, scores))/2}")

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self, net: nn.Module, data_loader: DataLoader, progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(
            data_loader, disable=not progress or not comm.is_main_process()
        ):
            data = batch["data"]#.cuda()
            label = batch["label"]#.cuda()
            pred, conf = self.postprocess(net, data)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).detach().numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list
