import polars
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from openood.postprocessors import BasePostprocessor
from openood.utils.cc_kde import DimensionWiseKdeOOD
from openood.utils.cc_kde_multi import DimensionWiseKdeOODMulti
from openood.utils.col_div import ColumnarDistributionDivergence
from openood.utils.sampling import stratified_subset_dataloader
from openood.utils.scanner import NetworkScanner

class LikelihoodProfilingPostprocessor(BasePostprocessor):

    def __init__(self, config):
        super(LikelihoodProfilingPostprocessor, self).__init__(config)

        self.target_layer_names: list = ["layer4.1.conv2", "fc"]
        self.inference_layer_names: list | None = None

        self.APS_mode: bool = False
        self.config = config

        #self.dw_kde = DimensionWiseKdeOOD()
        self.dw_kde = DimensionWiseKdeOODMulti()
        self.reference_activations: polars.DataFrame | None = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        id_loader_dict["train"] = stratified_subset_dataloader(id_loader_dict["train"], 100, 1)
        scanner = NetworkScanner(net, target_layer_names=self.target_layer_names)
        scan, _ = scanner.predict(id_loader_dict["train"])  # val
        scan.write_parquet("1_scan.parquet")
        self.reference_activations = scan

        discriminator = ColumnarDistributionDivergence(self.reference_activations)
        col_dis_discr = discriminator.transform(
            like=self.target_layer_names,
            label_col="label",
            method="hellinger",
            bin_selection_method="fd",
        )

        col_dis_discr.write_parquet('2_discr.parquet')

        select_first_n = self.config["postprocessor"]["first_n"]
        first_n_cols = col_dis_discr.sort("average_divergence", descending=True).head(select_first_n).select("column")
        self.inference_layer_names = first_n_cols.get_column("column").to_list()
        print(f"\nPicking {self.inference_layer_names}")

        scan_subset = scan.select(self.inference_layer_names)
        scan_subset = scan_subset.with_columns(scan.get_column('label'))
        self.dw_kde.fit(data=scan_subset, like=self.inference_layer_names, label_col="label")

    def inference(self, net: nn.Module, data_loader: DataLoader, progress: bool = True):
        scanner = NetworkScanner(net, target_layer_names=self.target_layer_names)
        scan, output = scanner.predict(data_loader)

        pp_data = scan.select(self.inference_layer_names)
        class_ll = self.dw_kde.transform(pp_data, metrics_to_compute=['ood_metric5_robust_dissent_c_star'])  # ood_metric5_robust_dissent_c_star
        class_ll = class_ll['ood_metric5_robust_dissent_c_star'].to_list()  #ood_metric5_robust_dissent_c_star

        all_labels = [
            batch["label"].cpu()
            for batch in data_loader
            if isinstance(batch, dict) and "label" in batch and isinstance(batch["label"], torch.Tensor)
        ]

        logits_tensor = torch.tensor(output)
        prob = torch.softmax(logits_tensor, dim=1)
        pred = torch.argmax(prob, dim=1).tolist()

        conf = [-score for score in class_ll]

        results_dict = {
            "prediction": pred,
            "likelihood_difference": conf,
            "true_label": torch.cat(all_labels, dim=0).tolist()
        }

        results_df = polars.DataFrame(results_dict)
        results_df.write_parquet("3_result.parquet")

        return pred, conf, torch.cat(all_labels, dim=0).tolist()
