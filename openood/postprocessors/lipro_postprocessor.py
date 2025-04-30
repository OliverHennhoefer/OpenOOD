import polars
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from openood.postprocessors import BasePostprocessor
from openood.utils.cc_kde import DimensionWiseKdeOOD
from openood.utils.col_div import ColumnarDistributionDivergence
from openood.utils.cut_off import ScreeCutoffSelector
from openood.utils.scanner import NetworkScanner

import math

def invert_values_scaled(data):
  min_val = min(data)
  max_val = max(data)
  value_range = max_val - min_val

  # Handle case where all elements are the same (range is zero)
  # Check floats carefully
  is_float = any(isinstance(x, float) for x in data)
  if (is_float and math.isclose(value_range, 0.0)) or (not is_float and value_range == 0):
      # Avoid division by zero. Scaled value is undefined.
      # Conventionally, return the middle of the target scale (0.5 for 0-1).
      return [0.5] * len(data)

  # Calculate inverted scaled value: 1 - (x - min_val) / value_range
  # Simplified algebraicly to: (max_val - x) / value_range
  inverted_scaled_list = [(max_val - x) / value_range for x in data]
  return inverted_scaled_list


class LikelihoodProfilingPostprocessor(BasePostprocessor):

    def __init__(self, config):
        super(LikelihoodProfilingPostprocessor, self).__init__(config)

        self.target_layer_names: list = ["layer4.1.conv2", "fc"]
        self.inference_layer_names: list | None = None

        self.APS_mode: bool = False

        self.dw_kde = DimensionWiseKdeOOD()
        self.reference_activations: polars.DataFrame | None = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        scanner = NetworkScanner(net, target_layer_names=self.target_layer_names)
        scan, _ = scanner.predict(id_loader_dict["train"])
        scan.write_parquet("1_scan.parquet")
        print(scan)
        self.reference_activations = scan

        discriminator = ColumnarDistributionDivergence(self.reference_activations)
        col_dis_discr = discriminator.transform(
            like=self.target_layer_names,
            label_col="label",
            method="hellinger",
            bin_selection_method="fd",
        )

        col_dis_discr.write_parquet('2_discr.parquet')

        if "first_n" in self.config:
            select_first_n = self.config['first_n']
        else:
            selector = ScreeCutoffSelector(sensitivity=0.1)
            select_first_n = selector.propose_cutoff(
                col_dis_discr, value_col="average_divergence", method="kneedle"
            )

        first_n_cols = col_dis_discr.head(select_first_n).select("column")
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

        conf = [-score for score in class_ll]

        results_dict = {
            "prediction": pred,
            "likelihood_difference": conf,
            "true_label": torch.cat(all_labels, dim=0).tolist()
        }

        results_df = polars.DataFrame(results_dict)
        results_df.write_parquet("3_result.parquet")

        return pred, conf, torch.cat(all_labels, dim=0).tolist()
