from typing import Any

import torch
from torch import nn
# optim is not needed anymore for this specific conformal approach
# from torch import optim
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class ConformalCalibrationPostprocessor(BasePostprocessor):
    """
    A postprocessor that calibrates model confidences using a conformal approach.
    It computes non-conformity scores on a calibration set and uses these
    to adjust the confidence of predictions at inference time, yielding p-values.
    """

    def __init__(self, config):
        super(ConformalCalibrationPostprocessor, self).__init__(config)
        self.config = config
        self.calibration_scores = None  # To store non-conformity scores from calibration set
        self.setup_flag = False
        # Define a small epsilon for numerical stability if needed, though not typically for this
        # self.epsilon = config.get('epsilon', 1e-7)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            assert "val" in id_loader_dict.keys(), \
                "No validation (calibration) dataset found for Conformal Calibration!"

            cal_dl = id_loader_dict["val"]

            # Determine device of the network
            device = next(net.parameters()).device

            print("Starting Conformal Calibration setup...")
            non_conformity_scores_list = []
            net.eval()  # Ensure model is in eval mode

            with torch.no_grad():
                for batch in tqdm(cal_dl, desc="Calibrating"):
                    data = batch["data"].to(device)
                    labels = batch["label"].to(device)

                    logits = net(data)
                    softmax_scores = torch.softmax(logits, dim=1)

                    # Gather softmax scores for the true labels
                    # labels.unsqueeze(1) makes labels [batch_size, 1] for gather
                    true_class_probs = torch.gather(softmax_scores, 1, labels.unsqueeze(1)).squeeze()

                    # Non-conformity score: 1 - probability of true class
                    batch_non_conformity_scores = 1.0 - true_class_probs
                    non_conformity_scores_list.append(batch_non_conformity_scores.cpu())

            self.calibration_scores = torch.cat(non_conformity_scores_list)

            # It can be useful to sort them, but not strictly necessary for p-value calculation
            # self.calibration_scores, _ = torch.sort(self.calibration_scores)

            print(f"Conformal calibration setup complete. "
                  f"Stored {len(self.calibration_scores)} non-conformity scores.")
            # print(f"Min calibration score: {self.calibration_scores.min():.4f}, "
            #       f"Max: {self.calibration_scores.max():.4f}, "
            #       f"Mean: {self.calibration_scores.mean():.4f}")

            self.setup_flag = True
        else:
            print("Conformal Calibration already set up.")
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        if not self.setup_flag:
            raise RuntimeError(
                "ConformalCalibrationPostprocessor has not been set up. Call 'setup' first."
            )

        # Determine device of the network and move data
        device = next(net.parameters()).device
        if isinstance(data, torch.Tensor):
            input_tensor = data.to(device)
        elif isinstance(data, dict) and "data" in data:  # handle if data is a batch dict
            input_tensor = data["data"].to(device)
        else:
            raise ValueError("Input 'data' type not recognized. Expected Tensor or dict with 'data' key.")

        net.eval()  # Ensure model is in eval mode
        logits = net(input_tensor)
        softmax_scores = torch.softmax(logits, dim=1)

        # Get raw predictions and confidences (max softmax probability)
        raw_conf, pred = torch.max(softmax_scores, dim=1)

        # Non-conformity scores for the current batch's predictions
        # s_test = 1 - max_prob(x_test)
        # Move to CPU for comparison with calibration_scores which are on CPU
        test_non_conformity_scores = (1.0 - raw_conf).cpu()

        # Ensure calibration_scores is on CPU as well (it should be from setup)
        if self.calibration_scores.device.type != 'cpu':
            self.calibration_scores = self.calibration_scores.cpu()

        # Vectorized p-value calculation:
        # For each test_score, count how many calibration_scores are >= it.
        # test_non_conformity_scores: [batch_size]
        # self.calibration_scores: [num_cal_samples]

        # Expand test_scores to [batch_size, 1]
        # Expand calibration_scores to [1, num_cal_samples]
        # Comparison then yields a [batch_size, num_cal_samples] boolean tensor
        comparison = self.calibration_scores.unsqueeze(0) >= test_non_conformity_scores.unsqueeze(1)

        # Sum over calibration samples dimension (dim=1)
        num_greater_equal = torch.sum(comparison, dim=1)

        # Calculate p-value (calibrated confidence)
        # Add 1 to numerator and denominator for the current sample itself
        calibrated_conf = (num_greater_equal.float() + 1.0) / (len(self.calibration_scores) + 1.0)

        # Move calibrated_conf to the original device of predictions
        return pred, calibrated_conf.to(pred.device)

    # _temperature_scale method is no longer needed
    # def _temperature_scale(self, logits):
    #     return logits / self.temperature