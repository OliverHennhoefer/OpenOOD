from multiprocessing import freeze_support

import os
import torch

from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32

if __name__ == "__main__":
    freeze_support()

    #for name, module in net.named_modules():
    #    print(name, "->", module)

    methods = [
        "ash",
        "ebo",
        #"gmm",
        #"knn"
        #"lipro"
        "mds_ensemble",
        #"odin"
        "react",
        "she",
        "temp_scaling",
        "gen",
        "gram",
        "rmds"
    ]

    setup = "cifar10"
    chkpt = "cifar10_resnet18_32x32_base_e100_lr0.1_default"

    for method in methods:

        for chkpt_id in [0, 1, 2]:

            net = ResNet18_32x32(num_classes=10)
            net.load_state_dict(
                torch.load(f'results/checkpoints/{chkpt}/s{chkpt_id}/best.ckpt',
                           map_location=torch.device('cpu'))
            )

            evaluator = Evaluator(
                net,
                id_name="cifar10",  # the target ID dataset
                data_root="./data",  # change if necessary
                config_root='./configs',#"./config/postprocessors/lipro.yml",  # see notes above
                preprocessor=None,  # default preprocessing for the target ID dataset
                postprocessor_name=method,  # the postprocessor to use
                postprocessor=None,  # if you want to use your own postprocessor
                batch_size=256,  # for certain methods the results can be slightly affected by batch size
                shuffle=False,
                num_workers=2,
            )  # could use more num_workers outside colab

            metrics = evaluator.eval_ood(fsood=False)
            print(metrics)

            output_filename = f"./results/eval/{method}/{setup}_{method}_ckpt{chkpt_id}.txt"

            try:
                output_dir = os.path.dirname(output_filename)
                os.makedirs(output_dir, exist_ok=True)
                with open(output_filename, 'w', encoding='utf-8') as f:
                    formatted_string = str(metrics)
                    f.write(formatted_string)
                    if not formatted_string.endswith('\n'):
                        f.write('\n')
                print(f"Successfully wrote formatted object output to '{output_filename}'")
            except Exception as e:
                print(f"An error occurred: {e}")

