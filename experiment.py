from multiprocessing import freeze_support

import torch

from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32
from openood.postprocessors import LikelihoodProfilingPostprocessor, TemperatureScalingPostprocessor

if __name__ == "__main__":
    freeze_support()
    net = ResNet18_32x32(num_classes=10)
    net.load_state_dict(
        torch.load('results/checkpoints/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt',
                   map_location=torch.device('cpu'))
    )

    #for name, module in net.named_modules():
    #    print(name, "->", module)

    lipro = LikelihoodProfilingPostprocessor(config={})
    evaluator = Evaluator(
        net,
        id_name="cifar10",  # the target ID dataset
        data_root="./data",  # change if necessary
        config_root="./config/postprocessors/lipro.yml",  # see notes above
        preprocessor=None,  # default preprocessing for the target ID dataset
        postprocessor_name=None,  # the postprocessor to use
        postprocessor=lipro,  # if you want to use your own postprocessor
        batch_size=256,  # for certain methods the results can be slightly affected by batch size
        shuffle=False,
        num_workers=2,
    )  # could use more num_workers outside colab

    metrics = evaluator.eval_ood(fsood=False)
    print(metrics)
