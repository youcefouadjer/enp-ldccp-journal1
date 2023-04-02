import torch
import numpy as np
import ptflops as pt
import argparse
import models


def complexity_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--net', default='GestureNet', type=str)
    parser.add_argument('--repetitions', default=300, type=int, help='N° of repetitions for inference time')
    parser.add_argument('--list', nargs='+', default=[16, 9, 32], help="compute complexity")
    args = parser.parse_args()

    return args


args = complexity_parser()
input_data = args.list
input_data = tuple(input_data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## Conditions


# A) Define function to compute complexity in FLOPs and model size in millions of parameters.

def flops_counter(network, x):
    with torch.cuda.device(0):
        macs, params = pt.get_model_complexity_info(network, x, as_strings=True, print_per_layer_stat=True,
                                                    verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# B) Define a function inference_time()

def inference_time(network, input_data, repetitions):
    input_data = torch.randn(input_data).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))

    # GPU-WARM-UP
    for _ in range(10):
        _ = network(input_data)

    # MEASURE PERFORMANCE

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = network(input_data)
            ender.record()
            # wait for GPU syncronization
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)


if 'MobileNetV1' in args.net:
    model = models.MobileNetV1(input_data[0], input_data[1], input_data[2], num_classes=2, layers=[2, 2, 6, 1])
    print("commuting complexity and inference time.. model-->", args.net)

elif 'MobileNetV2' in args.net:
    print("commuting complexity and inference time.. model-->", args.net)
    model = models.MobileNetV2(input_data[0], input_data[1], input_data[2], num_classes=2)


elif 'EfficientNetB0' in args.net:
    model = models.EfficientNet(input_data[0], input_data[1], input_data[2], num_classes=2)



else:
    model = models.GestureNet(input_data[0], input_data[1], input_data[2], num_classes=2)

model.to(device)

print("1st compute complexity.......\n")

flops_counter(model, input_data)

print("2nd compute infernece time...\n")
inference_time(model, input_data, args.repetitions)
