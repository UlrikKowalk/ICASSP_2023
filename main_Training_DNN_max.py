import argparse
import os

import torch.nn
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from DNN_max_gadoae_structure import DNN_max_gadoae_structure
from Dataset_Training_DNN_max import Dataset_Training_DNN_max
from Timer import Timer
from Training_DNN import Training_DNN

# This DNN takes as input the maxima of a 5ch-GCC-PHAT -> Trained for 1 geometry

writer = SummaryWriter("runs/gcc")

NUM_SAMPLES = 10000
EPOCHS = 1000
NUM_CLASSES = 72
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_WORKERS = 15

BASE_DIR_ML = os.getcwd()
SAMPLE_DIR_GENERATIVE = BASE_DIR_ML + "/libriSpeechExcerpt/"
NOISE_TABLE = BASE_DIR_ML + "/noise/noise_table.mat"
RATIO = 0.8

PARAMETERS = {'base_dir': BASE_DIR_ML,
              'sample_dir': SAMPLE_DIR_GENERATIVE,
              'noise_table': NOISE_TABLE,
              'sample_rate': 8000,
              'signal_length': 1,
              'min_rt_60': 0.13,
              'max_rt_60': 1.0,
              'min_snr': 0,
              'max_snr': 30,
              'room_dim': [9, 5, 3],
              'room_dim_delta': [1.0, 1.0, 0.5],
              'mic_center': [4.5, 2.5, 2],
              'mic_center_delta': [0.5, 0.5, 0.5],
              'min_source_distance': 1.0, #1.0
              'max_source_distance': 3.0, #3.0
              'proportion_noise_input': 0.5,
              'noise_lookup': True,
              'frame_length': 256,
              'num_channels': 5,
              'max_sensor_spread': 0.2, #lookup noise: only up to 0.2
              'min_array_width': 0.2,
              'rasterize_array': True,
              'sensor_grid_digits': 3, #2: 0.01m
              'num_classes': 72,
              'num_samples': NUM_SAMPLES,
              'max_uncertainty': 0.00,
              'dimensions_array': 2}

is_training = True
is_continue = False


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('net', type=ascii)
    args = parser.parse_args()

    print(f'Net: {args.net[1:-1]}')

    # with Timer("CNN computation"):

    # if torch.cuda.is_available():
    # device = 'cuda'
    #     trained_net = "cnnfourth_GPU.pth"
    # else:
    device = "cpu"
    trained_net = f'{BASE_DIR_ML}/{args.net[1:-1]}'
    print(f"Using device '{device}'.")

    if is_training:

        dataset = Dataset_Training_DNN_max(parameters=PARAMETERS, device=device)

        # creating dnn and pushing it to CPU/GPU(s)
        dnn = DNN_max_gadoae_structure(output_classes=dataset.get_num_classes())

        if is_continue:
            sd = torch.load(trained_net)
            dnn.load_state_dict(sd)

        dnn.to(device)

        loss_fn = nn.CrossEntropyLoss()

        optimiser = torch.optim.Adam(dnn.parameters(), lr=LEARNING_RATE)

        Trainer = Training_DNN(model=dnn, loss_fn=loss_fn, optimiser=optimiser, dataset=dataset,
                               batch_size=BATCH_SIZE, ratio=RATIO, num_gpu=0, device=device,
                               filename=trained_net, num_workers=NUM_WORKERS)

        with Timer("Training online"):
            # train model
            Trainer.train(epochs=EPOCHS)

    writer.close()
    print("done.")


