import argparse
# from Timer import Timer
import math
import os
import time

import numpy as np
import pandas as pd
import torch.nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import Evaluation
from DNN_GADOAE5_max import DNN_GADOAE5_max
from Dataset_Testing_GADOAE_max_Size import Dataset_Testing_GADOAE_max_Size
from MUSIC import MUSIC
from SRP_PHAT import SRP_PHAT

NUM_SAMPLES = 10000
BATCH_SIZE = 1
MAX_THETA = 360.0
NUM_CLASSES = 72
NUM_WORKERS = 15

BASE_DIR_ML = os.getcwd()
SAMPLE_DIR_GENERATIVE = BASE_DIR_ML + "/libriSpeechExcerpt/"
NOISE_TABLE = BASE_DIR_ML + "/noise/noise_table.mat"

LIST_SNR = [20]
LIST_T60 = [0.50]
LIST_UNCERTAINTY = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

PARAMETERS = {'base_dir': BASE_DIR_ML,
              'sample_dir': SAMPLE_DIR_GENERATIVE,
              'noise_table': NOISE_TABLE,
              'sample_rate': 8000,
              'signal_length': 5,
              'min_rt_60': 0.13,
              'max_rt_60': 0.13,
              'min_snr': 30,
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
              'min_array_width': 0.4,
              'rasterize_array': False,
              'sensor_grid_digits': 2, #2: 0.01m
              'num_classes': 72,
              'num_samples': NUM_SAMPLES,
              'max_uncertainty': 0.00,
              'dimensions_array': 2}


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')
    device = "cpu"
    print(f"Using device '{device}'.")
    list_mean = []

    for SNR in LIST_SNR:
        for T60 in LIST_T60:



            for UNCERTAINTY in LIST_UNCERTAINTY:

                PARAMETERS['min_snr'] = SNR
                PARAMETERS['max_snr'] = SNR
                PARAMETERS['min_rt_60'] = T60
                PARAMETERS['max_rt_60'] = T60
                PARAMETERS['max_uncertainty'] = UNCERTAINTY

                print(f'SNR: {SNR}, T60: {T60}, UNCERTAINTY: {UNCERTAINTY}')

                dataset = Dataset_Testing_GADOAE_max_Size(parameters=PARAMETERS, device=device)
                class_mapping = dataset.get_class_mapping()
                num_classes = dataset.get_num_classes()

                n_true = 0
                n_false = 0
                list_rt_60_testing = []
                list_snr_testing = []
                list_signal_type_testing = []
                list_ir_type_testing = []

                print('Testing')

                num_zeros = int(np.ceil(np.log10(NUM_SAMPLES)) + 1)

                n_total = len(dataset)
                n_test = int(n_total * 1)
                n_val = n_total - n_test

                test_data, val_data = torch.utils.data.random_split(dataset=dataset, lengths=[n_test, n_val])

                test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                              persistent_workers=True, shuffle=True)

                idx = 0

                list_occ = [0] * NUM_CLASSES

                list_size = []

                idx = 0

                for max_size in test_data_loader:
                    list_size.append(max_size.item())
                    print(idx)
                    idx += 1

                mean_max = np.mean(list_size)
                list_mean.append(np.mean(list_size))


    print(list_mean)
    plt.plot(list_mean)
    plt.show()
    print("done.")




    # os.system('shutdown -s')