import argparse
# from Timer import Timer
import math
import os

import numpy as np
import pandas as pd
import torch.nn
from torch.utils.data import DataLoader

import Evaluation
from DNN_GADOAE5_max import DNN_GADOAE5_max
from Dataset_Testing_GADOAE_max import Dataset_Testing_GADOAE_max
from MUSIC import MUSIC
from SRP_PHAT import SRP_PHAT

NUM_SAMPLES = 100
BATCH_SIZE = 1
MAX_THETA = 360.0
NUM_CLASSES = 72
NUM_WORKERS = 1

BASE_DIR_ML = os.getcwd()
SAMPLE_DIR_GENERATIVE = BASE_DIR_ML + "/libriSpeechExcerpt/"
NOISE_TABLE = BASE_DIR_ML + "/noise/noise_table.mat"

LIST_SNR = [20]
LIST_T60 = [0.50]
LIST_UNCERTAINTY = [0.00]#, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

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
              'rasterize_array': True,
              'sensor_grid_digits': 3, #2: 0.01m
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

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('net', type=ascii)
    args = parser.parse_args()

    print(f'Net: {args.net[1:-1]}')

    device = "cpu"
    if torch.cuda.is_available():
        device_inference = 'cuda'
    else:
        device_inference = device

    trained_net = f'{BASE_DIR_ML}/{args.net[1:-1]}'
    print(f"Using device '{device}'.")

    for SNR in LIST_SNR:
        for T60 in LIST_T60:
            for UNCERTAINTY in LIST_UNCERTAINTY:

                PARAMETERS['min_snr'] = SNR
                PARAMETERS['max_snr'] = SNR
                PARAMETERS['min_rt_60'] = T60
                PARAMETERS['max_rt_60'] = T60
                PARAMETERS['max_uncertainty'] = UNCERTAINTY

                print(f'SNR: {SNR}, T60: {T60}')

                dataset = Dataset_Testing_GADOAE_max(parameters=PARAMETERS, device=device)

                # creating dnn and pushing it to CPU/GPU(s)
                dnn = DNN_GADOAE5_max(num_channels=PARAMETERS['num_channels'],
                          num_dimensions=PARAMETERS['dimensions_array'],
                          output_classes=dataset.get_num_classes())

                map_location = torch.device(device)
                sd = torch.load(trained_net, map_location=map_location)

                dnn.load_state_dict(sd)
                dnn.to(device_inference)

                class_mapping = dataset.get_class_mapping()
                num_classes = dataset.get_num_classes()

                n_true = 0
                n_false = 0
                list_rt_60_testing = []
                list_snr_testing = []
                list_signal_type_testing = []
                list_ir_type_testing = []

                print('Testing')

                list_predictions = []
                list_predictions_srpphat = []
                list_predictions_music = []
                list_targets = []
                list_var = []
                list_kalman = []
                list_error = []
                list_error_srpphat = []
                list_error_music = []

                num_zeros = int(np.ceil(np.log10(NUM_SAMPLES)) + 1)

                n_total = len(dataset)
                n_test = int(n_total * 1)
                n_val = n_total - n_test

                test_data, val_data = torch.utils.data.random_split(dataset=dataset, lengths=[n_test, n_val])

                test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                              persistent_workers=True, shuffle=True)

                idx = 0

                list_weird = []
                list_occ = [0] * NUM_CLASSES

                for bulk_sample, bulk_target, coordinates, parameters, y in test_data_loader:

                    srp_phat = SRP_PHAT(num_channels=y.shape[1], coordinates=coordinates, parameters=PARAMETERS, device=device)
                    music = MUSIC(num_channels=y.shape[1], coordinates=coordinates, parameters=PARAMETERS)

                    predicted, expected, variance, kalman = Evaluation.predict_with_interpolation(model=dnn,
                                                                                             sample=bulk_sample.squeeze(dim=0),
                                                                                             target=bulk_target,
                                                                                             class_mapping=class_mapping,
                                                                                             device=device_inference,
                                                                                             PARAMETERS=PARAMETERS,
                                                                                             MAX_THETA=MAX_THETA,
                                                                                             NUM_CLASSES=NUM_CLASSES)

                    predicted_srpphat, _ = Evaluation.predict_srp_phat(model=srp_phat,
                                                                              sample=y.squeeze(dim=0),
                                                                              target=bulk_target,
                                                                              class_mapping=class_mapping,
                                                                              PARAMETERS=PARAMETERS,
                                                                              MAX_THETA=MAX_THETA,
                                                                              NUM_CLASSES=NUM_CLASSES)

                    predicted_music, _ = Evaluation.predict_music(model=music,
                                                                              sample=y.squeeze(dim=0),
                                                                              target=bulk_target,
                                                                              class_mapping=class_mapping,
                                                                              PARAMETERS=PARAMETERS,
                                                                              MAX_THETA=MAX_THETA,
                                                                              NUM_CLASSES=NUM_CLASSES)

                    list_occ[int(expected)] += 1

                    list_predictions.append(predicted)
                    list_predictions_srpphat.append(predicted_srpphat)
                    list_predictions_music.append(predicted_music)
                    list_targets.append(expected)
                    # list_var.append(variance)
                    # list_kalman.append(kalman)
                    list_rt_60_testing.append(parameters['rt_60'])
                    list_snr_testing.append(parameters['snr'])
                    list_signal_type_testing.append(parameters['signal_type'])

                    list_error.append(Evaluation.angular_error(expected, predicted, NUM_CLASSES) / NUM_CLASSES * MAX_THETA)
                    list_error_srpphat.append(
                        Evaluation.angular_error(expected, predicted_srpphat, NUM_CLASSES) / NUM_CLASSES * MAX_THETA)
                    list_error_music.append(
                        Evaluation.angular_error(expected, predicted_music, NUM_CLASSES) / NUM_CLASSES * MAX_THETA)

                    if list_error[idx] > 10:
                        list_weird.append((expected, predicted))

                    print(
                        f"{idx:0{num_zeros}d}/{NUM_SAMPLES:0{num_zeros}d} DNN: Angular error: {list_error[idx]} degrees")
                    print(
                        f"{idx:0{num_zeros}d}/{NUM_SAMPLES:0{num_zeros}d} SRP: Angular error: {list_error_srpphat[idx]} degrees")
                    print(
                        f"{idx:0{num_zeros}d}/{NUM_SAMPLES:0{num_zeros}d} MUSIC: Angular error: {list_error_music[idx]} degrees")

                    idx += 1

                # Write results to pandas table
                df = pd.DataFrame({
                    'Target': list_targets,
                    'Prediction': list_predictions,
                    'Prediction_SRPPHAT': list_predictions_srpphat,
                    'Prediction_MUSIC': list_predictions_music,
                    # 'Tracked': list_kalman,
                    'T60': list_rt_60_testing,
                    'SNR': list_snr_testing,
                    'Signal Type': list_signal_type_testing
                })
                df.to_csv(
                    path_or_buf=f'Results/coordinates_known_GADOAEmax_SNR_{SNR}_T60_{T60}_uncertainty_{UNCERTAINTY}.csv',
                    index=False)

                rmse_GADOAEmax = math.sqrt(np.square(list_error).mean())
                rmse_SRPPHAT = math.sqrt(np.square(list_error_srpphat).mean())
                rmse_MUSIC = math.sqrt(np.square(list_error_music).mean())

                acc_model, acc_srpphat, acc_music = Evaluation.calculate_accuracy(df, NUM_CLASSES)

                print(
                    f"GADOAE_max: Average angular error: {np.mean(list_error)} [{np.median(list_error)}] degrees, RMSE: {rmse_GADOAEmax}, Accuracy: {acc_model}")
                print(
                    f"SRP-PHAT: Average angular error: {np.mean(list_error_srpphat)} [{np.median(list_error_srpphat)}] degrees, RMSE: {rmse_SRPPHAT}, Accuracy: {acc_srpphat}")
                print(
                    f"MUSIC: Average angular error: {np.mean(list_error_music)} [{np.median(list_error_music)}] degrees, RMSE: {rmse_MUSIC}, Accuracy: {acc_music}")

    Evaluation.plot_error(df=df, num_classes=NUM_CLASSES)
    print("done.")


    # os.system('shutdown -s')