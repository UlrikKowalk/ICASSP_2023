import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import COLORS
from matplotlib import colors

list_files = [
    'coordinates_known_GADOAEmax_SNR_20_T60_0.5_uncertainty_0.0'
            ]

# filename = 'GADOAE_SRP-PHAT-MUSIC_random'

# new_list = []
# for item in list_files:
#     new_list.append(item.replace('0.5', '0.13'))
# list_files = new_list
# filename = filename.replace('0-50','0-13')


# x_labels = ['0.00', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.10']


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)


def angular_error(prediction, label, num_classes):
    if len(prediction) > 0:
        error = [0] * len(prediction)
        for idx in range(len(prediction)):
            error[idx] = fac * np.abs(np.angle(np.exp(1j * 2 * np.pi * float(prediction[idx] - label[idx]) / num_classes))) * num_classes / (2 * np.pi)
        return error
    else:
        return None


def calculate_accuracy(prediction, label, num_classes, margin):
    if len(prediction) > 0:
        error_list = angular_error(prediction, label, num_classes)
        accuracy = np.zeros(len(error_list))
        for idx, error in enumerate(error_list):
            accuracy[idx] = (error <= (margin / num_classes * 360))
        return sum(accuracy) / len(prediction) * 100


def calculate_rmse(prediction, label, num_classes):
    if len(prediction) > 0:
        error_list = angular_error(prediction, label, num_classes)
        return np.sqrt(np.mean(np.power(error_list, 2)))


def calculate_mae(prediction, label, num_classes):
    if len(prediction) > 0:
        error_list = angular_error(prediction, label, num_classes)
        return np.mean(error_list)


num_classes = 72
margin = 1

data_dnn_acc = []
data_srp_acc = []
data_music_acc = []
data_dnn_rmse = []
data_srp_rmse = []
data_music_rmse = []
data_dnn_mae = []
data_srp_mae = []
data_music_mae = []

fac = 5.0

for idx, item in enumerate(list_files):
    df = pd.read_csv(f'data/{item}.csv')

    data_dnn_acc.append(calculate_accuracy(df['Prediction'].tolist(), df['Target'].tolist(), num_classes, 1))
    data_dnn_rmse.append(calculate_rmse(df['Prediction'].tolist(), df['Target'].tolist(), num_classes))
    data_dnn_mae.append(calculate_mae(df['Prediction'].tolist(), df['Target'].tolist(), num_classes))

    data_srp_acc.append(calculate_accuracy(df['Prediction_SRPPHAT'].tolist(), df['Target'].tolist(), num_classes, 1))
    data_music_acc.append(calculate_accuracy(df['Prediction_MUSIC'].tolist(), df['Target'].tolist(), num_classes, 1))
    data_srp_rmse.append(calculate_rmse(df['Prediction_SRPPHAT'].tolist(), df['Target'].tolist(), num_classes))
    data_music_rmse.append(calculate_rmse(df['Prediction_MUSIC'].tolist(), df['Target'].tolist(), num_classes))
    data_srp_mae.append(calculate_mae(df['Prediction_SRPPHAT'].tolist(), df['Target'].tolist(), num_classes))
    data_music_mae.append(calculate_mae(df['Prediction_MUSIC'].tolist(), df['Target'].tolist(), num_classes))


print(f'GADOAE: Accuracy: {data_dnn_acc[0]}, MAE: {data_dnn_mae[0]}, RMSE: {data_dnn_rmse[0]}')
print(f'SRP-PHAT: Accuracy: {data_srp_acc[0]}, MAE: {data_srp_mae[0]}, RMSE: {data_srp_rmse[0]}')
print(f'MUSIC: Accuracy: {data_music_acc[0]}, MAE: {data_music_mae[0]}, RMSE: {data_music_rmse[0]}')

