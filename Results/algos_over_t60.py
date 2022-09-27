import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import COLORS


list_files = [
    'coordinates_known_CNNflat_SNR_20_T60_0.13_uncertainty_0.0',
    'coordinates_known_CNNflat_SNR_20_T60_0.2_uncertainty_0.0',
    'coordinates_known_CNNflat_SNR_20_T60_0.3_uncertainty_0.0',
    'coordinates_known_CNNflat_SNR_20_T60_0.4_uncertainty_0.0',
    'coordinates_known_CNNflat_SNR_20_T60_0.5_uncertainty_0.0',
    'coordinates_known_CNNflat_SNR_20_T60_0.6_uncertainty_0.0',
    'coordinates_known_CNNflat_SNR_20_T60_0.7_uncertainty_0.0',
    'coordinates_known_CNNflat_SNR_20_T60_0.8_uncertainty_0.0',
    'coordinates_known_CNNflat_SNR_20_T60_0.9_uncertainty_0.0',
    'coordinates_known_CNNflat_SNR_20_T60_1.0_uncertainty_0.0',
    #
    'coordinates_known_DNNfull_SNR_20_T60_0.13_uncertainty_0.0',
    'coordinates_known_DNNfull_SNR_20_T60_0.2_uncertainty_0.0',
    'coordinates_known_DNNfull_SNR_20_T60_0.3_uncertainty_0.0',
    'coordinates_known_DNNfull_SNR_20_T60_0.4_uncertainty_0.0',
    'coordinates_known_DNNfull_SNR_20_T60_0.5_uncertainty_0.0',
    'coordinates_known_DNNfull_SNR_20_T60_0.6_uncertainty_0.0',
    'coordinates_known_DNNfull_SNR_20_T60_0.7_uncertainty_0.0',
    'coordinates_known_DNNfull_SNR_20_T60_0.8_uncertainty_0.0',
    'coordinates_known_DNNfull_SNR_20_T60_0.9_uncertainty_0.0',
    'coordinates_known_DNNfull_SNR_20_T60_1.0_uncertainty_0.0',
    #
    'coordinates_known_DNNmax_SNR_20_T60_0.13_uncertainty_0.0',
    'coordinates_known_DNNmax_SNR_20_T60_0.2_uncertainty_0.0',
    'coordinates_known_DNNmax_SNR_20_T60_0.3_uncertainty_0.0',
    'coordinates_known_DNNmax_SNR_20_T60_0.4_uncertainty_0.0',
    'coordinates_known_DNNmax_SNR_20_T60_0.5_uncertainty_0.0',
    'coordinates_known_DNNmax_SNR_20_T60_0.6_uncertainty_0.0',
    'coordinates_known_DNNmax_SNR_20_T60_0.7_uncertainty_0.0',
    'coordinates_known_DNNmax_SNR_20_T60_0.8_uncertainty_0.0',
    'coordinates_known_DNNmax_SNR_20_T60_0.9_uncertainty_0.0',
    'coordinates_known_DNNmax_SNR_20_T60_1.0_uncertainty_0.0'
            ]

x_labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']


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



num_classes = 72
margin = 1

data_dnn_acc = []
data_srp_acc = []
data_music_acc = []
data_dnn_rmse = []
data_srp_rmse = []
data_music_rmse = []

fac = 5.0

for item in list_files:
    df = pd.read_csv(f'data/{item}.csv')

    data_dnn_acc.append(calculate_accuracy(df['Prediction'].tolist(), df['Target'].tolist(), num_classes, 1))
    data_srp_acc.append(calculate_accuracy(df['Prediction_SRPPHAT'].tolist(), df['Target'].tolist(), num_classes, 1))
    data_music_acc.append(calculate_accuracy(df['Prediction_MUSIC'].tolist(), df['Target'].tolist(), num_classes, 1))

    data_dnn_rmse.append(calculate_rmse(df['Prediction'].tolist(), df['Target'].tolist(), num_classes))
    data_srp_rmse.append(calculate_rmse(df['Prediction_SRPPHAT'].tolist(), df['Target'].tolist(), num_classes))
    data_music_rmse.append(calculate_rmse(df['Prediction_MUSIC'].tolist(), df['Target'].tolist(), num_classes))

# ACCURACY

fig1, ax = plt.subplots()
plt.plot(data_dnn_acc[0:10], 'b', label='CNN')
plt.plot(data_dnn_acc[10:20], 'c', label='DNN_full')
plt.plot(data_dnn_acc[20:30], 'm', label='DNN_max')
plt.plot(data_srp_acc[0:10], 'r', label='SRP-PHAT')
plt.plot(data_music_acc[0:10], 'g', label='MUSIC')

plt.title('Effect of reverberation on DOA estimation Accuracy [SNR=20dB]')
ax.set_xticks(range(0, 10))
ax.set_xticklabels(x_labels)
ax.set_axisbelow(True)
ax.yaxis.grid(color='lightgray', linestyle='dashed')
plt.ylabel('Accuracy [%]')
plt.xlabel('Reverberation [s]')
fig1.set_size_inches(7, 4)
plt.legend()
plt.gcf().subplots_adjust(bottom=0.2)
# plt.show()

plt.savefig(f'Algos_over_T60_ACCURACY', bbox_inches='tight', transparent="True", pad_inches=0.1, dpi=300)
plt.close()


# RMSE

fig2, ax = plt.subplots()
plt.plot(data_dnn_rmse[0:10], 'b', label='CNN')
plt.plot(data_dnn_rmse[10:20], 'c', label='DNN_full')
plt.plot(data_dnn_rmse[20:30], 'm', label='DNN_max')
plt.plot(data_srp_rmse[0:10], 'r', label='SRP-PHAT')
plt.plot(data_music_rmse[0:10], 'g', label='MUSIC')

plt.title('Effect of reverberation on DOA estimation RMSE [SNR=20dB]')
ax.set_xticks(range(0, 10))
ax.set_xticklabels(x_labels)
ax.set_axisbelow(True)
ax.yaxis.grid(color='lightgray', linestyle='dashed')
plt.ylabel('RMSE [°]')
plt.xlabel('Reverberation [s]')
fig2.set_size_inches(7, 4)
plt.legend()
plt.gcf().subplots_adjust(bottom=0.2)
# plt.show()

plt.savefig(f'Algos_over_T60_RMSE', bbox_inches='tight', transparent="True", pad_inches=0.1, dpi=300)
plt.close()


print('done.')