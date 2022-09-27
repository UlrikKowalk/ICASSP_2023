import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import COLORS


list_files = [
    'coordinates_unknown_CNNflat_SNR_20_T60_0.5_uncertainty_0.0',
    'coordinates_unknown_CNNflat_SNR_20_T60_0.5_uncertainty_0.01',
    'coordinates_unknown_CNNflat_SNR_20_T60_0.5_uncertainty_0.02',
    'coordinates_unknown_CNNflat_SNR_20_T60_0.5_uncertainty_0.03',
    'coordinates_unknown_CNNflat_SNR_20_T60_0.5_uncertainty_0.04',
    'coordinates_unknown_CNNflat_SNR_20_T60_0.5_uncertainty_0.05',
    'coordinates_unknown_CNNflat_SNR_20_T60_0.5_uncertainty_0.06',
    'coordinates_unknown_CNNflat_SNR_20_T60_0.5_uncertainty_0.07',
    'coordinates_unknown_CNNflat_SNR_20_T60_0.5_uncertainty_0.08',
    'coordinates_unknown_CNNflat_SNR_20_T60_0.5_uncertainty_0.09',
    'coordinates_unknown_CNNflat_SNR_20_T60_0.5_uncertainty_0.1',
    #
    'coordinates_unknown_DNNfull_SNR_20_T60_0.5_uncertainty_0.0',
    'coordinates_unknown_DNNfull_SNR_20_T60_0.5_uncertainty_0.01',
    'coordinates_unknown_DNNfull_SNR_20_T60_0.5_uncertainty_0.02',
    'coordinates_unknown_DNNfull_SNR_20_T60_0.5_uncertainty_0.03',
    'coordinates_unknown_DNNfull_SNR_20_T60_0.5_uncertainty_0.04',
    'coordinates_unknown_DNNfull_SNR_20_T60_0.5_uncertainty_0.05',
    'coordinates_unknown_DNNfull_SNR_20_T60_0.5_uncertainty_0.06',
    'coordinates_unknown_DNNfull_SNR_20_T60_0.5_uncertainty_0.07',
    'coordinates_unknown_DNNfull_SNR_20_T60_0.5_uncertainty_0.08',
    'coordinates_unknown_DNNfull_SNR_20_T60_0.5_uncertainty_0.09',
    'coordinates_unknown_DNNfull_SNR_20_T60_0.5_uncertainty_0.1',
    #
    'coordinates_unknown_DNNmax_SNR_20_T60_0.5_uncertainty_0.0',
    'coordinates_unknown_DNNmax_SNR_20_T60_0.5_uncertainty_0.01',
    'coordinates_unknown_DNNmax_SNR_20_T60_0.5_uncertainty_0.02',
    'coordinates_unknown_DNNmax_SNR_20_T60_0.5_uncertainty_0.03',
    'coordinates_unknown_DNNmax_SNR_20_T60_0.5_uncertainty_0.04',
    'coordinates_unknown_DNNmax_SNR_20_T60_0.5_uncertainty_0.05',
    'coordinates_unknown_DNNmax_SNR_20_T60_0.5_uncertainty_0.06',
    'coordinates_unknown_DNNmax_SNR_20_T60_0.5_uncertainty_0.07',
    'coordinates_unknown_DNNmax_SNR_20_T60_0.5_uncertainty_0.08',
    'coordinates_unknown_DNNmax_SNR_20_T60_0.5_uncertainty_0.09',
    'coordinates_unknown_DNNmax_SNR_20_T60_0.5_uncertainty_0.1',
            ]

# new_list = []
# for item in list_files:
#     new_list.append(item.replace('0.5', '0.13'))
# list_files = new_list

x_labels = ['0.00', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.10']

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

for idx, item in enumerate(list_files):
    df = pd.read_csv(f'data/{item}.csv')

    data_dnn_acc.append(calculate_accuracy(df['Prediction'].tolist(), df['Target'].tolist(), num_classes, 1))
    data_dnn_rmse.append(calculate_rmse(df['Prediction'].tolist(), df['Target'].tolist(), num_classes))

    data_srp_acc.append(calculate_accuracy(df['Prediction_SRPPHAT'].tolist(), df['Target'].tolist(), num_classes, 1))
    data_music_acc.append(calculate_accuracy(df['Prediction_MUSIC'].tolist(), df['Target'].tolist(), num_classes, 1))
    data_srp_rmse.append(calculate_rmse(df['Prediction_SRPPHAT'].tolist(), df['Target'].tolist(), num_classes))
    data_music_rmse.append(calculate_rmse(df['Prediction_MUSIC'].tolist(), df['Target'].tolist(), num_classes))

# ACCURACY

fig1, ax = plt.subplots()
plt.plot(data_dnn_acc[0:11], 'b', label='CNN')
plt.plot(data_dnn_acc[11:22], 'c', label='DNN_full')
plt.plot(data_dnn_acc[22:33], 'm', label='DNN_max')
plt.plot(data_srp_acc[0:11], 'r', label='SRP-PHAT')
plt.plot(data_music_acc[0:11], 'g', label='MUSIC')

plt.title('Effect of coordinate deviation on DOA estimation Accuracy')
ax.set_xticks(range(0, 11))
ax.set_xticklabels(x_labels)
ax.set_axisbelow(True)
ax.yaxis.grid(color='lightgray', linestyle='dashed')
plt.ylabel('Accuracy [%]')
plt.xlabel('Deviation [m]')
fig1.set_size_inches(7, 4)
plt.legend()
plt.ylim([0,100])
plt.gcf().subplots_adjust(bottom=0.2)
# plt.show()

plt.savefig(f'Algos_when_coordinates_change_ACCURACY', bbox_inches='tight', transparent="True", pad_inches=0.1, dpi=300)
plt.close()


# RMSE

fig2, ax = plt.subplots()
plt.plot(data_srp_rmse[0:11], 'k:', label='SRP-PHAT')
plt.plot(data_music_rmse[0:11], 'k--', label='MUSIC')
plt.plot(data_dnn_rmse[0:11], 'k-', label='CNN')
plt.plot(data_dnn_rmse[11:22], 'k-d', label='DNN_full')
plt.plot(data_dnn_rmse[22:33], 'k-x', label='DNN_max')

plt.title('Effect of coordinate deviation on DOA estimation performance')
ax.set_xticks(range(0, 11))
ax.set_xticklabels(x_labels)
ax.set_axisbelow(True)
ax.yaxis.grid(color='lightgray', linestyle='dashed')
plt.ylabel('RMSE [°]')
plt.xlabel('Coordinate deviation [m]')
fig2.set_size_inches(7, 3.5)
plt.legend()
plt.gcf().subplots_adjust(bottom=0.1)
# plt.show()

plt.savefig(f'Algos_when_coordinates_change_RMSE', bbox_inches='tight', transparent="True", pad_inches=0.1, dpi=300)
plt.close()


print('done.')

### BOTH


fig1, axs = plt.subplots(2, sharex=True)
axs[0].plot(data_srp_acc[0:11], 'k:', label='SRP-PHAT')
axs[0].plot(data_music_acc[0:11], 'k--', label='MUSIC')
axs[0].plot(data_dnn_acc[0:11], 'k-', label='CNN')
axs[0].plot(data_dnn_acc[11:22], 'k-d', label='DNN_full')
axs[0].plot(data_dnn_acc[22:33], 'k-x', label='DNN_max')

# plt.title('Effect of coordinate deviation on DOA estimation Accuracy [T60=0.5s, SNR=20dB]')
axs[0].set_xticks(range(0, 11))
axs[0].set_xticklabels(x_labels)
axs[0].set_yticks([0, 25, 50, 75, 100])
axs[0].set_axisbelow(True)
axs[1].set(ylim = [0, 105])
axs[0].yaxis.grid(color='lightgray', linestyle='dashed')
axs[0].set(ylabel ='Accuracy [%]')
# axs[0].xlabel('Deviation [m]')
# fig1.set_size_inches(7, 4)
# axs[0].legend()
# axs[0].ylim([0,100])
plt.gcf().subplots_adjust(bottom=0.2)

axs[1].plot(data_srp_rmse[0:11], 'k:', label='SRP-PHAT')
axs[1].plot(data_music_rmse[0:11], 'k--', label='MUSIC')
axs[1].plot(data_dnn_rmse[0:11], 'k-', label='CNN')
axs[1].plot(data_dnn_rmse[11:22], 'k-d', label='DNN_full')
axs[1].plot(data_dnn_rmse[22:33], 'k-x', label='DNN_max')

# plt.title('Effect of coordinate deviation on DOA estimation performance')
axs[1].set_xticks(range(0, 11))
axs[1].set_xticklabels(x_labels)
axs[1].set(ylim = [-5, 90])
axs[1].set_axisbelow(True)
axs[1].yaxis.grid(color='lightgray', linestyle='dashed')
axs[1].set(ylabel = 'RMSE [°]')
axs[1].set(xlabel = 'Coordinate deviation [m]')
# fig2.set_size_inches(7, 3.5)
axs[1].legend(loc='center left', bbox_to_anchor=(0.0, 1.1), framealpha=1.0)
# plt.gcf().subplots_adjust(bottom=0.1)
# plt.show()

plt.savefig(f'Algos_when_coordinates_change_0-50_BOTH', bbox_inches='tight', transparent="True", pad_inches=0.1, dpi=300)
plt.close()


print('done.')