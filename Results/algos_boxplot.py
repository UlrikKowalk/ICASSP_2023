import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import COLORS


list_files = [
    'coordinates_known_CNNflat_SNR_20_T60_0.5_uncertainty_0.0',
    #
    'coordinates_known_DNNfull_SNR_20_T60_0.5_uncertainty_0.0',
    #
    'coordinates_known_DNNmax_SNR_20_T60_0.5_uncertainty_0.0',
    #
    'coordinates_known_GADOAEmax_SNR_20_T60_0.5_uncertainty_0.0',
    #
    'coordinates_known_GADOAEfull_SNR_20_T60_0.5_uncertainty_0.0',
            ]

x_labels = ['CNN', 'DNN_max', 'DNN_full', 'SRP-PHAT', 'MUSIC']
x_labels_gadoae = ['CNN', 'DNN_max', 'DNN_full', 'GADOAE_max', 'GADOAE_full', 'SRP-PHAT', 'MUSIC']


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
snr_low = 0
snr_high = 30

margin = 1

data_dnn = []
data_srp = []
data_music = []

fac = 5.0

for item in list_files:

    df = pd.read_csv(f'data/{item}.csv')

    data_dnn.append(angular_error(df['Prediction'].tolist(), df['Target'].tolist(), num_classes))
    data_srp.append(angular_error(df['Prediction_SRPPHAT'].tolist(), df['Target'].tolist(), num_classes))
    data_music.append(angular_error(df['Prediction_MUSIC'].tolist(), df['Target'].tolist(), num_classes))


### BOXPLOT normal

fig1, ax = plt.subplots()

box1 = plt.boxplot(data_dnn[0:3], positions=range(3), medianprops={'color':'red'}, showfliers=False, widths=([0.6]*3))
box2 = plt.boxplot(data_srp[0], positions=(3,), medianprops={'color':'red'}, showfliers=False, widths=(0.6))
box3 = plt.boxplot(data_music[0], positions=(4,), medianprops={'color':'red'}, showfliers=False, widths=(0.6))

plt.title('DOA Estimation Algorithms [T60=0.5s, SNR=20dB]')
ax.set_xticks(range(0, 5))
ax.set_xticklabels(x_labels)
ax.set_axisbelow(True)
ax.yaxis.grid(color='lightgray', linestyle='dashed')
plt.ylabel('Angular error [°]')
plt.xticks(rotation=45, ha='right')
fig1.set_size_inches(7, 4)
plt.xlim([-0.5, 4.5])
plt.gcf().subplots_adjust(bottom=0.2)
# plt.yscale('log')
# plt.show()

# plt.savefig(f'Algos_Boxplot', bbox_inches='tight', transparent="True", pad_inches=0.1, dpi=300)
# plt.close()

### BOXPLOT mit GADOAE

fig1, ax = plt.subplots()

box1 = plt.boxplot(data_dnn[0:5], positions=range(5), medianprops={'color':'red'}, showfliers=False, widths=([0.6]*5))
box2 = plt.boxplot(data_srp[0], positions=(5,), medianprops={'color':'red'}, showfliers=False, widths=(0.6))
box3 = plt.boxplot(data_music[0], positions=(6,), medianprops={'color':'red'}, showfliers=False, widths=(0.6))

plt.title('DOA Estimation Algorithms [T60=0.5s, SNR=20dB]')
ax.set_xticks(range(0, 7))
ax.set_xticklabels(x_labels_gadoae)
ax.set_axisbelow(True)
ax.yaxis.grid(color='lightgray', linestyle='dashed')
plt.ylabel('Angular error [°]')
plt.xticks(rotation=45, ha='right')
fig1.set_size_inches(7, 4)
plt.xlim([-0.5, 6.5])
plt.gcf().subplots_adjust(bottom=0.2)
# plt.yscale('log')
# plt.show()

plt.savefig(f'Algos_Boxplot_gadoae', bbox_inches='tight', transparent="True", pad_inches=0.1, dpi=300)
plt.close()


print('done.')