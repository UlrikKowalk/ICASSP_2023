import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import COLORS


list_files = [
    'coordinates_known_DNNfull_SNR_20_T60_0.5_uncertainty_0.0',
    'coordinates_known_DNNfull_SNR_20_T60_0.5_uncertainty_0.01',
    'coordinates_known_DNNfull_SNR_20_T60_0.5_uncertainty_0.02',
    'coordinates_known_DNNfull_SNR_20_T60_0.5_uncertainty_0.03',
    'coordinates_known_DNNfull_SNR_20_T60_0.5_uncertainty_0.04',
    'coordinates_known_DNNfull_SNR_20_T60_0.5_uncertainty_0.05',
    'coordinates_known_DNNfull_SNR_20_T60_0.5_uncertainty_0.06',
    'coordinates_known_DNNfull_SNR_20_T60_0.5_uncertainty_0.07',
    'coordinates_known_DNNfull_SNR_20_T60_0.5_uncertainty_0.08',
    'coordinates_known_DNNfull_SNR_20_T60_0.5_uncertainty_0.09',
    'coordinates_known_DNNfull_SNR_20_T60_0.5_uncertainty_0.1'
            ]

x_labels = ['0.0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.1']


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
        # [float(i) for i in line.split()]
        accuracy = np.zeros(len(error_list))
        for idx, error in enumerate(error_list):
            accuracy[idx] = (error <= (margin / num_classes * 360))
        return sum(accuracy) / len(prediction)


num_classes = 72
snr_low = 0
snr_high = 30

margin = 1

attention_base = []
attention_mean_dnn = []
attention_mean_srp = []
attention_mean_music = []
attention_median_dnn = []
attention_median_srp = []
attention_median_music = []
attention_label_dnn = []
attention_label_srp = []
attention_label_music = []

data_dnn = []
data_srp = []
data_music = []
data_all = []

fac = 5.0

for item in list_files:
    # df = pd.read_csv(f'Attention_Multisource/{item}.csv')
    df = pd.read_csv(f'data/{item}.csv')
    attention_label_dnn.append('CNN: ' + item[16:])
    attention_label_srp.append('SRPPHAT: ' + item[16:])
    attention_label_music.append('MUSIC: ' + item[16:])
    attention_base.append(len(attention_base))
    # attention_base.append(len(attention_base))

    # These are now all values with matching SNR and T60
    # snr_limited = df.loc[df['SNR'] > snr_low]
    # snr_limited = snr_limited.loc[df['SNR'] <= snr_high]
    snr_limited = df

    data_dnn.append(calculate_accuracy(snr_limited['Prediction'].tolist(), snr_limited['Target'].tolist(), num_classes, 5))
    data_srp.append(calculate_accuracy(snr_limited['Prediction_SRPPHAT'].tolist(), snr_limited['Target'].tolist(), num_classes, 5))
    data_music.append(calculate_accuracy(snr_limited['Prediction_MUSIC'].tolist(), snr_limited['Target'].tolist(), num_classes, 5))
    data_all.append(calculate_accuracy(snr_limited['Prediction'].tolist(), snr_limited['Target'].tolist(), num_classes, margin=margin))
    data_all.append(calculate_accuracy(snr_limited['Prediction_SRPPHAT'].tolist(), snr_limited['Target'].tolist(), num_classes, margin=margin))
    data_all.append(calculate_accuracy(snr_limited['Prediction_MUSIC'].tolist(), snr_limited['Target'].tolist(), num_classes, margin=margin))
    # data_srp.append(angular_error(snr_limited['Prediction_srpphat'].tolist(), snr_limited['Target'].tolist(), num_classes))

# ### boxplot

# print(attention_base)
list_median = np.median(data_dnn, axis=-1)
list_median_srp = np.median(data_srp, axis=-1)
list_median_music = np.median(data_music, axis=-1)

fig1, ax = plt.subplots()
# plt.plot([0, 1, 2, 3, 4], list_median[0:5], 'b', label='M=3')
# plt.plot([0, 1, 2, 3, 4], list_median[0:5], '.k')
# plt.plot([0, 1, 2, 3, 4], list_median[5:10], 'r', label='M=5')
# plt.plot([0, 1, 2, 3, 4], list_median[5:10], '.k')
# plt.plot([0, 1, 2, 3, 4], list_median[10:15], 'g', label='M=10')
# plt.plot([0, 1, 2, 3, 4], list_median[10:15], '.k')
# plt.plot([0, 1, 2, 3, 4], list_median[15:20], '.k')
# plt.plot([0, 1, 2, 3, 4], list_median[15:20], 'c', label='M=15')


# plt.plot([0, 1, 2, 3, 4], list_median_srp[0:5], '--b')
# plt.plot([0, 1, 2, 3, 4], list_median_srp[0:5], '.k')
# plt.plot([0, 1, 2, 3, 4], list_median_srp[5:10], '--r')
# plt.plot([0, 1, 2, 3, 4], list_median_srp[5:10], '.k')
# plt.plot([0, 1, 2, 3, 4], list_median_srp[10:15], '--g')
# plt.plot([0, 1, 2, 3, 4], list_median_srp[10:15], '.k')
# plt.plot([0, 1, 2, 3, 4], list_median_srp[15:20], '.k')
# plt.plot([0, 1, 2, 3, 4], list_median_srp[15:20], '--c', label='M=15')

# plt.boxplot(data_dnn, showfliers=False, positions=range(0, 33, 3), patch_artist=True, boxprops=dict(color='blue', facecolor='white'), medianprops=dict(color='black'), widths=([0.6]*11))
# plt.boxplot(data_srp, showfliers=False, positions=[1, 4], patch_artist=True, boxprops=dict(color='red', facecolor='white'), medianprops=dict(color='black'), widths=(0.6, 0.6))
# plt.boxplot(data_music, showfliers=False, positions=[2, 5], patch_artist=True, boxprops=dict(color='green', facecolor='white'), medianprops=dict(color='black'), widths=(0.6, 0.6))

plt.plot(data_dnn, 'b')
plt.plot(data_srp, 'r')
plt.plot(data_music, 'g')

plt.plot(-10, 1, 'b', label='DNN')
plt.plot(-10, 1, 'r', label='SRP-PHAT')
plt.plot(-10, 1, 'g', label='MUSIC')
# plt.title('Effect of attention on estimation performance')
# ax.set_xticks(range(0, 11))
# ax.set_xticklabels(x_labels)
# ax.set_axisbelow(True)
# ax.yaxis.grid(color='lightgray', linestyle='dashed')
# plt.yscale('log')
plt.ylabel('Angular error [deg]')
plt.xlabel('Coordinate deviation [m]')
# plt.xticks(rotation=45, ha='right')
# plt.plot([1.5, 1.5], [-5, 38*fac], ':', color="lightgrey")
# plt.plot([3.5, 3.5], [-5, 38*fac], ':', color="lightgrey")
# plt.plot([5.5, 5.5], [-5, 38*fac], ':', color="lightgrey")
# plt.ylim((-1, 38*fac))
fig1.set_size_inches(7, 4)
plt.xlim([-0.5, 33.5])
# plt.ylim([0, list_scaling[idx]])
plt.legend()
plt.title('Sensitivity towards wrong coordinates [T60=0.5s, SNR=20dB]')
# plt.gcf().subplots_adjust(bottom=0.2)
# plt.yscale('log')
plt.show()

# plt.savefig(f'DNN_vs_SRPPHAT_vs_MUSIC_with_wrong_coordinates', bbox_inches='tight', transparent="True", pad_inches=0.1, dpi=300)
# plt.close()



#
# Accuracy
# fig2, ax2 = plt.subplots()
# box0 = plt.bar(0, data_all[0], width=0.6, color='b')
# box2 = plt.bar(2, data_all[2], width=0.6, color='b')
# box4 = plt.bar(4, data_all[4], width=0.6, color='b')
# box1 = plt.bar(1, data_all[1], width=0.6, color='r')
# box3 = plt.bar(3, data_all[3], width=0.6, color='r')
# box5 = plt.bar(5, data_all[5], width=0.6, color='r')
# plt.plot(-10, 1, 'b', label='CNN')
# plt.plot(-10, 1, 'r', label='SRP-PHAT')
# # plt.title('Effect of attention on estimation performance')
# ax2.set_xticks([0, 1, 2, 3, 4, 5])
# ax2.set_xticklabels(['0.13s', '0.13s', '0.5s', '0.5s', '1.0s', '1.0s'])
# ax2.set_axisbelow(True)
# ax2.yaxis.grid(color='lightgray', linestyle='dashed')
# # plt.yscale('log')
# plt.ylabel('Angular error [deg]')
# # plt.xlabel('Reverberation time [s]')
# # plt.xticks(rotation=45, ha='right')
# # plt.plot([1.5, 1.5], [-5, 38 * fac], ':', color="lightgrey")
# # plt.plot([3.5, 3.5], [-5, 38 * fac], ':', color="lightgrey")
# # plt.plot([5.5, 5.5], [-5, 38 * fac], ':', color="lightgrey")
# # plt.ylim((-1, 38*fac))
# fig2.set_size_inches(7, 4)
# plt.xlim([-0.5, 5.5])
# # plt.ylim([0, list_scaling[idx]])
# plt.legend(loc='lower right')
# plt.gcf().subplots_adjust(bottom=0.2)
# plt.show()
# #
# plt.savefig(f'Accuracy_T60_{dB}dB', bbox_inches='tight', transparent="True", pad_inches=0.1, dpi=300)
# # plt.close()
#
#

print('done.')