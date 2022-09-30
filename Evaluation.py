import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt

import FindPeaks
# from SourceManager import SourceManager


def predict_srp_phat(model, sample, target, class_mapping, PARAMETERS, MAX_THETA, NUM_CLASSES):

    prediction = model.model_no_mask(sample)
    estimates = [np.argmax(prediction)]
    # predicted_index = np.argmax(prediction)
    # estimates = FindPeaks.find_peaks(prediction)

    # estimates = scipy.signal.find_peaks(prediction)[0]
    estimates = FindPeaks.find_real_peaks(prediction, estimates, MAX_THETA, NUM_CLASSES)
    estimates = FindPeaks.sort_by_strength(estimates)[0]
    class_predicted = estimates[0]

    tar_list = []

    for val in target:
        if val != -255:
            tar_list.append(int(val))

    class_expected = np.mean(tar_list)
    # print(class_expected)

    return class_predicted, class_expected


def predict_music(model, sample, target, class_mapping, PARAMETERS, MAX_THETA, NUM_CLASSES):

    prediction = model.model_no_mask(sample)
    estimates = [np.argmax(prediction)]
    # predicted_index = np.argmax(prediction)
    # estimates = FindPeaks.find_peaks(prediction)

    # estimates = scipy.signal.find_peaks(prediction)[0]
    estimates = FindPeaks.find_real_peaks(prediction, estimates, MAX_THETA, NUM_CLASSES)
    estimates = FindPeaks.sort_by_strength(estimates)[0]
    class_predicted = estimates[0]

    tar_list = []

    for val in target:
        if val != -255:
            tar_list.append(int(val))

    class_expected = np.mean(tar_list)
    # print(class_expected)

    return class_predicted, class_expected

def predict_no_bs(model, sample, target, class_mapping, device, PARAMETERS, MAX_THETA, NUM_CLASSES):

    model.eval()
    predictions = model(sample)
    estimates_per_frame = torch.argmax(predictions, dim=-1)
    return np.median(estimates_per_frame), target[0], 0, 0


def predict_no_bs_VAD(model, sample, target, class_mapping, device, PARAMETERS, MAX_THETA, NUM_CLASSES):
    model.eval()

    predictions = model(sample)
    estimates_per_frame = []
    targets_per_frame = []
    for pred, tar in zip(predictions, target):
        cost = pred.cpu().detach().numpy()
        max_idx = np.argmax(cost)

        if tar[0] != -255:
            estimates_per_frame.append(class_mapping[max_idx])
            targets_per_frame.append(float(tar))

    return np.median(estimates_per_frame), np.median(targets_per_frame), 0, 0



def predict_as_sum(model, sample, target, class_mapping, device, PARAMETERS, MAX_THETA, NUM_CLASSES):

    model.eval()
    predictions = model(sample)
    sum_of_predictions = torch.zeros(size=(predictions.shape[1],))

    list_targets = []
    for idx, tar in enumerate(target[:-2]):
        if tar[0] != -255:
            list_targets.append(float(tar[0][0]))
            sum_of_predictions += predictions[idx, :]

    sum_of_predictions = sum_of_predictions / len(list_targets)

    sum_of_predictions_original = torch.mean(predictions, dim=0)

    estimates = FindPeaks.find_real_peaks(sum_of_predictions, [torch.argmax(sum_of_predictions)], MAX_THETA, NUM_CLASSES)
    estimate = FindPeaks.sort_by_strength(estimates)[0]

    print(estimate)

    if len(estimate) > 1:
        index = estimate[0].cpu().detach().numpy()
        value = estimate[1].cpu().detach().numpy()
    else:
        index = estimate[0]
        value = estimate[1]

    printable_sum = sum_of_predictions.cpu().detach().numpy()
    printable_sum_original = sum_of_predictions_original.cpu().detach().numpy()

    # plt.plot(printable_sum_original, 'r')
    # plt.plot(printable_sum, 'k')
    # plt.plot(index, value, 'go')
    # plt.plot([np.median(list_targets), np.median(list_targets)], [0,value], 'b:')
    # plt.show()

    return index, np.median(list_targets), 0, 0


def predict_with_interpolation(model, sample, target, class_mapping, device, PARAMETERS, MAX_THETA, NUM_CLASSES):
    model.eval()

    # predictions = model(sample)
    predictions = model.forward(sample)
    estimates_per_frame = []
    targets_per_frame = []
    for pred, tar in zip(predictions, target):
        cost = pred.cpu().detach().numpy()
        max_idx = np.argmax(cost)
        estimates = FindPeaks.find_real_peaks(cost, [max_idx], MAX_THETA, NUM_CLASSES)
        estimate = FindPeaks.sort_by_strength(estimates)[0][0]
        if tar[0] != -255:
            estimates_per_frame.append(estimate)
            targets_per_frame.append(float(tar))

    return np.median(estimates_per_frame), np.median(targets_per_frame), 0, 0


def predict_with_interpolation_noVAD(model, sample, target, class_mapping, device, PARAMETERS, MAX_THETA, NUM_CLASSES):
    model.eval()

    predictions = model(sample)
    estimates_per_frame = []
    targets_per_frame = []
    for pred, tar in zip(predictions, target):
        cost = pred.cpu().detach().numpy()
        max_idx = np.argmax(cost)
        estimates = FindPeaks.find_real_peaks(cost, [max_idx], MAX_THETA, NUM_CLASSES)
        estimate = FindPeaks.sort_by_strength(estimates)[0][0]

        estimates_per_frame.append(estimate)
        targets_per_frame.append(float(tar))

    return np.median(estimates_per_frame), np.median(targets_per_frame), 0, 0


def predict(model, sample, target, class_mapping):
    model.eval()
    with torch.no_grad():
        prediction = model(sample)
        predicted_index = torch.argmax(prediction[0])
        class_predicted = class_mapping[predicted_index]
        class_expected = class_mapping[target]
    return class_predicted, class_expected


def predict_multiframe_with_source_management(model, sample, target, class_mapping, device, PARAMETERS, MAX_THETA, NUM_CLASSES):
    model.eval()
    with torch.no_grad():
        prediction = model(sample)

        logit_accu = np.zeros(shape=(PARAMETERS['num_classes'], sample.shape[0]))

        dt = PARAMETERS['frame_length'] / PARAMETERS['sample_rate']
        source_manager = SourceManager(dt, PARAMETERS['num_classes'], device)

        class_argmax = []
        class_expected = []
        class_all = []
        class_kalman = []
        class_kalman_all_sources = []
        all_estimates = []

        frame = 0
        for pred, tar in zip(prediction, target):

            pred = pred.cpu().detach().numpy()
            logit_accu[:, frame] = pred
            logits = pred.tolist()

            lin_logits = logits

            # lin_logits = FindPeaks.convert_logits_to_lin(logits)
            # lin_logits = logits

            # len_filter = 3
            # plt.plot(lin_logits, 'k')
            # lin_logits = scipy.signal.medfilt(volume=lin_logits, kernel_size=(len_filter,))
            # plt.plot(lin_logits, 'r')
            # plt.show()

            # estimates = FindPeaks.find_peaks(lin_logits, peak_importance=3)

            estimates = scipy.signal.find_peaks(lin_logits)[0]
            # estimates = [np.argmax(pred)]
            # print(estimates)



            # print(len(estimates))
            estimates = FindPeaks.find_real_peaks(lin_logits, estimates, MAX_THETA, NUM_CLASSES)
            estimates = FindPeaks.sort_by_strength(estimates)

            all_estimates.append(estimates)

            sources = source_manager.track_sources(estimates)

            if len(sources) > 0 and sources[0] != -255:
                frm = []
                srcs = []
                for src in sources:
                    frm.append((src.get_angle(), src.get_strength()))
                    srcs.append([src.get_angle(), src.get_strength()])
                class_kalman_all_sources.append(frm)
                srcs = sorted(srcs, key=lambda x: x[1], reverse=True)
                if np.isnan(srcs[0][0]):
                    print('NAN')
                class_kalman.append(srcs[0][0])
            else:
                class_kalman.append(-255)
                class_kalman_all_sources.append([])

            if tar != -255:

                # idx = np.argmax(pred)
                # Interpolated maximum (not really an index now)
                idx = estimates[0][0]

                class_argmax.append(idx)
                class_expected.append(class_mapping[tar])
                # class_all.append(idx)

            # else:
            #     class_all.append(None)

            frame += 1

        # fig, axs = plt.subplots(2)
        # axs[0].imshow(logit_accu, origin='lower', aspect='auto')
        #
        # for idx, frm in enumerate(all_estimates):
        #     axs[1].plot(idx, len(class_kalman_all_sources[idx]), 'k.')
        #     for esti in frm:
        #         axs[0].plot(idx, esti[0], 'wx', alpha=max([0.5, min([1.0, esti[1]])]))
        #
        # axs[0].plot(class_all, 'b--')
        #
        # # plot kalman sources
        # for idx, frm in enumerate(class_kalman_all_sources):
        #     if frm:
        #         for ifr, src in enumerate(frm):
        #             if ifr == 0:
        #                 axs[0].plot(idx, src[0], 'm.', alpha=max([0.5, min([1.0, src[1]])]))
        #             else:
        #                 axs[0].plot(idx, src[0], 'r.', alpha=max([0.5, min([1.0, src[1]])]))

        # plot argmax
        # for idx, estimates in enumerate(all_estimates):
        #     axs[0].plot(idx, estimates[0][0], 'm.')


        # axs[1].set_ylim([0, 6])
        # axs[0].set_xlim([0, frame-1])
        # axs[1].set_xlim([0, frame-1])
        # plt.show()



        class_var = np.var(class_argmax)
        class_argmax = np.median(class_argmax)
        class_expected = np.mean(class_expected)

        kalman_estimates = [i for i in class_kalman if i !=-255] # Fehlermaß ist nicht optimal weil wrap nicht berücksichtigt wird!
        if len(kalman_estimates) > 0:
            class_kalman = np.median(kalman_estimates)

        else:
            class_kalman = -255

    return class_argmax, class_expected, class_var, class_kalman


def predict_multiframe(model, sample, target, class_mapping):
    model.eval()
    with torch.no_grad():
        prediction = model(sample)
        predicted_index = torch.argmax(prediction, dim=-1)

        class_predicted = []
        class_expected = []
        class_all = []
        for idx, tar in zip(predicted_index, target):
            if tar != -255:
                class_predicted.append(class_mapping[idx])
                class_expected.append(class_mapping[tar])
                class_all.append(class_mapping[idx])
            else:
                class_all.append(-1)

        class_var = np.var(class_predicted)
        class_predicted = np.mean(class_predicted)
        class_expected = np.mean(class_expected)

    return class_predicted, class_expected, class_var


def predict_multisource(model, sample, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(sample)
        predicted_index = torch.argmax(predictions[0])
        class_predicted = class_mapping[predicted_index]

        class_expected = []
        for val in range(len(target)):
            class_expected.append(class_mapping[val])
    return class_predicted, class_expected


def predict_regression(model, sample, target):
    model.eval()
    with torch.no_grad():
        prediction = model(sample)
    return prediction.cpu().detach().numpy(), target.cpu().detach().numpy()


def angular_error(prediction, label, num_classes):
    error = np.abs(np.angle(np.exp(1j * 2 * np.pi * (prediction - label) / num_classes))) * num_classes / (2 * np.pi)
    return error


def plot_error(df, num_classes):

    targets = range(num_classes)
    errors = [[None]] * num_classes
    predictions = [[None]] * num_classes
    for target in targets:
        predictions[target] = df.loc[df['Target'] == target]['Prediction']
        tmp = df.loc[df['Target'] == target]['Prediction']
        tmp_err = []
        for err in tmp:
            tmp_err.append(angular_error(err, target, num_classes))
        errors[target] = tmp_err

    fig, ax = plt.subplots(2, 1)
    ax[0].boxplot(predictions)
    ax[1].boxplot(errors)
    ax[0].set_title('Predictions')
    ax[1].set_title('Errors')
    plt.gcf().subplots_adjust(bottom=0.1)
    plt.show()


def plot_distribution(labels, num_classes):

    target = range(num_classes)
    freq = np.zeros(num_classes)

    for label in labels:
         freq[label] += 1

    plt.bar(target, freq)
    plt.title('Target distribution')
    plt.show()


def angular_error_multisource(prediction, label, num_classes):

    print(prediction)
    print(label)

    error = 0
    for pred, lab in zip(prediction, label):
        error = error + np.abs(np.angle(np.exp(1j * 2 * np.pi * float(pred - lab) / num_classes))) * num_classes / (2 * np.pi)
    return error


def calculate_accuracy(df, num_classes):

    accuracy_model = 0
    accuracy_srpphat = 0
    accuracy_music = 0

    for idx in range(len(df)):

        tmp = df['Prediction'][idx]
        tmp_srpphat = df['Prediction_SRPPHAT'][idx]
        tmp_music = df['Prediction_MUSIC'][idx]
        target = df['Target'][idx]

        if angular_error(target, tmp, num_classes) <= 1:
            accuracy_model += 1
        if angular_error(target, tmp_srpphat, num_classes) <= 1:
            accuracy_srpphat += 1
        if angular_error(target, tmp_music, num_classes) <= 1:
            accuracy_music += 1

    return accuracy_model/len(df)*100, accuracy_srpphat/len(df)*100, accuracy_music/len(df)*100



