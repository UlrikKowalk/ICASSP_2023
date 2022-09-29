import glob
import math
import random

import numpy as np
import pyroomacoustics as pra
import torch
import torchaudio
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from Coordinates import Coordinates
from Feature_GADOAE import Feature_GADOAE
from Mask import Mask
from NoiseTable import NoiseTable
from inv_sabine import inv_sabine
import scipy.io as sio


class Dataset_Testing_GADOAE_max_Size(Dataset):

    def __init__(self, parameters, device):

        self.base_dir = parameters['base_dir']
        self.class_mapping = np.arange(0, parameters['num_classes'])
        self.num_classes = parameters['num_classes']
        self.sample_rate = parameters['sample_rate']
        self.device = device

        self.mic_coordinates_array = sio.loadmat(self.base_dir + '/array_doa_dnn_5.mat')['coordinates']

        self.num_channels = parameters['num_channels']
        self.max_sensor_spread = parameters['max_sensor_spread']
        self.min_array_width = parameters['min_array_width']
        self.frame_length = parameters['frame_length']
        self.len_s = parameters['signal_length']
        self.num_samples = parameters['num_samples']
        self.nC = 344.0
        self.rasterize = parameters['rasterize_array']
        self.sensor_grid_digits = parameters['sensor_grid_digits']

        # sample loading parameters
        self.sample_dir = parameters['sample_dir'] + 'test'
        self.proportion_noise = parameters['proportion_noise_input']
        self.threshold = -4.0
        self.hop_size = int(0.5 * self.frame_length)

        # Noise parameters
        self.min_snr = parameters['min_snr']
        self.max_snr = parameters['max_snr']
        self.noise_lookup = parameters['noise_lookup']
        # self.noise_table = NoiseTable(parameters['noise_table'], self.sensor_grid_digits)

        # Room acoustics parameters
        self.max_rt_60 = parameters['max_rt_60']
        self.min_rt_60 = parameters['min_rt_60']
        self.min_source_distance = parameters['min_source_distance']
        self.max_source_distance = parameters['max_source_distance']
        self.room_dim = parameters['room_dim']
        self.room_dim_delta = parameters['room_dim_delta']

        # Array parameters
        self.mic_center = np.array(parameters['mic_center'])
        self.mic_center_delta = np.array(parameters['mic_center_delta'])
        self.max_uncertainty = parameters['max_uncertainty']
        self.dimensions_array = parameters['dimensions_array']

        self.len_IR = 4096
        self.dist_samples = -255

    def __len__(self):
        return self.num_samples

    def get_num_classes(self):
        return self.num_classes

    @staticmethod
    def calculate_level(input):
        return torch.mul(10, torch.log10(torch.std(input)))

    @staticmethod
    def normalise_level(signal):
        return signal / torch.std(signal)

    @staticmethod
    def change_level(signal, level):
        factor = 10 ** (level / 10).type(torch.float32)
        return signal * factor

    def generate_disturbance(self, noise_level_desired, length):

        # Noise type: uncorrelated
        noise = torch.randn(size=(self.num_channels, length), device=self.device, dtype=torch.float32)
        noise = self.normalise_level(noise)
        factor = 10 ** (noise_level_desired / 10).type(torch.float32)

        return noise * factor

    def get_class_mapping(self):
        return self.class_mapping

    def _hann_poisson(self, m, alpha):
        mo2 = (m - 1) / 2
        n = torch.arange(-mo2, mo2 + 1, device=self.device)
        scl = alpha / mo2
        p = torch.exp(-scl * torch.abs(n))
        scl2 = torch.pi / mo2
        h = 0.5 * (1 + torch.cos(scl2 * n))
        return p * h

    def generate_energy_label(self, signal, label):
        signal_energy = 10 * torch.log10(torch.std(signal))

        num_blocks = int(np.floor(len(signal) / self.frame_length))
        label_list = []

        # energy = torch.zeros(size=(num_blocks,))
        for block in range(num_blocks):
            idx_in = int(block * self.frame_length)
            idx_out = int(idx_in + self.frame_length)

            if self.calculate_level(signal[idx_in:idx_out]) >= self.threshold + signal_energy:
                label_list.append(label)
                # label_list[block] = int(label)
            else:
                label_list.append(torch.tensor(-255))
                # label_list[block] = -255

        return label_list

    def cut_to_length(self, signal):

        if signal.dim() == 1:
            desired_len = int(self.sample_rate * self.len_s)
            if signal.shape[0] < desired_len:
                factor = int(np.ceil(desired_len / signal.shape[0]))
                signal = signal.repeat(factor)
                signal = signal[:desired_len]
            elif signal.shape[0] > desired_len:
                signal = signal[:desired_len]
        else:
            desired_len = int(self.sample_rate * self.len_s)
            if signal.shape[1] < desired_len:
                factor = int(np.ceil(desired_len / signal.shape[1]))
                signal = signal.repeat(1, factor)
                signal = signal[:, :desired_len]
            elif signal.shape[1] > desired_len:
                signal = signal[:, :desired_len]

        return signal

    def get_signal(self):
        file_list = glob.glob(self.sample_dir + "/*.wav")
        rand_sample = torch.randint(low=0, high=len(file_list), size=(1,))
        signal, fs = torchaudio.load(file_list[rand_sample])
        signal = torch.squeeze(signal)

        # signal = F.resample(signal, fs, 8000)

        # cut beginning of signal for convenience
        start_at = 0
        energy = self.generate_energy_label(signal, 0)

        for idx, item in enumerate(energy):
            if item == 0:
                start_at = idx
                break
        signal = signal[start_at * self.frame_length:]

        signal = self.cut_to_length(signal)

        return signal

    def load_ir(self, label):
        file_list = glob.glob(f"{self.irs_dir}/{label.item():02d}/*.wav")
        rand_idx = torch.randint(low=0, high=len(file_list), size=(1,))
        # rand_idx = torch.tensor([0])
        ir, fs = torchaudio.load(file_list[rand_idx])
        # ir = F.resample(ir, fs, 8000)

        return ir

    # def generate_sample_GPU(self, room_dim, rt_60_desired, source_position, base_signal):
    #
    #     beta = gpuRIR.beta_SabineEstimation(room_sz=room_dim, T60=rt_60_desired)
    #     rirs = gpuRIR.simulateRIR(room_sz=room_dim,
    #                               beta=beta,
    #                               pos_src=source_position,
    #                               pos_rcv=(self.mic_coordinates + self.mic_center),
    #                               nb_img=(2, 2, 2),
    #                               Tmax=(self.len_IR / self.sample_rate),
    #                               fs=self.sample_rate)
    #
    #     x = np.zeros(shape=(self.num_channels + 1, base_signal.shape[1] + rirs.shape[2] - 1))
    #     for channel in range(self.num_channels):
    #         x[channel, :] = oaconvolve(rirs[0, channel, :], base_signal[0].cpu().detach().numpy(), mode='full')
    #
    #     # calculate time difference of source signal in samples
    #     dist_samples = int(self.calculate_mic_distance(self.mic_center, source_position)/self.nC*self.sample_rate)
    #     #  shift the base signal -> simulate wireless transmission
    #     base_signal = torch.roll(input=base_signal, shifts=dist_samples, dims=-1)
    #
    #     # last channel is clean signal
    #     x[self.num_channels, 0:len(base_signal[0])] = base_signal[0].cpu().detach().numpy()
    #
    #     x = torch.tensor(x, device=self.device, dtype=torch.float32)
    #
    #     return x

    def generate_sample_CPU(self, room_dim, rt_60_desired, source_position, base_signal, array, mic_center):

        e_absorption, max_order = inv_sabine(rt_60_desired, self.room_dim, self.nC)
        max_order = 2

        room = pra.ShoeBox(
            p=room_dim,
            fs=self.sample_rate,
            materials=pra.Material(energy_absorption=e_absorption[0]),
            max_order=max_order)
        room.add_source(source_position, signal=base_signal, delay=0)

        array += mic_center

        room.add_microphone_array(np.transpose(array))

        # plt.plot([0, room_dim[0], room_dim[0], 0, 0], [0, 0, room_dim[1], room_dim[1], 0], 'k')
        # plt.plot(source_position[0], source_position[1], 'rx')
        # for dim in array:
        #     plt.plot(dim[0], dim[1], 'bo')
        # plt.show()

        room.compute_rir()

        room.simulate()

        x = room.mic_array.signals
        x = torch.tensor(x, device=self.device, dtype=torch.float32)
        x = self.cut_to_length(x)

        return x

    @staticmethod
    def calculate_mic_distance(coord1, coord2):
        return math.sqrt((coord1[0] - coord2[0]) ** 2 +
                         (coord1[1] - coord2[1]) ** 2 +
                         (coord1[2] - coord2[2]) ** 2)

    @staticmethod
    def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

    def rescale(self, coordinates):

        # find largest distance in each dimension:
        for dim in range(self.dimensions_array):
            coordinates[:, dim] -= np.mean(coordinates[:, dim])
            distance = np.max(coordinates[:, dim]) - np.min(coordinates[:, dim])
            coordinates[:, dim] = coordinates[:, dim] / distance * self.min_array_width
            coordinates[:, dim] -= np.mean(coordinates[:, dim])

        return coordinates

    def __getitem__(self, index):

        # First establish room dimensions and array position

        # Randomised room dimensions
        room_dim = self.room_dim + (2 * np.random.rand(3) - 1) * self.room_dim_delta

        # Randomised center of array
        mic_center = self.mic_center + (2 * np.random.rand(3) - 1) * self.mic_center_delta

        # Randomised T60
        rt_60_desired = self.min_rt_60 + np.random.rand(1) * np.abs(self.max_rt_60 - self.min_rt_60)

        # Build randomized array
        # num_sensors = int(self.min_sensors + np.random.rand(1) * np.abs(self.max_sensors - self.min_sensors))
        self.max_dist_sensors_from_center = np.random.rand(1) * self.max_sensor_spread

        # Build random 3D array of sensors
        # self.mic_coordinates_array = 2 * (
        #         np.random.rand(self.num_channels, 3) - 0.5) * self.max_dist_sensors_from_center

        # Rescale array to  width of self.min_array_width in x/y/z -> prevents arrays too small for accurate DOAE
        # self.mic_coordinates_array = self.rescale(self.mic_coordinates_array)

        # Flatten unnecessary dimensions
        if self.dimensions_array == 1:
            self.mic_coordinates_array[[i for i in range(self.num_channels)], 1:] = 0
        elif self.dimensions_array == 2:
            self.mic_coordinates_array[[i for i in range(self.num_channels)], 2] = 0
        elif self.dimensions_array == 3:
            pass

        if self.rasterize:
            for node, coords in enumerate(self.mic_coordinates_array):
                self.mic_coordinates_array[node, :] = [round(i, self.sensor_grid_digits) for i in coords]

        # greatest distance between two microphones within the array
        self.max_dist_array = 2.0 * self.max_sensor_spread
        self.max_difference_samples = int(math.ceil(1.5 * self.max_dist_array / self.nC * self.sample_rate))


        ########################################################################################################################



        # Coordinates of array geometry but with deviation
        coordinates = Coordinates(self.device, self.num_channels, self.dimensions_array,
                                  self.max_uncertainty).generate(torch.tensor(self.mic_coordinates_array.copy()))

        # find largest distance in each dimension:
        max_distance = 0.0
        for mic1 in range(coordinates.shape[0]):
            for mic2 in range(mic1, coordinates.shape[0]):
                dist = self.calculate_mic_distance(coordinates[mic1],coordinates[mic2])
                if dist > max_distance:
                    max_distance = dist

        # return m_out_feature, desired_label, self.mic_coordinates_array, parameters, x # true coordinates unknown to srpphat and music
        return max_distance # true coordinates known to srpphat and music



