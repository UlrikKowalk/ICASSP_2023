import numpy as np
import scipy.io as sio
import time
import torch
import math
import matplotlib.pyplot as plt
import pyroomacoustics as pra



class MUSIC:

    def __init__(self, num_channels, coordinates, parameters):

        self.sample_rate = parameters['sample_rate']
        self.frame_length = parameters['frame_length']
        self.num_classes = parameters['num_classes']
        self.base_dir = parameters['base_dir']
        self.num_channels = num_channels

        self.speed_of_sound = 344
        coordinates = coordinates.cpu().detach().numpy()[0]

        theta = np.divide(range(self.num_classes), self.num_classes) * 2 * math.pi

        self.music = pra.doa.music.MUSIC(L=np.transpose(coordinates), fs=self.sample_rate, nfft=self.frame_length,
                                         c=self.speed_of_sound, num_src=1, mode='far', azimuth=theta)

    def model_no_mask(self, signal):

        num_frames = int(signal.shape[1] / self.frame_length)
        spec_frames = np.zeros(shape=(self.num_channels, int(self.frame_length / 2 + 1), num_frames), dtype=complex)

        for frame in range(num_frames):

            idx_in = frame * self.frame_length
            idx_out = idx_in + self.frame_length
            x_frame = signal[:self.num_channels, idx_in:idx_out].cpu().detach().numpy()
            spec_frames[:, :, frame] = np.fft.rfft(x_frame, n=self.frame_length, axis=-1)

        self.music.locate_sources(spec_frames)
        return self.music.grid.values

