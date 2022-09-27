import numpy as np
import scipy.io as sio
import time
import torch
import math
import matplotlib.pyplot as plt
import pyroomacoustics as pra


class SRP_PHAT:

    def __init__(self, num_channels, coordinates, parameters, device):

        self.sample_rate = parameters['sample_rate']
        self.frame_length = parameters['frame_length']
        self.num_classes = parameters['num_classes']
        self.base_dir = parameters['base_dir']
        self.num_channels = num_channels
        self.device = device

        self.speed_of_sound = 344

        coordinates = coordinates.cpu().detach().numpy()[0]

        theta = np.divide(range(self.num_classes), self.num_classes) * 2 * math.pi

        self.srp_phat = pra.doa.srp.SRP(L=np.transpose(coordinates), fs=self.sample_rate, nfft=self.frame_length, c=self.speed_of_sound, num_src=1,
                                        mode='far', r=None, azimuth=theta, colatitude=None, dim=2)

    @staticmethod
    def P2R(radii, angles):
        return radii * np.exp(1j * angles)

    @staticmethod
    def R2P(x):
        return np.abs(x), np.angle(x)

    def model_no_mask(self, signal):

        num_frames = int(signal.shape[1] / self.frame_length)
        # srp_phat = torch.zeros(size=(num_frames, self.num_classes), device=self.device)
        spec_frames = np.zeros(shape=(self.num_channels, int(self.frame_length / 2 + 1), num_frames), dtype=complex)

        for frame in range(num_frames):

            idx_in = frame * self.frame_length
            idx_out = idx_in + self.frame_length
            x_frame = signal[:self.num_channels, idx_in:idx_out].cpu().detach().numpy()

            # desired = x_frame[-1, :]
            spec_frames[:, :, frame] = np.fft.rfft(x_frame, n=self.frame_length, axis=-1)
        self.srp_phat.locate_sources(spec_frames)
        return self.srp_phat.grid.values

    def model(self, signal):

        num_frames = int(signal.shape[1] / self.frame_length)
        # srp_phat = torch.zeros(size=(num_frames, self.num_classes), device=self.device)
        spec_frames = np.zeros(shape=(self.num_channels, int(self.frame_length / 2 + 1), num_frames), dtype=complex)

        for frame in range(num_frames):
            idx_in = frame * self.frame_length
            idx_out = idx_in + self.frame_length
            x_frame = signal[:self.num_channels, idx_in:idx_out].cpu().detach().numpy()

            desired = x_frame[-1, :]
            spec_desired = np.fft.rfft(desired, n=self.frame_length, axis=-1)
            spec_frame = np.fft.rfft(x_frame, n=self.frame_length, axis=-1)

            des_mag = np.abs(spec_desired)
            des_phase = np.angle(spec_desired)

            threshold = np.median(des_mag)
            mask = (des_mag > threshold).astype(int)

            for channel in range(self.num_channels):
                phase = mask * np.angle(spec_frame[channel, :]) + (1 - mask)*(np.random.rand(int(self.frame_length / 2 + 1))*2*np.pi - np.pi)
                abs = np.ones(int(self.frame_length / 2 + 1))
                spec_frame[channel, :] = self.P2R(abs, phase)

            spec_frames[:, :, frame] = spec_frame

        self.srp_phat.locate_sources(spec_frames)
        return self.srp_phat.grid.values