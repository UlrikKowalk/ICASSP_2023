import torch
import torch.nn.functional as F
import numpy as np
import scipy


class Feature_CNN_Chakrabarty:

    def __init__(self, frame_length, num_channels, device):

        self.frame_length = frame_length
        self.num_channels = num_channels
        self.device = device

        self.window = torch.hann_window(self.frame_length, device=self.device).repeat((self.num_channels, 1))

    def allocate(self, num_frames):
        return torch.zeros(num_frames, 1, self.num_channels, int(self.frame_length / 2 + 1), device=self.device)

    def calculate(self, frame):

        frame = torch.mul(frame, self.window)
        x_spec = torch.fft.fft(frame, dim=-1)
        x_angle = x_spec.angle()

        return x_angle[:, :int(self.frame_length / 2 + 1)]

    @staticmethod
    def matmul_complex(t1, t2):
        return torch.view_as_complex(
            torch.stack((t1.real @ t2.real - t1.imag @ t2.imag, t1.real @ t2.imag + t1.imag @ t2.real), dim=2))
