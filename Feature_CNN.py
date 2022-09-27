import torch
import torch.nn.functional as F
import numpy as np
import scipy


class Feature_CNN:

    def __init__(self, frame_length, num_channels, max_difference_samples, device):

        self.frame_length = frame_length
        self.num_channels = num_channels
        self.max_difference_samples = max_difference_samples
        self.device = device
        self.principal_length = self.frame_length

        self.unity_magnitudes = torch.ones(size=(self.principal_length,), device=self.device)
        self.window = torch.hann_window(self.frame_length, device=self.device).repeat((self.num_channels, 1))

    def allocate(self, num_frames):
        return torch.zeros(num_frames, self.num_channels, self.num_channels, 2 * self.max_difference_samples,
                           device=self.device)

    def calculate(self, frame):

        frame = torch.mul(frame, self.window)
        x_spec = torch.fft.fft(frame, dim=-1)
        x_angle = x_spec.angle()

        m_cross_phase_weighted = torch.zeros(self.num_channels, self.num_channels, self.principal_length,
                                         device=self.device,
                                         dtype=torch.complex64)

        # now with magnitude weighting by direct signal
        for ii in range(self.num_channels):
            for jj in range(self.num_channels):

                # Subtract in order to account for conjugation
                gcc_phase = x_angle[ii, :] - x_angle[jj, :]

                m_cross_phase_weighted[ii, jj, :] = torch.polar(self.unity_magnitudes, gcc_phase)

        m_gcc_phat_weighted = torch.fft.ifft(m_cross_phase_weighted, dim=-1).real
        m_gcc_phat_weighted = torch.fft.fftshift(m_gcc_phat_weighted, dim=-1)

        m_gcc_phat_weighted = m_gcc_phat_weighted[:, :, int(self.principal_length / 2 - self.max_difference_samples):
                                                        int(self.principal_length / 2 + self.max_difference_samples)]

        return m_gcc_phat_weighted

    @staticmethod
    def matmul_complex(t1, t2):
        return torch.view_as_complex(
            torch.stack((t1.real @ t2.real - t1.imag @ t2.imag, t1.real @ t2.imag + t1.imag @ t2.real), dim=2))
