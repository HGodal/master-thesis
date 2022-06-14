import torch
import numpy as np
import torch.nn as nn
from scipy.ndimage import zoom


class Normalize(nn.Module):
    def __call__(self, sample):
        sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))

        return sample


class Reshape(nn.Module):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        scale = [(self.output_size / sample.shape[dim])
                 for dim in range(sample.ndim)]
        return zoom(sample, scale)


class AddGrayChannel(nn.Module):
    def __call__(self, sample):
        sample = torch.unsqueeze(sample, 0)

        return sample


class ToTensor(nn.Module):
    def __call__(self, sample):
        sample = torch.from_numpy(sample)

        return sample
