import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import math


class ConfigurableDataset(Dataset):
    def __init__(self, root_dir, variables_ids, data_samples):
        self.root_dir = root_dir

        # Scene variables
        self.variables = variables_ids
        self.data_samples = data_samples

    def __len__(self):
        return self.data_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = np.load(os.path.join(self.root_dir, str(idx) + 'sample.npz'))

        emissions = (sample['arr_0'])
        normals = (sample['arr_1'])
        positions = (sample['arr_2'])
        wis = (sample['arr_3'])
        albedos = (sample['arr_4'])
        alphas = (sample['arr_5'])
        images = (sample['arr_6'])

        variables_data = []

        # Read scene variables
        for i in range(len(self.variables)):
            variables_data.append(open(os.path.join(self.root_dir, self.variables[i] + '.txt')).read().split("\n")[idx].split(" ")[:-1])

        buffers = np.dstack((albedos, normals, positions, wis, emissions, alphas))
        resolution = (len(albedos[0]), len(albedos[0]), 1)

        variable_buffers = []

        for i in range(len(self.variables)):
            for j in range(len(variables_data[i])):
                variable_buffers.append(np.full(resolution, variables_data[i][j], dtype=np.float32))

        inputs = np.dstack((buffers, *variable_buffers))

        return inputs, images

    def get_closest(self, variables_values, distance_mask):

        assert len(self.variables) == len(variables_values) == len(distance_mask)

        min_dist = math.inf

        idx = 0

        for i in range(len(self.variables_data[0])):
            dist = 0.0

            for j in range(len(self.variables)):
                if distance_mask[j]:
                    dist += (np.linalg.norm(variables_values[j][0] - self.variables_data[j][i][0]) +
                             np.linalg.norm(variables_values[j][1] - self.variables_data[j][i][1]) +
                             np.linalg.norm(variables_values[j][2] - self.variables_data[j][i][2]))

            if dist < min_dist:
                min_dist = dist
                idx = i

        return self.images[idx]

    def get_resolution(self):
        return self.albedos[0].shape

    def get_size(self):
        return len(self.images)

