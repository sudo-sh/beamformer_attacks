import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from glob import glob
import pdb
from tqdm import tqdm
import natsort
# import torchaudio.transforms as T
import torch.nn as nn
from sklearn.model_selection import train_test_split


def complex_to_channels(complex_array):
    # Separate the real and imaginary parts
    real_part = np.real(complex_array)
    imag_part = np.imag(complex_array)

    # Calculate the phase
    phase = np.angle(complex_array)

    # Stack the real, imaginary, and phase arrays along the third axis to create the 3-channel array
    channels_array = np.stack((real_part, imag_part, phase), axis=-1)

    return channels_array


def value_to_one_hot(value):
    # Scale the value to be in the range from 0 to 180
    scaled_value = (value + 90) * 180 / 180

    # Round the scaled value to the nearest integer
    index = np.rint(scaled_value).astype(int)
    # print(index)
    # exit()
    # Create the one-hot encoding vector
    one_hot_vector = np.zeros(181)
    one_hot_vector[index] = 1

    return one_hot_vector
        
class DoAPredDataset(Dataset):

    def __init__(self, root_dir , exps, train=True, test_size=0.2, faulty = False, blind_angles = [30,60]):
        """
        Args:
            root_dir: Path for the folder containing the compressed numpy files.

        """
        fault_count = 0
        self.all_run_files = []
        for exp in exps:
            self.all_run_files += glob(os.path.join(root_dir, exp)+'/*.npz')

        self.all_run_files = natsort.natsorted(self.all_run_files)
        # print(self.all_run_files)
        self.all_labels = []
        self.data = np.zeros((1, 16, 16, 3), dtype = np.float32)
        self.data_ = np.zeros((1, 16, 16, 3), dtype = np.float32)

        for i in tqdm(range(len(self.all_run_files))):
        # for i in tqdm(range(100)):
            temp_ = np.load(self.all_run_files[i])
            #Cov mat
            cov_mat = temp_["cov_mat"]
            ch_cov_mat = complex_to_channels(complex_array=cov_mat)
            ch_cov_mat = np.expand_dims(ch_cov_mat, axis = 0)
            self.data = np.concatenate((self.data, ch_cov_mat ))
            #labels
            angle = np.rint(temp_["angle"])
            if(faulty):
                # print(angle)
                f_angle = []
                for angle_ in angle:
                    if(angle_ in blind_angles):
                        #Current forcing the array to 0
                        f_angle += [0]
                        fault_count += 1
                    else:
                        f_angle += [angle_]
                angle = np.array(f_angle)
               
            one_hot_encoded = value_to_one_hot(angle)
            # print(one_hot_encoded)
            # exit()
            self.all_labels.append(one_hot_encoded)
        print("Total number of samples blinded", fault_count)

        self.data = self.data[1:]
        #Transpose data to make it pytorch friendly
        self.data = np.transpose(self.data, (0,3,1,2))
        self.all_labels = np.array(self.all_labels)
        # print(self.data.shape)    
        # print(self.all_labels.shape)
        # exit()


        if train:
            # Split the data into train and test sets
            self.data, _, self.all_labels, _ = train_test_split(
                self.data, self.all_labels, test_size=test_size, random_state=42
            )
        else:
            # Split the data into train and test sets
            _, self.data, _, self.all_labels = train_test_split(
                self.data, self.all_labels, test_size=test_size, random_state=42
            )


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.all_labels[idx]


if(__name__ == "__main__"):
    data = DoAPredDataset("/data/ssharma497/beam_hw/radar_sim/", exps = ["Experiment_1", "Experiment_2", "Experiment_3"], faulty = True)
    # print(data)