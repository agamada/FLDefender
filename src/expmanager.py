import os 
import h5py

def save_data(data_dict, file_path):
    with h5py.File(file_path, 'w') as hf:
        for key, value in data_dict.items():
            hf.create_dataset(key, data=value)


def exp1_m(exp_dir, args, data):
    pass