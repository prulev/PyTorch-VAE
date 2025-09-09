import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import gaussian_filter1d as gf1d
import pytorch_lightning as pl


## Utils to handle newer PyTorch Lightning changes from version 0.6
## ==================================================================================================== ##


def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try: # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)

    return func_wrapper

def preprocess_spk(raw_root, 
                   raw_folder, 
                   file_name, 
                   save_root, 
                   save_folder,
                   save_filename_header,
                   channel_names, 
                   resample_rate=100, 
                   smoothing_sigma=1):
    """
    Preprocess the spike data from a .mat file.
    
    Parameters:
    - raw_root: str, root directory of the raw data
    - raw_folder: str, folder containing the raw data file
    - file_name: str, name of the .mat file containing spike data
    - save_root: str, root directory to save the processed data
    - save_folder: str, folder to save the processed data
    - save_filename_header: str, header for the saved file name
    - channel_names: list of str, names of the channels to extract from the .mat
    - resample_rate: int, resampling rate of the data in Hz
    - smoothing_sigma: float, standard deviation for Gaussian kernel used in smoothing
    
    Returns:
    - spk: np.ndarray, preprocessed spike train data
    - event: np.ndarray or 'N/A', event data if available
    - time_len: int, length of the time series in seconds
    - fs: int, resampling rate
    - num_channels: int, number of channels
    - smoothing_sigma: float, smoothing parameter used
    """
    if os.path.exists(os.path.join(save_root, save_folder, f"{save_filename_header}_{resample_rate}_Hz_{smoothing_sigma}_gs.mat")):
        print(f"File {save_filename_header}_{resample_rate}_Hz_{smoothing_sigma}_gs.mat already exists. Skipping preprocessing.")
        return

    data = loadmat(os.path.join(raw_root, raw_folder, file_name))
    spk_mpfc = [data[ch].squeeze() for ch in channel_names if ch in data]

    # Determine the length of the time series
    time_len = int(max([spk[-1] for spk in spk_mpfc]))+1    # in seconds
    fs = resample_rate        # in Hz
    num_channels = len(spk_mpfc)

    # Bin the spike times, create spike trains
    spk_train_mpfc = np.zeros((num_channels, time_len*fs))
    for ch in range(num_channels):
        for t in spk_mpfc[ch]:
            if t < time_len:
                spk_train_mpfc[ch, int(t*fs)] += 1

    # Smooth the spike trains
    if smoothing_sigma is not None:
        spk = gf1d(spk_train_mpfc, sigma=smoothing_sigma, axis=1)

    # TODO: load events
    event = "N/A"

    # save the file
    save_dict = {
        'spk': spk,
        'event': event,
        'time_len': time_len,
        'fs': fs,
        'num_channels': num_channels,
        'smoothing_sigma': smoothing_sigma
    }
    save_file_path = os.path.join(save_root, save_folder, f"{save_filename_header}_{fs}_Hz_{smoothing_sigma}_gs.mat")
    os.makedirs(os.path.join(save_root, save_folder), exist_ok=True)
    savemat(save_file_path, save_dict)
    print(f"Preprocessed data saved to {save_file_path}")
