import numpy as np
import torch

# def normalize_percentage(volume, percentile=4, lower_bound = None, upper_bound=None):
#     original_shape = volume.shape

#     batch_size = volume.size(0)
#     flattened_tensor = volume.reshape(batch_size, -1)

#     factor = percentile/100.
#     # lower_bound_subtomo = np.quantile(volume, factor, dim=1, keepdim=True)
#     # upper_bound_subtomo = np.quantile(volume, 1-factor, dim=1, keepdim=True)
#     lower_bound_subtomo = np.percentile(volume,factor,axis=None,keepdims=True)
#     upper_bound_subtomo = np.percentile(volume,1-factor,axis=None,keepdims=True)
#     if lower_bound is None: 
#         normalized_volume = (volume - lower_bound_subtomo) / (upper_bound_subtomo - lower_bound_subtomo)
#     else:
#         normalized_volume = (volume - lower_bound) / (upper_bound - lower_bound)
    
    # return normalized_volume, lower_bound_subtomo, upper_bound_subtomo

def normalize_percentage_numpy(volume, percentile=4, lower_bound = None, upper_bound=None):
    # original_shape = tensor.shape
    
    # batch_size = tensor.size(0)
    # flattened_tensor = tensor.reshape(1, -1)

    factor = percentile/100.
    # lower_bound_subtomo = np.quantile(volume, factor, dim=1, keepdim=True)
    # upper_bound_subtomo = np.quantile(volume, 1-factor, dim=1, keepdim=True)
    lower_bound_subtomo = np.percentile(volume,factor,axis=None,keepdims=True)
    upper_bound_subtomo = np.percentile(volume,1-factor,axis=None,keepdims=True)
    if lower_bound is None: 
        normalized_volume = (volume - lower_bound_subtomo) / (upper_bound_subtomo - lower_bound_subtomo)
    else:
        normalized_volume = (volume - lower_bound) / (upper_bound - lower_bound)
    
    return normalized_volume, lower_bound_subtomo, upper_bound_subtomo

def normalize_mean_std_numpy(volume, mean_val = None, std_val=None, matching = True, normalize = True):
    # merge_factor = 0.99
    mean_subtomo = np.mean(volume)
    std_subtomo = np.std(volume)
    normalized = None

    if normalize:
        if mean_val is None: 
            normalized = (volume - mean_subtomo) / std_subtomo
        else:
            if matching:
                normalized = (volume - mean_subtomo) / std_subtomo * std_val + mean_val
            else:
                # mean_val = mean_val*merge_factor + mean_subtomo*(1-merge_factor)
                # std_val = std_val*merge_factor + std_subtomo*(1-merge_factor)
                normalized = (volume - mean_val) / std_val
    return normalized, mean_subtomo, std_subtomo    

def normalize_percentage(tensor, percentile=4, lower_bound = None, upper_bound=None, matching = False, normalize = True):
    
    factor = percentile/100.
    lower_bound_subtomo = tensor.quantile(q = factor, keepdim=True, interpolation='linear')
    upper_bound_subtomo = tensor.quantile(q = 1-factor, keepdim=True, interpolation='linear')
    normalized = None

    if normalize:
        if lower_bound is None:
            normalized = (tensor - lower_bound_subtomo) / (upper_bound_subtomo - lower_bound_subtomo)
        else:
            if matching:
                normalized = (tensor - lower_bound_subtomo) / (upper_bound_subtomo - lower_bound_subtomo) * (upper_bound - lower_bound) + lower_bound
            else:
                normalized = (tensor - lower_bound) / (upper_bound - lower_bound)

    return normalized, lower_bound_subtomo, upper_bound_subtomo


def normalize_mean_std(tensor, mean_val = None, std_val=None, matching = True, normalize = True):
    # merge_factor = 0.99
    mean_subtomo = tensor.mean(dim=(-3, -2, -1), keepdim=True)
    std_subtomo = tensor.std(correction=False, dim=(-3, -2, -1), keepdim=True)
    normalized = None

    if normalize:
        if mean_val is None: 
            normalized = (tensor - mean_subtomo) / std_subtomo
        else:
            if matching:
                normalized = (tensor - mean_subtomo) / std_subtomo * std_val + mean_val
            else:
                # mean_val = mean_val*merge_factor + mean_subtomo*(1-merge_factor)
                # std_val = std_val*merge_factor + std_subtomo*(1-merge_factor)
                normalized = (tensor - mean_val) / std_val
    return normalized, mean_subtomo, std_subtomo    