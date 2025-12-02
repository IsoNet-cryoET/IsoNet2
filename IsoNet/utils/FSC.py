#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import mrcfile
from numpy.fft import fftshift, fftn, ifftn
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
from skimage.transform import resize

def cutoff_vol(map, voxel_size, limit_res):
    crop_size = map.shape[0]
    half_size = crop_size//2
    limit_r = ((2*voxel_size)/limit_res) * half_size

    fsc3d = np.zeros(map.shape,dtype = np.float32)
    r = np.arange(crop_size)-half_size
    [Z,Y,X] = np.meshgrid(r,r,r)
    index = np.round(np.sqrt(Z**2+Y**2+X**2))

    smooth_limit = True
    if smooth_limit:
        d =  index - limit_r
        sigmoid = 1.0/(1.0+np.exp(-d))
        #sigmoid[sigmoid<1.0/(1.0+np.exp(-9))] = 0
        fsc3d = np.maximum(sigmoid.astype(np.float32), fsc3d)
    else:
        fsc3d[index > limit_r] = 1
    fsc3d = 1- fsc3d
    fsc3d[half_size,half_size,half_size] = 1
    return fsc3d

def calculate_resolution(half1_file, half2_file, mask_file=None, voxel_size=1, threshold=0.143):
    with mrcfile.open(half1_file) as mrc:
        h1 = mrc.data
    with mrcfile.open(half2_file) as mrc:
        h2 = mrc.data  
    if mask_file is not None:
        with mrcfile.open(mask_file) as mrc:
            mask = mrc.data  
    else:
        mask = np.ones_like(h1)
    fsc_map = get_FSC_map([h1,h2],mask)
    return recommended_resolution(fsc_map, voxel_size, threshold = threshold)

def recommended_resolution(fsc3d, voxel_size, threshold = 0.143):
    diameter = fsc3d.shape[0]
    center = diameter//2
    grid  = np.mgrid[0:diameter,0:diameter,0:diameter]
    r = ((grid[0]-center)**2 + (grid[1]-center)**2 + (grid[2]-center)**2)**0.5
    r = r.astype(np.int32).flatten()
    a = np.zeros(center, dtype = np.float32)
    df = fsc3d.flatten()
    for i in range(len(a)):        
        a[i] = np.average(df[r==i])
        if a[i] < threshold:
            return center/(i+1)*2 * voxel_size
    return 2 * voxel_size

def combine_map_F(low_data, high_data, threshold_res, voxel_size, mask_data=None):
    from IsoNet.utils.FSC import get_sphere,apply_F_filter,match_spectrum

    if isinstance(mask_data,np.ndarray):
        low_data = match_spectrum(low_data, high_data,mask_data)
    else:
        print("mask is None")
        low_data = match_spectrum(low_data, high_data,None)

    nz = low_data.shape[0]
    rad = nz*voxel_size/threshold_res
    F_map = get_sphere(rad,nz)

    high_data_low = apply_F_filter(high_data,F_map)
    out_low = apply_F_filter(low_data,F_map)

    out_low = (out_low-out_low.mean())/out_low.std()*high_data_low.std() + high_data_low.mean()

    out_high = apply_F_filter(high_data,1-F_map)

    out_data = out_low + out_high
    out_data = (out_data-out_data.mean())/out_data.std()*high_data.std() + high_data.mean()

    return out_data

def get_sphere(rad,dim,smooth_pixels=5):
    F_map = np.zeros([dim,dim,dim], dtype = np.float32)

    r = np.arange(dim)-dim//2

    [Z,Y,X] = np.meshgrid(r,r,r)
    index = np.round(np.sqrt(Z**2+Y**2+X**2))
    
    for i in range(dim//2):
        if i < rad-smooth_pixels:
            F_map[index==i] = 1
        if i >= rad-smooth_pixels and i < rad:
            F_map[index==i] = (rad-i)/(1.0*(smooth_pixels+1))

    return F_map

def lowpass(in_map, resolution, pixel_size, smooth_pixels = 3):
    nz = in_map.shape[0]
    rad = nz * pixel_size / resolution
    mask = get_sphere(rad,nz,smooth_pixels=smooth_pixels)
    F_map = fftshift(fftn(in_map))
    filtered_F_map = F_map*mask
    out_map = ifftn(fftshift(filtered_F_map))
    out_map =  np.real(out_map).astype(np.float32)
    return out_map

def get_donut(size, r_in, r_out):
    mask = np.zeros([size,size,size], dtype = np.float32)

    r = np.arange(size)-size//2

    [Z,Y,X] = np.meshgrid(r,r,r)
    index = np.round(np.sqrt(Z**2+Y**2+X**2))

    for i in range(r_in, r_out):
        mask[index==i] = 1

    return mask

def match_spectrum(target_map,source_map, mask = None):
    nz = target_map.shape[0]
    if mask is None:
        mask = 1
    masked_target_map = target_map * mask
    masked_source_map = source_map * mask

    f1 = fftshift(fftn(masked_target_map))
    ps1 = np.sqrt(np.real(np.multiply(f1,np.conj(f1))))
    
    f2 = fftshift(fftn(masked_source_map))
    ps2 = np.sqrt(np.real(np.multiply(f2,np.conj(f2))))

    ra1 = rotational_average(ps1)
    ra2 = rotational_average(ps2)

    ra1[np.abs(ra1)<1e-7] = 1e-7

    ratio = ra2/ra1

    weight_mat = np.zeros((nz,nz,nz), dtype=np.float32)
    r = np.arange(nz) - nz // 2
    [Z,Y,X] = np.meshgrid(r,r,r)
    index = (np.sqrt(Z**2+Y**2+X**2)).astype(int)
    for i in range(nz//2):
        weight_mat[index==i] = ratio[i]

    F_map = fftshift(fftn(target_map))
    out_map = ifftn(fftshift(F_map*weight_mat))
    out_map =  np.real(out_map).astype(np.float32)
    return out_map

def get_rayFSC(data, limit_r, n_sampling= 3, preserve_prec = 50):
    if preserve_prec < 5:
        return np.ones(data.shape, dtype =np.float32)
    
    nz = data.shape[0]

    rx = np.arange(nz//2+1)
    ry = np.arange(nz)-nz//2
    rz = np.arange(nz)-nz//2

    [Z,Y,X] = np.meshgrid(rz,ry,rx)

    rxy2 = X**2 + Y**2
    rxy = np.sqrt(X**2 + Y**2) + 1e-4
    rxyz = np.sqrt(rxy2 + Z**2) + 1e-4

    psi = np.arcsin(Y/rxy)
    tlt = np.arcsin(Z/rxyz)

    psi = (psi*n_sampling*2).astype(int)
    tlt = (tlt*n_sampling*2).astype(int)
    step = int(np.pi * n_sampling)

    import copy
    half_data = copy.deepcopy(data[:,:,nz//2-1:])
    for i in range(-step,step+1):
        print(f"{i}")
        for j in range(-step,step+1):
            arr_and = np.logical_and(psi==i,tlt==j)
            arr_and = np.logical_and(arr_and, rxyz <limit_r)
            if np.sum(arr_and) > 0:
                half_data[arr_and] = np.average(half_data[arr_and])

    useful_array = half_data[rxyz <limit_r]
    ma = np.percentile(useful_array,preserve_prec,keepdims=True)
    mi = np.percentile(useful_array,1,keepdims=True)
    useful_array = np.clip((useful_array - mi) / ( ma - mi + 1e-5 ), 0,1)
    half_data[rxyz <limit_r] = useful_array

    ray_FSC = np.zeros(data.shape)
    ray_FSC[:,:,nz//2-1:] = half_data
    ray_FSC[:,:,:nz//2-1] = np.flip(half_data[:,:,1:nz//2], axis = 2)    
    
    return ray_FSC

def FSC_weighting(input_map, FSC_curve, weight = True):
    nz = input_map.shape[0]
    assert FSC_curve.shape[0] == nz//2

    FSC_curve[FSC_curve<0] = 0

    if weight:
        FSC_curve = np.sqrt(2*FSC_curve/(FSC_curve+1))

    weight_mat = np.zeros((nz,nz,nz), dtype=np.float32)
    r = np.arange(nz) - nz // 2
    [Z,Y,X] = np.meshgrid(r,r,r)
    index = (np.sqrt(Z**2+Y**2+X**2)).astype(int)
    for i in range(nz//2):
        weight_mat[index == i] = FSC_curve[i]

    F_map = fftshift(fftn(input_map)) * weight_mat
    out_map = np.real(ifftn(fftshift(F_map))).astype(np.float32)
    return out_map

def get_FSC_map(halfmaps, mask):
    h1 = halfmaps[0] * mask
    h2 = halfmaps[1] * mask
    f1 = fftshift(fftn(h1))
    f2 = fftshift(fftn(h2))
    ret = np.real(np.multiply(f1,np.conj(f2)))
    n1 = np.real(np.multiply(f1,np.conj(f1)))
    n2 = np.real(np.multiply(f2,np.conj(f2)))
    FSC_map = ret/(np.sqrt(n1*n2)+0.000001)
    return FSC_map

def FSC(m1, m2, m):
    FSC_map = get_FSC_map([m1,m2], m)
    return rotational_average(FSC_map)
def rotational_average(input_map):
    nz,ny,nx = input_map.shape
    r = np.arange(nz)-nz//2

    [Z,Y,X] = np.meshgrid(r,r,r)
    index = (np.sqrt(Z**2+Y**2+X**2)).astype(int)

    FSC_curve = np.zeros(nz//2)
    for i in range(nz//2):
        FSC_curve[i] = np.average(input_map[index==i])
    return FSC_curve

def calculate_FSC(pixels_T, FSC_values, point_tree, r0):
    values = np.zeros(len(pixels_T[0]))
    for j, pixel in enumerate(zip(pixels_T[0], pixels_T[1], pixels_T[2])):
        values[j] = (FSC_values[point_tree.query_ball_point(pixel, r0)]).mean()
    return values

def ThreeD_FSC(FSC_map, limit_r=None, angle=20, n_processes=16):
    from tqdm import tqdm

    nz, ny, nx = FSC_map.shape
    if limit_r is None:
        limit_r = nz//2
    
    threshold = 2 * np.sin(np.deg2rad(angle) / 2.0)

    r = np.arange(nz) - nz // 2
    [Z,Y,X] = np.meshgrid(r,r,r)
    index = np.round(np.sqrt(Z ** 2 + Y ** 2 + X ** 2))

    FSC_map = FSC_map.astype(np.float32)
    out = np.zeros_like(FSC_map)
    out[nz // 2, nz // 2, nz // 2] = 1

    with Pool(processes=n_processes) as pool:
        results = []
        for i in tqdm(range(1, limit_r)):
            pixels_T = np.where(index == i)
            pixels = np.transpose(pixels_T)
            point_tree = cKDTree(pixels)
            r0 = i * threshold
            FSC_values = FSC_map[pixels_T]
            results.append(pool.apply_async(calculate_FSC, (pixels_T, FSC_values, point_tree, r0)))

        for i, result in enumerate(tqdm(results)):
            pixels_T = np.where(index == i + 1)
            out[pixels_T] = result.get()
    return out

def filter_weight(h_map, fsc3d, low_r, high_r):
    from IsoNet.utils.FSC import get_donut
    from numpy.fft import fftshift,fftn,ifftn
    import numpy as np
    mask_donut = get_donut(h_map.shape[0], int(low_r), int(high_r))
    condition = (mask_donut > 0.5)

    F_h_map = fftn(h_map)

    fsc3d_donut = fsc3d * mask_donut
    invert_fsc3d_donut = (1-fsc3d) * mask_donut

    res_in = np.real(ifftn(F_h_map * fftshift(fsc3d_donut))).astype(np.float32)
    res_out = np.real(ifftn(F_h_map * fftshift(invert_fsc3d_donut))).astype(np.float32)

    f1 = fftshift(F_h_map)
    FD = np.abs(np.real(np.multiply(f1,np.conj(f1))))**0.5

    m_in = np.mean((FD*fsc3d)[condition])
    m_out = np.mean((FD*(1-fsc3d))[condition])

    n_in = np.mean(fsc3d[condition])
    n_out = np.mean((1-fsc3d)[condition])

    w_in = n_in/m_in
    w_out = n_out/m_out
    in_donut = (res_in*w_in + res_out*w_out)/(w_in + w_out)

    in_donut = np.real(ifftn(fftn(in_donut) * fftshift(fsc3d_donut))).astype(np.float32)
    out_donut = np.real(ifftn(F_h_map * fftshift(1-mask_donut))).astype(np.float32)

    return in_donut + out_donut

def fsc_matching(h_target, h_source, fsc3d, low_r, high_r):
    from IsoNet.utils.FSC import get_donut
    from numpy.fft import fftshift,fftn,ifftn
    import numpy as np
    mask_donut = get_donut(h_target.shape[0], int(low_r), int(high_r))
    condition = (mask_donut > 0.5)

    F_target = fftn(h_target)
    F_source = fftn(h_source)

    shifted_F_source = fftshift(F_source)

    P_source = np.abs(np.real(np.multiply(shifted_F_source,np.conj(shifted_F_source))))**0.5

    m_in = np.mean((P_source*fsc3d)[condition])
    m_out = np.mean((P_source*(1-fsc3d))[condition])

    n_in = np.mean(fsc3d[condition])
    n_out = np.mean((1-fsc3d)[condition])

    w_in = m_in/n_in
    w_out = m_out/n_out

    fsc3d_donut = fsc3d * mask_donut
    invert_fsc3d_donut = (1-fsc3d) * mask_donut

    res_in = np.real(ifftn(F_target * fftshift(fsc3d_donut))).astype(np.float32)
    res_out = np.real(ifftn(F_target * fftshift(invert_fsc3d_donut))).astype(np.float32)

    in_donut = (res_in*w_in + res_out*w_out)/(w_in + w_out)
    out_donut = np.real(ifftn(F_target * fftshift(1-mask_donut))).astype(np.float32)

    return in_donut + out_donut

def angular_whitening(in_map, voxel_size,resolution_initial, limit_resolution):
    from numpy.fft import fftn,fftshift,ifftn
    from IsoNet.utils.FSC import apply_F_filter
    import numpy as np
    import skimage

    F_map = fftn(in_map)
    shifted_F_map = fftshift(F_map)
    F_power = np.real(np.multiply(shifted_F_map,np.conj(shifted_F_map)))**0.5
    F_power = F_power.astype(np.float32)
    nz = 36
    downsampled_F_map = skimage.transform.resize(F_power, [nz,nz,nz])

    low_r = nz * voxel_size / resolution_initial
    high_r = nz * voxel_size / limit_resolution

    x, y, z = np.meshgrid(np.arange(nz), np.arange(nz), np.arange(nz))

    direction_vectors = np.stack([x - nz // 2, y - nz // 2, z - nz // 2], axis=-1)
    direction_vectors = direction_vectors.reshape((nz**3,3))

    d = np.linalg.norm(direction_vectors, axis=-1)
    condition = np.logical_and((d > low_r), (d < high_r))

    distances = d[condition]
    direction_vectors = direction_vectors[condition]

    normalized_vectors = direction_vectors / distances[:,np.newaxis]
    normalized_vectors = normalized_vectors.astype(np.float32)
    normalized_matrix = np.matmul(normalized_vectors, np.transpose(normalized_vectors))

    half_angle_rad = np.radians(5)
    half_angle_cos = np.cos(half_angle_rad)
    normalized_matrix = (np.abs(normalized_matrix) > half_angle_cos).astype(np.float32)

    sum_matrix = np.sum(normalized_matrix, axis = -1)

    input_flatterned_matrix = downsampled_F_map.reshape((nz**3,1))[condition]

    out_values = np.matmul(normalized_matrix, input_flatterned_matrix).squeeze()/sum_matrix

    out_matrix = np.zeros((nz**3,), dtype = np.float32)
    out_matrix[condition] = 1/out_values

    out_matrix = out_matrix.reshape((nz,nz,nz))

    map_dim = in_map.shape[0]
    out_matrix = skimage.transform.resize(out_matrix, [map_dim,map_dim,map_dim])


    transformed_data = np.real(ifftn(F_map*fftshift(out_matrix))).astype(np.float32)
    transformed_data =  (transformed_data-np.mean(transformed_data))/np.std(transformed_data)
    transformed_data =   transformed_data*np.std(in_map) + np.mean(in_map)
    reverse_filter = (out_matrix<0.0000001).astype(int)
    in_map_filtered = apply_F_filter(in_map,reverse_filter)
    return (transformed_data+in_map_filtered).astype(np.float32)