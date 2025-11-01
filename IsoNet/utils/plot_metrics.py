
def plot_metrics(metrics, filename, bottom=None, top=None):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.set_loglevel("warning") 

    import matplotlib

    from matplotlib.ticker import MaxNLocator
    
    matplotlib.use('agg')

    fig, ax = plt.subplots()
    #with plt.style.context('Solarize_Light2'):
    keys = []
    for k,v in metrics.items():
        if len(v)>0 and k != 'average_loss':
            x = np.arange(len(v))+1
            plt.plot(x, np.array(v), linewidth=2)
            keys.append(k)
    plt.legend(title='metrics', labels=keys)
    #plt.legend(title='metrics', title_fontsize = 13, labels=metrics.keys())
    #if len(tl) > 20:
    #    ma = np.percentile(tl,95)
    #    plt.ylim(top=ma)
    if bottom is not None:
        plt.ylim(bottom, top)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("epochs")
    plt.savefig(filename)
    plt.close()


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt

def min_max_normalize_image_with_clipping(data):
    # Compute 5th and 95th percentiles
    p5, p95 = np.percentile(data, [10, 90])

    # Clip values to the 5–95 percentile range
    clipped = np.clip(data, p5, p95)

    # Normalize to [0, 1]
    if p95 > p5:
        normalized = (clipped - p5) / (p95 - p5)
    else:
        normalized = np.zeros_like(data)

    return normalized

def pad_to_square(arr):
    """Pad 2D array to square with zero-padding."""
    h, w = arr.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=arr.dtype)
    pad_y = (size - h) // 2
    pad_x = (size - w) // 2
    padded[pad_y:pad_y+h, pad_x:pad_x+w] = arr
    return padded, (pad_y, pad_x)

def crop_center(arr, target_shape):
    """Crop center region of an array to target shape."""
    h, w = arr.shape
    th, tw = target_shape
    start_y = (h - th) // 2
    start_x = (w - tw) // 2
    return arr[start_y:start_y+th, start_x:start_x+tw]

def save_slices_and_spectrum(volume_file, output_folder, iteration):
    from IsoNet.utils.fileio import read_mrc
    volume, _ = read_mrc(volume_file)
    os.makedirs(output_folder, exist_ok=True)

    zc, yc, xc = np.array(volume.shape) // 2
    xy_slice = volume[zc, :, :]
    xz_slice = volume[:, yc, :]
    yz_slice = volume[:, :, xc]

    # Pad XZ slice to square for isotropic FFT
    padded_xz, (pad_y, pad_x) = pad_to_square(xz_slice)
    fft_xz = np.fft.fft2(padded_xz)
    power_spectrum = np.abs(np.fft.fftshift(fft_xz)) ** 2
    power_spectrum = np.log1p(power_spectrum)

    # Crop power spectrum back to original xz shape
    power_spectrum = crop_center(power_spectrum, xz_slice.shape)
    s = min(xy_slice.shape[0],xy_slice.shape[1])
    xy_slice = crop_center(xy_slice,(s,s))
    xy_slice = min_max_normalize_image_with_clipping(xy_slice)
    xz_slice = min_max_normalize_image_with_clipping(xz_slice)
    yz_slice = min_max_normalize_image_with_clipping(yz_slice)
    image_data = [
        (xy_slice, "xy"),
        (xz_slice, "xz"),
        (yz_slice, "yz"),
        (power_spectrum, "power")
    ]

    for data, label in image_data:
        plt.figure(figsize=(6, 6))
        plt.imshow(data, cmap='gray')
        plt.axis('off')
        plt.axis('image')  # Keep square pixel ratio
        filename = os.path.join(output_folder, f"{label}_epoch_{iteration}.png")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"Saved all slices and square power spectrum for epoch {iteration} to '{output_folder}', the tomo file name is {volume_file}")
