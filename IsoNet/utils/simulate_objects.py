import numpy as np
import random
from scipy.ndimage import rotate

# Function to rotate a 3D array
def rotate_object(obj, angles):
    # Rotate around each axis by the given angles
    obj = rotate(obj, angle=angles[0], axes=(1, 2), reshape=False, order=1)
    obj = rotate(obj, angle=angles[1], axes=(0, 2), reshape=False, order=1)
    obj = rotate(obj, angle=angles[2], axes=(0, 1), reshape=False, order=1)
    return obj

# Function to place an object in the main array with random orientation
def place_object(array, obj, center):
    x0, y0, z0 = center
    obj_size = obj.shape[0]
    half_size = obj_size // 2

    x_min = max(0, x0-half_size)
    x_max = min(array.shape[0], x0+half_size)
    y_min = max(0, y0-half_size)
    y_max = min(array.shape[1], y0+half_size)
    z_min = max(0, z0-half_size)
    z_max = min(array.shape[2], z0+half_size)
    
    # Compute the slices for the object
    obj_x_min = max(0, half_size - x0)
    obj_x_max = obj_size - max(0, x0 + half_size - array.shape[0])
    obj_y_min = max(0, half_size - y0)
    obj_y_max = obj_size - max(0, y0 + half_size - array.shape[1])
    obj_z_min = max(0, half_size - z0)
    obj_z_max = obj_size - max(0, z0 + half_size - array.shape[2])
    
    # Place the object in the array
    array[x_min:x_max, y_min:y_max, z_min:z_max] += obj[obj_x_min:obj_x_max, obj_y_min:obj_y_max, obj_z_min:obj_z_max]

# Function to create a 3D cube
def create_cube(size=16):
    return np.ones((size, size, size), dtype=np.uint8)

# Function to create a 3D pyramid
def create_pyramid(size=16):
    pyramid = np.zeros((size, size, size), dtype=np.uint8)
    half_size = size // 2
    for z in range(size):
        current_size = half_size - abs(z - half_size)
        pyramid[half_size-current_size:half_size+current_size, 
                half_size-current_size:half_size+current_size, 
                z] = 1
    return pyramid

# Function to create a 3D sphere
def create_sphere(size=16):
    sphere = np.zeros((size, size, size), dtype=np.uint8)
    radius = size // 2
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if (x - radius)**2 + (y - radius)**2 + (z - radius)**2 <= radius**2:
                    sphere[x, y, z] = 1
    return sphere

def generate_3D_image(array_size=(256, 512, 512), num_objects=200, size_object=16, rotate=True):
    # Initialize the 3D array
    array = np.zeros(array_size, dtype=np.float32)

    # Randomly place objects in the array with random orientation
    for _ in range(num_objects):
        shape = random.choice(['cube', 'pyramid', 'sphere'])
        center = (random.randint(size_object, array_size[0]-size_object),
                random.randint(size_object, array_size[1]-size_object),
                random.randint(size_object, array_size[2]-size_object))
        
        if shape == 'cube':
            obj = create_cube()
        elif shape == 'pyramid':
            obj = create_pyramid()
        elif shape == 'sphere':
            obj = create_sphere()

        # Apply random rotation
        if rotate:
            angles = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
            obj = rotate_object(obj, angles)
        
        # Place the object into the main array
        place_object(array, obj, center)
    return array


def generate_2D_image(array_size=(256, 512, 512), num_objects=200, size_object=16, CTF_image=None):
    array = generate_3D_image(array_size, num_objects, size_object)
    array_2D = np.average(array.astype(np.float32),axis=0)
    if CTF_image is not None:
        from IsoNet.util.Fourier import apply_F_filter
        array_2D = apply_F_filter(array_2D, CTF_image)
    return array_2D
    
if __name__ == '__main__':
    from IsoNet.util.CTF import ctf2d
    import mrcfile
    from IsoNet.util.Fourier import apply_F_filter

    CTF_image = ctf2d(2,300,2.7,1, 0.1, 0, 0, 1024)
    with mrcfile.new("CTF.mrc", overwrite=True) as mrc:
        mrc.set_data(CTF_image)
    array = generate_2D_image(array_size=(128, 1024, 1024),num_objects=600, CTF_image=None)
    array = array+np.random.randn(1024,1024).astype(np.float32)*0.1
    with mrcfile.new("test.mrc", overwrite=True) as mrc:
        mrc.set_data(array)
    array = apply_F_filter(array, CTF_image)
    with mrcfile.new("test_filtered.mrc", overwrite=True) as mrc:
        mrc.set_data(array)


