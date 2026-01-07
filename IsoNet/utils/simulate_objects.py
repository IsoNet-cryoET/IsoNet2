import numpy as np
import random
from scipy.ndimage import rotate
from IsoNet.utils.fileio import read_mrc
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

def create_structure(file_name = 'pdb3lel.mrc'):
    data, _ = read_mrc(file_name)
    return data

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
    coord = []
    # Randomly place objects in the array with random orientation
    for _ in range(num_objects):
        #shape = random.choice(['cube', 'pyramid', 'sphere'])
        shape = random.choice(['cube', 'pyramid', 'sphere'])
        center = (random.randint(41, array_size[0]-41),
                random.randint(41, array_size[1]-41),
                random.randint(41, array_size[2]-41))
        coord.append([center[2],center[1],center[0]])
        
        if shape == 'cube':
            obj = create_cube()
        elif shape == 'pyramid':
            obj = create_pyramid()
        elif shape == 'sphere':
            obj = create_sphere(size=20)
        elif shape == 'molecule':
            obj = create_structure()

        # Apply random rotation
        if rotate:
            angles = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
            obj = rotate_object(obj, angles)
        
        # Place the object into the main array
        place_object(array, obj, center)
    return array, coord


def generate_2D_image(array_size=(256, 512, 512), num_objects=200, size_object=16, CTF_image=None):
    array = generate_3D_image(array_size, num_objects, size_object)
    array_2D = np.average(array.astype(np.float32),axis=0)
    if CTF_image is not None:
        from IsoNet.util.Fourier import apply_F_filter
        array_2D = apply_F_filter(array_2D, CTF_image)
    return array_2D
    
if __name__ == '__main__':
    # from IsoNet.util.CTF import ctf2d
    import mrcfile
    from IsoNet.utils.missing_wedge import mw3D
    from IsoNet.utils.Fourier import apply_F_filter
    # cube_size = 64
    # diameter = 20
    # MW = mw3D(cube_size)
    # sphere = np.zeros((cube_size,cube_size,cube_size))
    # S = create_sphere(size=diameter)

    # sphere[cube_size//2-diameter//2:cube_size//2+diameter//2,cube_size//2-diameter//2:cube_size//2+diameter//2,cube_size//2-diameter//2:cube_size//2+diameter//2] = S
    # data = apply_F_filter(sphere, MW)
    # for i in range(20):
    #     with mrcfile.new(f"GT_{i}.mrc", overwrite=True) as mrc:
    #         mrc.set_data((sphere*-1).astype(np.float32))
    #     with mrcfile.new(f"sphere_{i}.mrc", overwrite=True) as mrc:
    #         mrc.set_data(data*-1)
    # with mrcfile.new("missingwedge.mrc", overwrite=True) as mrc:
    #     mrc.set_data(MW)
    # CTF_image = ctf2d(2,300,2.7,1, 0.1, 0, 0, 1024)
    # with mrcfile.new("CTF.mrc", overwrite=True) as mrc:
    #     mrc.set_data(CTF_image)
    array, coord = generate_3D_image(array_size=(256, 512, 512),num_objects=80, rotate=True)
    #array = array+np.random.randn(1024,1024).astype(np.float32)*0.1
    with mrcfile.new("test.mrc", overwrite=True) as mrc:
        mrc.set_data(array*-1)
    np.savetxt('particles.txt',coord, fmt='%f')
    # array = apply_F_filter(array, CTF_image)
    # with mrcfile.new("test_filtered.mrc", overwrite=True) as mrc:
    #     mrc.set_data(array)


