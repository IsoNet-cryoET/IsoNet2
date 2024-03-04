
import mrcfile
import numpy as np
def get_F_cone(size=160, angle=45):
    data = np.zeros((size,size,size), dtype = np.float32)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                r = ( (i-size/2)**2 + (j-size/2)**2 ) **0.5
                z = k  - size/2
                threshold = r*np.tan(np.radians(angle))
                if abs(z) > threshold:
                    data[k,j,i] = 1
    data=1-data
                
    with mrcfile.new("F_cone.mrc", overwrite=True) as mrc:
        mrc.set_data(data)
    return data

def get_F_wedge(size=160, angle=45):
    data = np.zeros((size,size,size), dtype = np.float32)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                r = abs(i-size/2)
                z = k  - size/2
                threshold = r*np.tan(np.radians(angle))
                if abs(z) > threshold:
                    data[k,j,i] = 1
    data=1-data

    with mrcfile.new("F_wedge.mrc", overwrite=True) as mrc:
        mrc.set_data(data)
    return data

def get_F_double_wedge(size=160, angle=45):
    data = np.zeros((size,size,size), dtype = np.float32)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                r = abs(i-size/2)
                z = k  - size/2
                threshold = r*np.tan(np.radians(angle))
                if abs(z) > threshold:
                    data[k,j,i] = 1
                if data[k,j,i] ==0:
                    r = abs(j-size/2)
                    z = k  - size/2
                    threshold = r*np.tan(np.radians(angle))
                    if abs(z) > threshold:
                        data[k,j,i] = 1

    data=1-data

    with mrcfile.new("F_double_wedge.mrc", overwrite=True) as mrc:
        mrc.set_data(data)
    return data


def apply(data, F):
    mw = F
    mw = np.fft.fftshift(mw)
#    mw = mw * ld1 + (1-mw) * ld2

    f_data = np.fft.fftn(data)
    outData = mw*f_data
    inv = np.fft.ifftn(outData)
    outData = np.real(inv).astype(np.float32)
    return outData

def normalize(d):
    d = (d-d.mean()) / d.std()
    return d

def generate_command(halfmap_file, FSC_file, mask_file):
    return f"isonet.py map_refine {halfmap_file} {FSC_file} {mask_file}"

if __name__ == '__main__':
    #F = get_F_double_wedge(400,60)
    map_diameter = 400
    input_file = "8b0x-3A-1grid.mrc"
    angle = 45

    F = get_F_wedge(map_diameter,angle)
    with mrcfile.new(f"wedge_{angle}.mrc",overwrite=True) as mrc:
        mrc.set_data(F.astype(np.float32))

    with mrcfile.open(input_file,'r') as mrc:
        data = mrc.data
    data = normalize(data)    
    out_data = apply(data,F)

    with mrcfile.new(f"data_{angle}.mrc",overwrite=True) as mrc:
        mrc.set_data(out_data)



    
