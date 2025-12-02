
import numpy as np

def mw2D_odd(dim,missingAngle=[30,30]):
    mw=np.zeros((dim,dim),dtype=np.double)
    missingAngle = np.array(missingAngle)
    missing=np.pi/180*(90-missingAngle)
    for i in range(dim):
        for j in range(dim):
            y=(i-dim/2)
            x=(j-dim/2)
            if x==0:# and y!=0:
                theta=np.pi/2
            #elif x==0 and y==0:
            #    theta=0
            #elif x!=0 and y==0:
            #    theta=np.pi/2
            else:
                theta=abs(np.arctan(y/x))

            if x**2+y**2<=(dim/2)**2:
                if x > 0 and y > 0 and theta < missing[1]:
                    mw[i,j]=1#np.cos(theta)
                if x < 0 and y < 0 and theta < missing[1]:
                    mw[i,j]=1#np.cos(theta)
                if x > 0 and y < 0 and theta < missing[0]:
                    mw[i,j]=1#np.cos(theta)
                if x < 0 and y > 0 and theta < missing[0]:
                    mw[i,j]=1#np.cos(theta)

            if int(y) == 0:
                mw[i,j]=1

    return mw

def mw2D(dim, missingAngle=[30,30], tilt_step=None, start_dim = 48):

    if tilt_step not in [None,"None"]:
        import os
        from IsoNet.utils.fileio import read_mrc
        path = os.path.abspath(__file__).split("IsoNet/utils")[0]
        file = f"{path}missing_wedges/simulated_F_{dim}_{tilt_step}.mrc"
        precalculated_mw,_ = read_mrc(file)
    else:
        precalculated_mw = np.ones((dim, dim))
        start_dim = 0


    mw=np.zeros((dim,dim),dtype=np.double)
    missingAngle = np.array(missingAngle)
    missing=90-missingAngle

    for i in range(dim):
        for j in range(dim):
            y=(i-dim/2)
            x=(j-dim/2)
            if x==0:
                theta=90
            else:
                theta=np.rad2deg(abs(np.arctan(y/x)))

            R2 = x**2+y**2
            if R2>((start_dim)/2)**2:
                if x > 0 and y > 0 and theta < missing[1]:
                    mw[i,j]=precalculated_mw[i][j]
                if x < 0 and y < 0 and theta < missing[1]:
                    mw[i,j]=precalculated_mw[i][j]
                if x > 0 and y < 0 and theta < missing[0]:
                    mw[i,j]=precalculated_mw[i][j]
                if x < 0 and y > 0 and theta < missing[0]:
                    mw[i,j]=precalculated_mw[i][j]
            else:
                if x > 0 and y > 0 and theta < missing[1]:
                    mw[i,j]=1
                if x < 0 and y < 0 and theta < missing[1]:
                    mw[i,j]=1
                if x > 0 and y < 0 and theta < missing[0]:
                    mw[i,j]=1
                if x < 0 and y > 0 and theta < missing[0]:
                    mw[i,j]=1       

            if int(y) == 0:
                mw[i,j]=1


    return mw.astype(np.float32)

def mw3D(dim,missingAngle=[30,30],tilt_step=None, start_dim = 48, spherical=True, dose_weight=1):
    mw = mw2D(dim,missingAngle,tilt_step, start_dim)
    mw = np.repeat(mw[:,np.newaxis,:], dim, axis=1).astype(np.float32)
    if spherical:
        center = dim // 2
        x, y, z = np.ogrid[:dim, :dim, :dim]
        distance_squared = ((x - center)/dose_weight) ** 2 + ((y - center)/dose_weight) ** 2 + ((z - center)/dose_weight) ** 2
        mask = distance_squared > center ** 2
        mw[mask] = 0
    return mw


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
    return data


    
