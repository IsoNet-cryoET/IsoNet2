
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

def mw2D(dim,missingAngle=[30,30]):
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

            if x**2+y**2<=min(dim/2,dim/2)**2:
                if x > 0 and y > 0 and theta < missing[0]:
                    mw[i,j]=1#np.cos(theta)
                if x < 0 and y < 0 and theta < missing[0]:
                    mw[i,j]=1#np.cos(theta)
                if x > 0 and y < 0 and theta < missing[1]:
                    mw[i,j]=1#np.cos(theta)
                if x < 0 and y > 0 and theta < missing[1]:
                    mw[i,j]=1#np.cos(theta)

            if int(y) == 0:
                mw[i,j]=1
    #from mwr.util.image import norm_save
    #norm_save('mw.tif',self._mw)
    return mw

# def apply(data, F):
#     mw = F
#     mw = np.fft.fftshift(mw)
# #    mw = mw * ld1 + (1-mw) * ld2

#     f_data = np.fft.fftn(data)
#     outData = mw*f_data
#     inv = np.fft.ifftn(outData)
#     outData = np.real(inv).astype(np.float32)
#     return outData

#import tensorflow as tf
# def apply_wedge_dcube(ori_data, mw2d = None, mw3d=None, ld1 = 1, ld2 = 0):
#     # For apply wedge for multiple volumes
#     if mw3d is None:
#         if len(ori_data.shape) > 4:
#             ori_data = np.squeeze(ori_data)
#         if mw2d is None:
#             mw2d = mw2D(ori_data.shape[-2])
#         data = np.rot90(ori_data, k=1, axes=(1,2)) #clock wise of counter clockwise??
#         data = np.fft.ifft2(np.fft.fftshift(mw2d)[np.newaxis, np.newaxis, :, :] * np.fft.fft2(data))
#         #data = tf.signal.ifft2d(np.fft.fftshift(mw2d)[np.newaxis, np.newaxis, :, :] * tf.signal.fft2d(data))
#         data = np.real(data)
#         data=np.rot90(data, k=3, axes=(1,2))

#     else:
#         mw = mw3d
#         mw = mw*ld1 + (1-mw) * ld2
#         mwshift = np.fft.fftshift(mw)
#         data = np.zeros_like(ori_data)
#         for i,d in enumerate(ori_data):
#             f_data = np.fft.fftn(d)
#             outData = mwshift*f_data
#             inv = np.fft.ifftn(outData)
#             data[i] = np.real(inv).astype(np.float32)
#     return data

def apply_wedge(ori_data, mw, ld1 = 1, ld2 =0):
    mw = np.fft.fftshift(mw)
    mw = mw * ld1 + (1-mw) * ld2
    f_data = np.fft.fftn(ori_data)
    outData = mw*f_data
    inv = np.fft.ifftn(outData)
    outData = np.real(inv).astype(np.float32)
    return outData



    
