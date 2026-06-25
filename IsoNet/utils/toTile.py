import numpy as np

class reform3D:
    def __init__(self,data3D, cubesize, cropsize, edge_depth):
        self._sp = np.array(data3D.shape)
        self._orig_data = data3D
        self.cubesize = cubesize
        self.cropsize = cropsize
        self.edge_depth = edge_depth
        self._sidelen = np.ceil((self._sp + edge_depth * 2)/self.cubesize).astype(int)
        #self._sidelen = np.ceil((1.*self._sp)/self.cubesize).astype(int)

    def pad_and_crop(self):
        
        #----------------------------|---------------------------
        #|                           |
        #|     ---------------|------|--------edge---------------
        #|     |  ------------|------|--------image_edge----------  
        #|     |  |           |      |
        #|     |  |           |      |
        #|     |  |           |      |
        #|     |  |           |      |
        #|     -----cube-------      | 
        #|     |  |                  |
        #|     |  |                  |
        #|----------crop--------------       
        #|     |  |        
        #|     |  |       
        #|     |  |       
        #|     |  |
        pad_left = int((self.cropsize - self.cubesize)/2 + self.edge_depth)
        
        # pad_right + pad_left + shape = sidelen * cube_zize + (crop_size-cube_size)
        # pad_right + pad_left + shape >= (self._sp + edge_depth * 2) + (crop_size-cube_size)
        pad_right = (self._sidelen * self.cubesize + (self.cropsize-self.cubesize) - pad_left - self._sp).astype(int)

        data = np.pad(self._orig_data,((pad_left,pad_right[0]),(pad_left,pad_right[1]),(pad_left,pad_right[2])),'symmetric').astype(np.float32)
        outdata=[]

        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    cube = data[i*self.cubesize:i*self.cubesize+self.cropsize,
                            j*self.cubesize:j*self.cubesize+self.cropsize,
                            k*self.cubesize:k*self.cubesize+self.cropsize]
                    outdata.append(cube)
        outdata=np.array(outdata, dtype = np.float32)
        return outdata
    
    def mask(self, x_len, y_len, z_len):
        # need to consider should partisioned to len+1 so that left and right can add to one
        p = 2*self.edge_depth#(self.cropsize - self.cubesize)
        assert x_len > 2*p
        assert y_len > 2*p
        assert z_len > 2*p

        array_x = np.minimum(np.arange(x_len+1), p) / p
        array_x = array_x * np.flip(array_x)
        array_x  = array_x[np.newaxis,np.newaxis,:]

        array_y = np.minimum(np.arange(y_len+1), p) / p
        array_y = array_y * np.flip(array_y)
        array_y  = array_y[np.newaxis,:,np.newaxis]

        array_z = np.minimum(np.arange(z_len+1), p) / p
        array_z = array_z * np.flip(array_z)
        array_z  = array_z[:,np.newaxis,np.newaxis]

        out = array_x * array_y * array_z
        return out[:x_len,:y_len,:z_len].astype(np.float32)


    def restore(self,cubes):

        start = (self.cropsize-self.cubesize)//2-self.edge_depth
        end = (self.cropsize-self.cubesize)//2+self.cubesize+self.edge_depth
        cubes = cubes[:,start:end,start:end,start:end]

        restored = np.zeros((self._sidelen[0]*self.cubesize+self.edge_depth*2,
                        self._sidelen[1]*self.cubesize+self.edge_depth*2,
                        self._sidelen[2]*self.cubesize+self.edge_depth*2), dtype = np.float32)
        # logging.info("size restored", restored.shape)
        mask_cube = self.mask(self.cubesize+self.edge_depth*2,self.cubesize+self.edge_depth*2,self.cubesize+self.edge_depth*2)
        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    restored[i*self.cubesize:(i+1)*self.cubesize+self.edge_depth*2,
                        j*self.cubesize:(j+1)*self.cubesize+self.edge_depth*2,
                        k*self.cubesize:(k+1)*self.cubesize+self.edge_depth*2] \
                        += cubes[i*self._sidelen[1]*self._sidelen[2]+j*self._sidelen[2]+k]\
                            *mask_cube
                        
                    
        p =self.edge_depth*2 #int((self.cropsize-self.cubesize)/2+self.edge_depth)
        restored = restored[p:p+self._sp[0],p:p+self._sp[1],p:p+self._sp[2]]
        return restored

    def restore_to_volume(self, cube_sets, target, scale=1.0):
        start = (self.cropsize-self.cubesize)//2-self.edge_depth
        end = (self.cropsize-self.cubesize)//2+self.cubesize+self.edge_depth
        cube_shape = self.cubesize+self.edge_depth*2
        mask_cube = self.mask(cube_shape, cube_shape, cube_shape)
        crop_start = self.edge_depth*2
        crop_end = crop_start + self._sp

        cube_index = 0
        for cubes in cube_sets:
            for cube in cubes:
                cube = np.squeeze(cube)[start:end,start:end,start:end]
                grid_z = cube_index // (self._sidelen[1] * self._sidelen[2])
                grid_y = (cube_index // self._sidelen[2]) % self._sidelen[1]
                grid_x = cube_index % self._sidelen[2]

                restored_start = np.array([
                    grid_z * self.cubesize,
                    grid_y * self.cubesize,
                    grid_x * self.cubesize,
                ])
                restored_end = restored_start + cube_shape
                write_start = np.maximum(restored_start, crop_start)
                write_end = np.minimum(restored_end, crop_end)

                if np.all(write_start < write_end):
                    target_slices = tuple(
                        slice(int(s - crop_start), int(e - crop_start))
                        for s, e in zip(write_start, write_end)
                    )
                    cube_slices = tuple(
                        slice(int(s - rs), int(e - rs))
                        for s, e, rs in zip(write_start, write_end, restored_start)
                    )
                    target[target_slices] += cube[cube_slices] * mask_cube[cube_slices] * scale

                cube_index += 1

        expected_cubes = int(np.prod(self._sidelen))
        if cube_index != expected_cubes:
            raise ValueError(f"Expected {expected_cubes} cubes, got {cube_index}")

    def mask_old(self):
        from functools import reduce
        c = self.cropsize
        p = (self.cropsize - self.cubesize)
        mask = np.ones((c, c, c))
        f = lambda x: min(x, p)/p 
        for i in range(c):
            for j in range(c):
                for k in range(c):
                    d = [i, c-i, j, c-j, k, c-k]
                    d = map(f,d)
                    d = reduce(lambda a,b: a*b, d)
                    mask[i,j,k] = d
        return mask
    def restore_from_cubes(self,cubes):

        new = np.zeros((self._sidelen[0]*self.cubesize,
                        self._sidelen[1]*self.cubesize,
                        self._sidelen[2]*self.cubesize), dtype = np.float32)
        start=int((self.cropsize-self.cubesize)/2)
        end=int((self.cropsize+self.cubesize)/2)
        
        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    new[i*self.cubesize:(i+1)*self.cubesize,
                        j*self.cubesize:(j+1)*self.cubesize,
                        k*self.cubesize:(k+1)*self.cubesize] \
                        = cubes[i*self._sidelen[1]*self._sidelen[2]+j*self._sidelen[2]+k][start:end,start:end,start:end]
        return new[0:self._sp[0],0:self._sp[1],0:self._sp[2]]
    # def pad4times(self,time=4):
    #     sp = np.array(self._orig_data.shape)
    #     sp = np.expand_dims(sp,axis=0)
    #     padsize = (sp // time + 1) * time - sp
    #     self._padsize =padsize
    #     logging.info(padsize, np.zeros((len(self._orig_data.shape),1)))
    #     width = np.concatenate((np.zeros((len(self._orig_data.shape),1),int),padsize.T),axis=1)
    #     return np.pad(self._orig_data,width,'edge')
    def cropback(self,padded):
        sp = padded.shape
        ps = self._padsize
        orig_sp = sp - ps
        print ('orig_sp',orig_sp)
        return padded[:orig_sp[0][0]][:orig_sp[0][1]][:orig_sp[0][2]]

if __name__ == '__main__':
    pass
