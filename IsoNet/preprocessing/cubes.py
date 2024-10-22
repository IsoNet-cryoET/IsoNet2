import os
import numpy as np
import mrcfile
from IsoNet.utils.toTile import reform3D
def extract_with_overlap(current_map, crop_size, cube_size, output_dir, prefix='', wedge = None):
    r3d=reform3D(current_map, cube_size, crop_size, edge_depth=7)
    subtomos=r3d.pad_and_crop()
    mrc_list = []
    for j,s in enumerate(subtomos):
        im_name = '{}/subvolume{}_{:0>6d}.mrc'.format(output_dir, prefix, j)
        with mrcfile.new(im_name, overwrite=True) as output_mrc:
            output_mrc.set_data(s.astype(np.float32))

        mrc_list.append(im_name)
    return mrc_list

def extract_subvolume(current_map, seeds, crop_size, output_dir, start=0, wedge = None):
    subtomos=crop_cubes(current_map,seeds,crop_size)
    mrc_list = []
    # from IsoNet.utils.missing_wedge import mw3D
    # mw = mw3D(crop_size)
    for j,s in enumerate(subtomos):
        # if wedge is not None:
        #     s = apply_wedge(s, wedge)
        im_name = '{}/subvolume_{:0>6d}.mrc'.format(output_dir, start+j)
        with mrcfile.new(im_name, overwrite=True) as output_mrc:
            output_mrc.set_data(s.astype(np.float32))

        mrc_list.append(im_name)
    return mrc_list

def create_cube_seeds(img3D,nCubesPerImg,cubeSideLen,mask=None):
    sp=img3D.shape
    if mask is None:
        cubeMask=np.ones(sp)
    else:
        cubeMask=mask
    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip((cubeSideLen,cubeSideLen,cubeSideLen), sp)])
    valid_inds = np.where(cubeMask[border_slices])
    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]
    sample_inds = np.random.choice(len(valid_inds[0]), nCubesPerImg, replace=len(valid_inds[0]) < nCubesPerImg)
    rand_inds = [v[sample_inds] for v in valid_inds]
    return (rand_inds[0],rand_inds[1], rand_inds[2])

def mask_mesh_seeds(mask,sidelen,croplen,threshold=0.01,indx=0):
    #indx = 0 take the even indix element of seed list,indx = 1 take the odd 
    # Count the masked points in the box centered at mesh grid point, if greater than threshold*sidelen^3, Take the grid point as seed.
    sp = mask.shape
    ni = [(i-croplen)//sidelen +1 for i in sp]
    # res = [((i-croplen)%sidelen) for i in sp]
    margin = croplen//2 - sidelen//2
    ind_list =[]
    for z in range(ni[0]):
        for y in range(ni[1]):
            for x in range(ni[2]):
                if np.sum(mask[margin+sidelen*z:margin+sidelen*(z+1),
                margin+sidelen*y:margin+sidelen*(y+1),
                margin+sidelen*x:margin+sidelen*(x+1)]) > sidelen**3*threshold:
                    ind_list.append((margin+sidelen//2+sidelen*z, margin+sidelen//2+sidelen*y,
                margin+sidelen//2+sidelen*x))
    ind_list = ind_list[indx:-1:2]
    ind0 = [i[0] for i in ind_list]
    ind1 = [i[1] for i in ind_list]
    ind2 = [i[2] for i in ind_list]
    # return ind_list
    return (ind0,ind1,ind2)




def crop_cubes(img3D,seeds,cubeSideLen):
    size=len(seeds[0])
    cube_size=(cubeSideLen,cubeSideLen,cubeSideLen)
    cubes=[img3D[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,cube_size))] for r in zip(*seeds)]
    cubes=np.array(cubes)
    return cubes
