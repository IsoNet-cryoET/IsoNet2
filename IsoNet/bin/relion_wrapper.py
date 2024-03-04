# -*- coding: utf-8 -*-


# To execute this script in relion_refine it is necessary to use the argument "--external_reconstruct".
# It is also required to set the RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE environment variable 
# to point to this script.


#export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE="python /home/cii/software/IsoNet/bin/relion_wrapper.py"
#export CONDA_ENV="torch2-test"
#export CUDA_VISIBLE_DEVICES="0"



import os
import os.path
import sys
import time
import numpy as np
import mrcfile
import shutil
from subprocess import check_output



def execute_external_relion(star):

    params = ' relion_external_reconstruct'
    params += ' %s' %star
    os.system(params)     
    
def execute_deep(mrc1, mrc2, fsc3d, dir, gpu, limit_res, epochs = 1, mask_file = None, pretrained_model=None, alpha = None, acc_batches= None, batch_size = None, beta=None): 
    #data_file =  ' %s/%s_it%s_half%s_class001_external_reconstruct.mrc' %(dir, basename, var, half)   
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' IsoNet.py refine '
    params += f" {mrc1} {mrc2}"      
    params += ' --aniso_file %s' %(fsc3d) 
    params += ' --epochs %s --n_subvolume 1000'   %(epochs)
    if acc_batches is not None:
        params += ' --acc_batches %s'   %(acc_batches) 
    if batch_size is not None:
        params += ' --batch_size %s'   %(batch_size) 
    if alpha is not None:
        params += ' --alpha %s'   %(alpha) 
    if beta is not None:
        params += ' --beta %s'   %(beta) 
    params += ' --output_dir %s' %(dir) 
    params += ' --gpuID %s' %(gpu) 
    params += ' --limit_res %s' %(limit_res)
    if pretrained_model is not None:
        params += ' --pretrained_model %s' %(pretrained_model)
    if mask_file is not None:
        params += ' --mask %s' %(mask_file)

    print(params)
    os.system(params)

def execute_3dfsc(fn1,fn2,fscn,limit_res=None, mask_file=None): 
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' IsoNet.py fsc3d '
    params += ' %s' %(fn1)  
    params += ' %s' %(fn2) 
    params += ' -o %s' %(fscn)
    if limit_res is not None:
        params += ' --limit_res %s'%(limit_res)
    if mask_file is not None:
        params += ' --mask %s'%(mask_file)
    os.system(params)

def execute_whitening(fn1,fscn,mask,high_res,low_res=10): 
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' IsoNet.py whitening '
    params += ' %s' %(fn1)  
    params += ' -o %s' %(fscn)
    params += ' --mask %s' %(mask)
    params += ' --high_res %s'%(high_res)
    params += ' --low_res %s'%(low_res)
    print(params)
    os.system(params)

def execute_keep_lowres(f1,f2,f3,limit_res=20, mask_file=None): 
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' IsoNet.py combine_map '
    params += ' %s' %(f1)  
    params += ' %s' %(f2)  
    params += ' %s' %(f3)  
    params += ' %s' %(limit_res)
    params += ' --mask_file %s' %(mask_file)
    print(params)
    os.system(params)   

def execute_angular_whiten(in_name,out_name,resolution_initial, limit_resolution): 
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' IsoNet.py angular_whiten '
    params += ' %s' %(in_name)  
    params += ' %s' %(out_name)  
    params += ' %s' %(resolution_initial)  
    params += ' %s' %(limit_resolution)
    print(params)
    os.system(params)   

def parse_env(ENV_STRING, val_type, default, silence=True):
    val = os.getenv(ENV_STRING)
    if val_type == "bool":
        if val=="True" or val=="true" or val=="TRUE" or val==True:
            val = True
        elif val=="False" or val=="false" or val=="FALSE" or val==False:
            val = False
        else:
            #if not silence:
            #    print(f"{ENV_STRING}={val} does not match True or False")
            val = default

    elif val_type == 'int':
        if val is None or len(val) == 0:
            val = default
        else:
            val = int(val)

    elif val_type == 'float':
        if val is None or len(val) == 0:
            val = default
        else:
            val = float(val)
    
    elif val_type == 'str':
        if val is None or len(val) == 0:
            if not silence:
                print(f"{val} is None")
            val = default
    if not silence:
        print(f"set {ENV_STRING}={val}")
    return val

def parse_filename(star):
    dir=os.path.dirname(star)

    part = star.split('/')[-1].split('_')
    iter_string = part[1]
    basename = part[0]
    iter_number = int(iter_string[2:5])
    half_str = part[2]

    if int(iter_number) <= 9: 
        var = '00%d' %(iter_number)
        beforeVar = '00%d' %(iter_number-1)
    elif int(iter_number) == 10:
        var = '0%d' %(iter_number)
        beforeVar = '00%d' %(iter_number-1)
    else:
        var = '0%d' %(iter_number)
        beforeVar = '0%d' %(iter_number-1)
    return dir, basename, half_str, var, beforeVar

def wait_until_file(sync_file,interval_time=10,total_time=10000):
    time.sleep(interval_time)
    for i in range (1,total_time):
        try:
            with open(sync_file, 'r') as f:
                result =  f.read()
            s = f'rm {sync_file}'
            check_output(s, shell=True)
            return result
        except:
            time.sleep(interval_time)

def write_to_file(file_name, words='This is a temperal file to synchronize half 1 and 2 for isonet'):
    with open(file_name, 'w') as f:
        f.write(words)

def fsc_matching(h_target, h_source, fsc3d, low_r, high_r):
    from IsoNet.util.FSC import get_donut
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

    print(m_in,m_out,n_in,n_out,w_in,w_out)
    fsc3d_donut = fsc3d * mask_donut
    invert_fsc3d_donut = (1-fsc3d) * mask_donut

    res_in = np.real(ifftn(F_target * fftshift(fsc3d_donut))).astype(np.float32)
    res_out = np.real(ifftn(F_target * fftshift(invert_fsc3d_donut))).astype(np.float32)

    in_donut = (res_in*w_in + res_out*w_out)/(w_in + w_out)
    out_donut = np.real(ifftn(F_target * fftshift(1-mask_donut))).astype(np.float32)

    return in_donut + out_donut

def execute_3dfsd(star_file,fscn,map_dim = 256, apix=1.31,sym="c3",low_res=10, high_res=3): 
    #IsoNet.py fsd3d T00_HA_130K-Equalized_run-data.star 256 --sym c3 --low_res 10 --high_res 3 --apix 1.31
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' IsoNet.py fsd3d '
    params += ' %s' %(star_file)  
    params += ' %s' %(map_dim)
    params += ' -o %s' %(fscn)
    params += ' --apix %s'%(apix)
    params += ' --sym %s'%(sym)
    params += ' --low_res %s'%(low_res)
    params += ' --high_res %s'%(high_res)
    print(params)
#     params += ' -p wideTarget' 
    os.system(params)

if __name__=="__main__":  
    dir, basename, half_str, var, beforeVar = parse_filename(sys.argv[1])
    if half_str == "half2":
        silence = True
    else:
        silence = False
        print("iter =", var)

    import torch
    gpu_list = list(range(torch.cuda.device_count()))
    gpu_list=','.join(map(str, gpu_list))

    gpu = parse_env("CUDA_VISIBLE_DEVICES", "string", gpu_list, silence)
    CONDA_ENV = parse_env("CONDA_ENV", "string", False, silence)
    whitening = parse_env("ISONET_WHITENING", "bool", True, silence)
    whitening_low = parse_env("ISONET_WHITENING_LOW", "float", 10, silence)
    retrain = parse_env("ISONET_RETRAIN_EACH_ITER", "bool", True, silence)
    beta = parse_env("ISONET_BETA", "float", 0.5, silence)
    alpha = parse_env("ISONET_ALPHA", "float", 1, silence)
    
    limit_healpix = parse_env("ISONET_START_HEALPIX", "int", 3, silence)
    acc_batches = parse_env("ISONET_ACC_BATCHES", "int", 2, silence)
    epochs = parse_env("ISONET_EPOCHS", "int", 5, silence)
    start_epochs = epochs#parse_env("ISONET_START_EPOCHS", "int", 5, silence)
    keep_lowres = parse_env("ISONET_KEEP_LOWRES", "bool", False, silence)
    lowpass = parse_env("ISONET_LOWPASS", "bool", True, silence)

    awhiten = parse_env("ISONET_ANGULAR_WHITEN", "bool", False, silence)    
    use_3dfsd = parse_env("ISONET_3DFSD", "bool", False, silence)    
    use_fsc_05 = parse_env("ISONET_FSC_05", "bool", False, silence)    
    do_fsc_weighting = parse_env("ISONET_FSC_WEIGHTING", "bool", True, silence)   
    #do_fsc_weighting_after = parse_env("ISONET_FSC_WEIGHTING_AFTER", "bool", True, silence) 




    # resolution_initial
    with open("%s/%s_it000_optimiser.star" %(dir,basename)) as file:
        for line_number,li in enumerate(file.readlines()):
            if '--keep_lowres' in li:
                keep_lowres = True
                break
    if not silence:
        print("set ISONET_KEEP_LOWRES=",keep_lowres)       

    # healpix
    with open("%s/%s_it%s_sampling.star" %(dir,basename,beforeVar)) as file:
        for li in file.readlines():
            if "rlnHealpixOrder " in li: 
                healpix = int(li.split()[1]) 
                if not silence:
                    print("healpix = %s" %healpix)
            if "rlnSymmetryGroup " in li: 
                symmetry = li.split()[1]
                if not silence:
                    print("symmetry = %s" %symmetry)
                
    # mask_file
    mask_file = None
    with open("%s/%s_it%s_optimiser.star" %(dir,basename,beforeVar)) as file:
        for li in file.readlines():
            if "_rlnSolventMaskName " in li: 
                mask_file = li.split()[1]
                if not silence:
                    print("mask_file = %s" %mask_file)
                break
    
    sampling_index = None
    # sampling(pixelsize)        
    with open("%s/%s_it%s_data.star" %(dir,basename,beforeVar)) as f:
        for line in f.readlines():
            if "_rlnImagePixelSize" in line:
                sampling_index = int(line.split()[1].split("#")[1])
            if "opticsGroup1" in line:
                sampling = float(line.split()[sampling_index-1])
                if not silence:
                    print("pixel size = %s" %sampling) 

    # limit_resolution: last iter resolution at 0.143
    limit_resolution = 2*sampling
    start_check = 10000
    with open("%s/%s_it%s_half1_model.star" %(dir,basename,beforeVar)) as file:
        signal = True
        for line_number,li in enumerate(file.readlines()):
            if "_rlnEstimatedResolution " in li:
                resolution_index = int(li.split()[1].split("#")[1])
            if "_class001.mrc" in li:
                resolution = float(li.split()[resolution_index-1])
            if "_rlnAngstromResolution" in li:
                Aresolution_index = int(li.split()[1].split("#")[1])
            if "_rlnGoldStandardFsc" in li:
                FSC_index = int(li.split()[1].split("#")[1])
                start_check = line_number
            if line_number >= start_check:
                line_split = li.split()
                if len(line_split)>FSC_index:
                    if signal == True and float(line_split[FSC_index-1])<= 0.5:
                        limit_resolution_05 = float(line_split[Aresolution_index-1])
                        signal = False
                    if float(line_split[FSC_index-1])<= 0.143:
                        limit_resolution = float(line_split[Aresolution_index-1])
                        break
    if not silence:
        print(f"resolution at 0.5 and 0.143 are {limit_resolution_05} and {limit_resolution}")
    # resolution_initial
    with open("%s/%s_it000_optimiser.star" %(dir,basename)) as file:
        for line_number,li in enumerate(file.readlines()):
            if 'ini_high' in li:
                li_sp = li.split()
                index_ini = li_sp.index("--ini_high")
                resolution_initial = float(li_sp[index_ini+1])
                break

    check_final = (half_str == "class001") 
    execute_external_relion(sys.argv[1]) 

    if (check_final is True):
        mrc_final1 = '%s/%s_half1_class001_unfil.mrc' %(dir,basename)
        mrc_final2 = '%s/%s_half2_class001_unfil.mrc' %(dir,basename)
        final_fsc = '%s/%s_3DFSC.mrc' %(dir,basename)
        print("-----------")
        print("Important information for the final iteration!!!")
        print("The final half_unfil maps are generated with relion not IsoNet")
        print("You may want to further use IsoNet to improve the final maps")
        print("IsoNet commands to correct final maps are:")

    else:

        if float(limit_resolution) > resolution_initial:
            limit_resolution = resolution_initial
        if not silence:
            print(f"real limit resolution to {limit_resolution}")

        mrc_initial = '%s/%s_it000_%s_class001.mrc' %(dir,basename,half_str)
        mrc_unfil = '%s/%s_it%s_%s_class001_unfil.mrc' %(dir,basename,var,half_str)
        mrc_unfil_backup = '%s/%s_it%s_%s_class001_unfil_backup.mrc' %(dir,basename,var,half_str)
        mrc_lowpass_backup = '%s/%s_it%s_%s_class001_unfil_lowpass_backup.mrc' %(dir,basename,var,half_str)
        mrc_whiten_backup = '%s/%s_it%s_%s_class001_unfil_whiten_backup.mrc' %(dir,basename,var,half_str)
        mrc_keep_lowres_backup = '%s/%s_it%s_%s_class001_unfil_keep_lowres_backup.mrc' %(dir,basename,var,half_str)
        mrc_overwrite = '%s/%s_it%s_%s_class001_external_reconstruct.mrc' %(dir,basename,var,half_str)
        mrc_overwrite_backup = '%s/%s_it%s_%s_class001_external_reconstruct_backup.mrc' %(dir,basename,var,half_str)
        shutil.copy(mrc_unfil, mrc_unfil_backup)
        shutil.copy(mrc_overwrite, mrc_overwrite_backup)
        shutil.copy(mrc_unfil, mrc_overwrite) 

        if limit_resolution < whitening_low and whitening and (healpix >= limit_healpix):
            execute_whitening(mrc_unfil, mrc_unfil, mask_file, high_res=limit_resolution, low_res=whitening_low)  
            shutil.copy(mrc_unfil, mrc_whiten_backup)
            shutil.copy(mrc_unfil, mrc_overwrite)

        if keep_lowres:
            execute_keep_lowres(mrc_initial,mrc_unfil,mrc_unfil,resolution_initial, mask_file=mask_file) 
            shutil.copy(mrc_unfil, mrc_keep_lowres_backup)
            shutil.copy(mrc_unfil, mrc_overwrite)

        if False:
            s = f"relion_image_handler --i {mrc_unfil} --o {mrc_lowpass_backup} --lowpass {limit_resolution}; cp {mrc_lowpass_backup} {mrc_unfil}"
            print(s)
            check_output(s, shell=True)
            shutil.copy(mrc_unfil, mrc_overwrite) 

        # remove tempory
        remove_intermediate = True
        if remove_intermediate:
            try:
                os.remove('%s/%s_it%s_%s_class001_external_reconstruct_data_real.mrc' %(dir,basename,var,half_str))
                os.remove('%s/%s_it%s_%s_class001_external_reconstruct_data_imag.mrc' %(dir,basename,var,half_str)) 
                os.remove('%s/%s_it%s_%s_class001_external_reconstruct_weight.mrc' %(dir,basename,var,half_str)) 
            except:
                print("do not remove tempory files")

        sync3 = '%s/%s_it%s_class001_external_reconstruct.sync3' %(dir,basename,var)
        sync4 = '%s/%s_it%s_class001_external_reconstruct.sync4' %(dir,basename,var)

        if (healpix >= limit_healpix) and (limit_resolution < 15) and (half_str == "half2"):
            write_to_file(sync4)
            wait_until_file(sync3)

        elif (healpix >= limit_healpix) and (limit_resolution < 15) and (half_str == "half1"):
            wait_until_file(sync4)
            
            mrc1_overwrite = '%s/%s_it%s_half1_class001_external_reconstruct.mrc' %(dir,basename,var)
            mrc2_overwrite = '%s/%s_it%s_half2_class001_external_reconstruct.mrc' %(dir,basename,var)
            mrc1_unfil = '%s/%s_it%s_half1_class001_unfil.mrc' %(dir,basename,var)
            mrc2_unfil = '%s/%s_it%s_half2_class001_unfil.mrc' %(dir,basename,var)
            mrc1_cor = '%s/corrected_%s_it%s_half1_class001_unfil.mrc' %(dir,basename,var)
            mrc2_cor = '%s/corrected_%s_it%s_half2_class001_unfil.mrc' %(dir,basename,var)
            fscn='%s/%s_it%s_3DFSC.mrc' %(dir,basename,var)    

            #Force write pixelsize
            with mrcfile.open(mrc1_unfil) as f1:
                emMap1 = f1.data.astype(np.float32).copy()  
            with mrcfile.open(mrc2_unfil) as f2:
                emMap2 = f2.data.astype(np.float32).copy()  

            with mrcfile.new(mrc1_unfil, overwrite=True) as f1:
                f1.set_data(emMap1)             
                f1.voxel_size = tuple([sampling]*3)
            with mrcfile.new(mrc2_unfil, overwrite=True) as f2:
                f2.set_data(emMap2) 
                f2.voxel_size = tuple([sampling]*3) 

            # execute 3dfsc
            if use_3dfsd:
                print(f"using cryoEF like FSC3D file {fscn}")
                map_dim = emMap1.shape[0]
                execute_3dfsd("%s/%s_it%s_data.star" %(dir,basename,beforeVar), fscn, map_dim=map_dim, low_res=resolution_initial, high_res=limit_resolution,\
                        apix=sampling, sym=symmetry) 
            else:
                execute_3dfsc(mrc1_unfil, mrc2_unfil, fscn, limit_res=limit_resolution, mask_file=mask_file)    

            if do_fsc_weighting:
                with mrcfile.open(fscn, 'r') as mrc:
                    fsc_map = mrc.data
                from IsoNet.util.FSC import rotational_average, FSC_weighting
                fsc_curve = rotational_average(fsc_map)
                weighted_emMap1 = FSC_weighting(emMap1, fsc_curve, weight = False)
                weighted_emMap2 = FSC_weighting(emMap2, fsc_curve, weight = False)
                with mrcfile.new(mrc1_unfil, overwrite=True) as f1:
                    f1.set_data(weighted_emMap1)             
                    f1.voxel_size = tuple([sampling]*3)
                with mrcfile.new(mrc2_unfil, overwrite=True) as f2:
                    f2.set_data(weighted_emMap2) 
                    f2.voxel_size = tuple([sampling]*3) 

            # mean1_before =  emMap1.mean()                  
            # mean2_before =  emMap2.mean()  
            # std1_before =  emMap1.std()                  
            # std2_before =  emMap2.std()  
        
            # looking for pretrained model
            model = '%s/%s_it%s_half_class001_unfil.pt' %(dir,basename,beforeVar)
            print(model)
            if not os.path.isfile(model):
                print(f"first isonet reconstruction for {start_epochs} epochs, because previous iteration healpix order become {limit_healpix}")
                model = None
                epochs = start_epochs
            if retrain:
                print(f"retrain network each relion iteration")
                model = None
            if model is not None:
                print("reuse network from previous relion iteration")

            # train and predict
            if use_fsc_05:
                refine_limit_res = limit_resolution_05
            else:
                refine_limit_res = limit_resolution
            execute_deep(mrc1_unfil, mrc2_unfil, fscn, dir,  gpu, refine_limit_res, epochs = epochs, mask_file = mask_file, pretrained_model = model, batch_size = None, acc_batches=acc_batches, alpha=alpha, beta=beta)

            if awhiten:
                mrc_correct_backup1 = '%s/corrected_%s_it%s_half1_class001_unfil_backup.mrc' %(dir,basename,var)
                mrc_correct_backup2 = '%s/corrected_%s_it%s_half2_class001_unfil_backup.mrc' %(dir,basename,var)
                shutil.copy(mrc1_cor, mrc_correct_backup1)
                execute_angular_whiten(mrc1_cor,mrc1_cor,resolution_initial,limit_resolution)
                shutil.copy(mrc2_cor, mrc_correct_backup2)
                execute_angular_whiten(mrc2_cor,mrc2_cor,resolution_initial,limit_resolution)

            fsc_match = False
            if fsc_match and awhiten:
                with mrcfile.open(mrc1_cor) as f1:
                    awMap1 = f1.data.astype(np.float32).copy()  
                with mrcfile.open(mrc2_cor) as f2:
                    awMap2 = f2.data.astype(np.float32).copy() 
                with mrcfile.open(fscn) as f3:
                    fsc3d = f3.data.astype(np.float32).copy()    

                nz = emMap1.shape[0]
                low_r = nz*sampling/whitening_low
                high_r = nz*sampling/limit_resolution
                emMap1 = fsc_matching(awMap1, emMap1, fsc3d, low_r, high_r)
                emMap2 = fsc_matching(awMap2, emMap2, fsc3d, low_r, high_r)
                with mrcfile.new(mrc1_cor, overwrite=True) as f1:
                    f1.set_data(emMap1)             
                    f1.voxel_size = tuple([sampling]*3)
                with mrcfile.new(mrc2_cor, overwrite=True) as f2:
                    f2.set_data(emMap2) 
                    f2.voxel_size = tuple([sampling]*3) 

            do_fsc_weighting_after = False
            if do_fsc_weighting_after:
                with mrcfile.open(fscn, 'r') as mrc:
                    fsc_map = mrc.data
                with mrcfile.open(mrc1_cor) as f1:
                    emMap1 = f1.data.astype(np.float32).copy()  
                with mrcfile.open(mrc2_cor) as f2:
                    emMap2 = f2.data.astype(np.float32).copy() 
                from IsoNet.util.FSC import rotational_average, FSC_weighting
                fsc_curve = rotational_average(fsc_map)
                weighted_emMap1 = FSC_weighting(emMap1, fsc_curve, weight = False)
                weighted_emMap2 = FSC_weighting(emMap2, fsc_curve, weight = False)
                with mrcfile.new(mrc1_cor, overwrite=True) as f1:
                    f1.set_data(weighted_emMap1)             
                    f1.voxel_size = tuple([sampling]*3)
                with mrcfile.new(mrc2_cor, overwrite=True) as f2:
                    f2.set_data(weighted_emMap2) 
                    f2.voxel_size = tuple([sampling]*3) 

            with mrcfile.open(mrc1_cor) as d1:
                emDeep1 = d1.data.astype(np.float32).copy() 
            with mrcfile.open(mrc2_cor) as d2:
                emDeep2 = d2.data.astype(np.float32).copy()              

            #finalMap1 = emDeep1*float(std1_before)+mean1_before
            #finalMap2 = emDeep2*float(std2_before)+mean2_before
            finalMap1 = emDeep1
            finalMap2 = emDeep2

            
            #save mrcfile
            with mrcfile.new(mrc1_overwrite, overwrite=True) as fMap1:
                fMap1.set_data(finalMap1.astype(np.float32))
                fMap1.voxel_size = tuple([sampling]*3)
            with mrcfile.new(mrc2_overwrite, overwrite=True) as fMap2:
                fMap2.set_data(finalMap2.astype(np.float32))
                fMap2.voxel_size = tuple([sampling]*3)


            # lowpass again
            if lowpass:
                from subprocess import check_output
                s = f"relion_image_handler --i {mrc1_overwrite} --o tmp.mrc --lowpass {limit_resolution}; mv tmp.mrc {mrc1_overwrite}"
                check_output(s, shell=True)
                s = f"relion_image_handler --i {mrc2_overwrite} --o tmp.mrc --lowpass {limit_resolution}; mv tmp.mrc {mrc2_overwrite}"
                check_output(s, shell=True)

            print("finished IsoNet reconstruction")
            write_to_file(sync3)




        
            
