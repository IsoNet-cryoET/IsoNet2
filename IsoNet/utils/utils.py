import mrcfile
import logging
def mkfolder(folder, remove=True):
    import os
    try:
        os.makedirs(folder)
    except FileExistsError:
        if remove:
            logging.warning(f"The {folder} folder already exists. The old {folder} folder will be moved to {folder}~")
            import shutil
            if os.path.exists(folder+'~'):
                shutil.rmtree(folder+'~')
            os.system('mv {} {}'.format(folder, folder+'~'))
            os.makedirs(folder)
        else:
            logging.info(f"The {folder} folder already exists, outputs will write into this folder")

import os
import shutil

def create_folder(directory_name):
    # Check if the directory exists
    if os.path.exists(directory_name):
        # Rename the existing directory
        new_name = directory_name + '~'
        if os.path.exists(new_name):
            shutil.rmtree(new_name)
        shutil.move(directory_name, new_name)
        print(f"Directory '{directory_name}' already existed. Renamed to '{new_name}'.")
    
    # Create a new empty directory
    os.makedirs(directory_name)
    print(f"Created new empty directory '{directory_name}'.")


import mrcfile
def debug_matrix(mat, filename='debug.mrc'):
    print(mat.type())
    out_mat = mat.cpu().numpy().squeeze()
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(out_mat)

def process_gpuID(gpuID):
    if gpuID == None or gpuID == "None":
        import torch
        gpu_list = list(range(torch.cuda.device_count()))
        gpuID=','.join(map(str, gpu_list))
        print("using all GPUs in this node: %s" %gpuID)  
        ngpus = len(gpuID_list)

    if type(gpuID) == str:
        gpuID_list = list(set(gpuID.split(',')))
        gpuID_list = list(map(int,gpuID_list))
        ngpus = len(gpuID_list)
 
    elif type(gpuID) == tuple or type(gpuID) == list:
        gpuID_list = gpuID
        ngpus = len(gpuID)
        gpuID = ','.join(map(str, gpuID_list))

    elif type(gpuID) == int:
        ngpus = 1
        gpuID_list = [gpuID]
        gpuID = str(gpuID)

    import os    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpuID
    return ngpus, gpuID, gpuID_list

def process_ncpus(ncpus):
    #CPU
    ncpus = int(ncpus)
    from multiprocessing import cpu_count
    cpu_system = cpu_count()
    if cpu_system < ncpus:
        logging.info("requested number of cpus is more than the number of the cpu cores in the system")
        logging.info(f"setting ncpus to {cpu_system}")
        ncpus = cpu_system
    return ncpus

def process_batch_size(batch_size, ngpus):
    if batch_size == None or batch_size == "None":
        if ngpus == 1:
            batch_size = 4
        else:
            batch_size = 2 * ngpus


def parse_gpu(gpuID):
    import os
    if gpuID is None:
        import torch
        gpu_list = list(range(torch.cuda.device_count()))
        gpuID=','.join(map(str, gpu_list))
        print("using all GPUs in this node: %s" %gpuID)  

    ngpus, gpuID, gpuID_list = process_gpuID(gpuID)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpuID
    return ngpus, gpuID, gpuID_list

def parse_cpu(ncpus):
    #CPU
    from multiprocessing import cpu_count
    cpu_system = cpu_count()
    if cpu_system < ncpus:
        logging.info("requested number of cpus is more than the number of the cpu cores in the system")
        logging.info(f"setting ncpus to {cpu_system}")
        ncpus = cpu_system
    return ncpus






import mrcfile
def debug_matrix(mat, filename='debug.mrc'):
    out_mat = mat.cpu().detach().numpy().squeeze()
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(out_mat)

def process_gpuID(gpuID):

    if type(gpuID) == str:
        gpuID_list = list(set(gpuID.split(',')))
        gpuID_list = list(map(int,gpuID_list))
        ngpus = len(gpuID_list)
 
    elif type(gpuID) == tuple or type(gpuID) == list:
        gpuID_list = gpuID
        ngpus = len(gpuID)
        gpuID = ','.join(map(str, gpuID_list))

    elif type(gpuID) == int:
        ngpus = 1
        gpuID_list = [gpuID]
        gpuID = str(gpuID)
    
    return ngpus, gpuID, gpuID_list



def idx2list(tomo_idx):
    if tomo_idx is not None:
            if type(tomo_idx) is tuple:
                tomo_idx = list(map(str,tomo_idx))
            elif type(tomo_idx) is int:
                tomo_idx = [str(tomo_idx)]
            else:
                # tomo_idx = tomo_idx.split(',')
                txt=str(tomo_idx)
                txt=txt.replace(',',' ').split()
                tomo_idx=[]
                for everything in txt:
                    if everything.find("-")!=-1:
                        everything=everything.split("-")
                        for e in range(int(everything[0]),int(everything[1])+1):
                            tomo_idx.append(str(e))
                    else:
                        tomo_idx.append(str(everything))
    return tomo_idx




    
