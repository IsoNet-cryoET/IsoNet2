import mrcfile
import logging
import starfile
from tqdm import tqdm
import os
import linecache
import pathlib
    
def process_tomograms(star_path, output_dir, idx_str, desc, row_processor):
    linecache.clearcache()
    star_path = str(pathlib.Path(star_path).resolve())
    star = starfile.read(star_path)
    new_star = star.copy()
    idx_list = idx2list(idx_str, star.rlnIndex)
    os.makedirs(output_dir, exist_ok=True)

    with tqdm(total=len(idx_list), desc=desc, unit='tomogram') as pbar:
        for i, row in star.iterrows():
            if str(row.rlnIndex) in idx_list:
                print()
                row_processor(i, row, new_star)
                pbar.update(1)

    starfile.write(new_star, star_path)

def debug_matrix(mat, filename='debug.mrc'):
    if len(mat.shape) > 2:
        out_mat = mat.detach().cpu().numpy().squeeze()
        with mrcfile.new(filename, overwrite=True) as mrc:
            mrc.set_data(out_mat)

def process_gpuID(gpuID):
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    existing = os.environ.get("CUDA_VISIBLE_DEVICES")

    if gpuID == None or gpuID == "None":
        requested = None
    elif type(gpuID) == str:
        requested = [int(i.strip()) for i in gpuID.split(',') if i.strip()]
    elif type(gpuID) == tuple or type(gpuID) == list:
        requested = list(map(int, gpuID))
    elif type(gpuID) == int:
        requested = [gpuID]
    else:
        raise TypeError(f"Unsupported gpuID type: {type(gpuID).__name__}")

    if existing is None:
        import torch
        visible = [str(i) for i in range(torch.cuda.device_count())]
        if len(visible) == 0:
            raise ValueError("No CUDA GPUs detected on this system.")

        if requested is None:
            gpuID_list = visible
            logging.info("using all GPUs in this node: %s" % ','.join(gpuID_list))
        else:
            if min(requested) < 0 or max(requested) >= len(visible):
                raise ValueError(
                    f"Requested GPU index out of range. Got indices: {requested}; "
                    f"available indices: 0..{len(visible)-1}."
                )
            gpuID_list = [visible[i] for i in requested]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpuID_list)
    else:
        visible = [i.strip() for i in existing.split(',') if i.strip()]
        if len(visible) == 0:
            raise ValueError("CUDA_VISIBLE_DEVICES is set but empty.")

        if requested is None:
            gpuID_list = visible
        else:
            if min(requested) < 0 or max(requested) >= len(visible):
                raise ValueError(
                    f"Requested GPU index out of range for CUDA_VISIBLE_DEVICES={existing}. "
                    f"Got indices: {requested}; available indices: 0..{len(visible)-1}."
                )
            gpuID_list = [visible[i] for i in requested]
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpuID_list)

    gpuID = ','.join(gpuID_list)
    ngpus = len(gpuID_list)
    logging.info(f"using GPUs: {gpuID}")
    return ngpus, gpuID, gpuID_list

def process_batch_size(batch_size, ngpus):
    if batch_size in [None, "None", "auto"] :
        if ngpus == 1:
            batch_size = 4
        else:
            batch_size = 2 * ngpus
    return batch_size

def process_ncpus(ncpus):
    ncpus = int(ncpus)
    from multiprocessing import cpu_count
    cpu_system = cpu_count()
    if cpu_system < ncpus:
        logging.info("requested number of cpus is more than the number of the cpu cores in the system")
        logging.info(f"setting ncpus to {cpu_system}")
        ncpus = cpu_system

    return ncpus

def parse_params(batch_size_in, gpuID_in, ncpus_in, fit_ncpus_to_ngpus= False):
    ngpus, gpuID, gpuID_list = process_gpuID(gpuID_in)
    batch_size = process_batch_size(batch_size_in, ngpus)
    ncpus = process_ncpus(ncpus_in)
    if fit_ncpus_to_ngpus:
        n_workers = ncpus//ngpus
        if n_workers == 0:
            n_workers = 1
        ncpus = ngpus*n_workers
        logging.info(f"{n_workers} CPU cores per GPU, total {ncpus} CPUs")
    return batch_size, ngpus, ncpus



def idx2list(tomo_idx, all_tomo_idx):
    if tomo_idx not in  [None, "None", "all", "All"]:
        if type(tomo_idx) is tuple:
            tomo_idx = list(map(str,tomo_idx))
        elif type(tomo_idx) is int:
            tomo_idx = [str(tomo_idx)]
        else:
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
    else:
        tomo_idx = [str(i) for i in all_tomo_idx]
    return tomo_idx




    
