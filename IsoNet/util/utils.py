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