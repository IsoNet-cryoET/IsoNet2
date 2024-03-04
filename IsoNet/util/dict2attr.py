import json,sys
import logging
global logger 
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
global refine_param, predict_param, extract_param, param_to_check, param_to_set_attr

# refine_param = ['alpha','i','i2','mask','gpuID','ncpus','output_dir','limit_res','fsc_file','iterations','epochs','threshold','n_subvolume','crop_size','cube_size','mixed_precision','batch_size','acc_batches','learning_rate','predict_crop_size','cone_sampling_angle','pretrained_model']
# param_to_check = refine_param + ['self','run']
# param_to_set_attr = refine_param + ['iter_count','crop_size','cube_size','predict_cropsize','noise_dir','lr','ngpus','predict_batch_size','losses','metrics']
class Arg:
    def __init__(self,dictionary,from_cmd=True):
        for k, v in dictionary.items():
            if k not in param_to_check and from_cmd is True:
                logger.error("{} not recognized!".format(k))
                sys.exit(0)
            if k == 'gpuID' and type(v) is tuple:
                v = ','.join([str(i) for i in v])
            if k == 'noise_start_iter' and type(v) is int:
                v = tuple([v])
            if k == 'noise_level' and type(v) in [int,float]:
                v = tuple([v])
            if k in param_to_set_attr:
                setattr(self, k, v)
         
def save_args_json(args,file_name):
    filtered_dict = Arg(args.__dict__,from_cmd=False)
    encoded = json.dumps(filtered_dict.__dict__, indent=4, sort_keys=True)
    with open(file_name,'w') as f:
        f.write(encoded)

def load_args_from_json(file_name):
    with open(file_name,'r') as f:
        contents = f.read()
    encoded = json.loads(contents)
    return Arg(encoded,from_cmd=False)

def get_function_names(class_obj):
    # Use dir() to get all attributes of the class object
    all_attributes = dir(class_obj)
    # Filter out only the methods (functions) from the attributes
    method_names = [attr for attr in all_attributes if callable(getattr(class_obj, attr))]
    return method_names

def get_method_arguments(class_obj, method_name):
    import inspect
    method = getattr(class_obj, method_name, None)
    signature = inspect.signature(method)
    parameter_names = [param.name for param in signature.parameters.values()]
    return parameter_names

def check_parse(args_list):
    from IsoNet.bin.isonet import ISONET
    method_names = get_function_names(ISONET)
    
    if args_list[0] in method_names:
        check_list = get_method_arguments(ISONET, args_list[0])
        check_list.remove("self")
        check_list += ['help']
        first_letters = [word[0] for word in check_list]
        check_list += first_letters
    else:
        check_list = None

    if check_list is not None:
        for arg in args_list:
            if type(arg) is str and arg[0:2]=='--':
                if arg[2:] not in check_list:
                    logger.error(" '{}' not recognized!".format(arg[2:]))
                    sys.exit(0)


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

def txtval(txt):
    txt=str(txt)
    txt=txt.replace(',',' ').split()
    idx=[]
    for everything in txt:
        if everything.find("-")!=-1:
            everything=everything.split("-")
            for e in range(int(everything[0]),int(everything[1])+1):
                idx.append(e)
        else:
            idx.append(int(everything))
    return idx