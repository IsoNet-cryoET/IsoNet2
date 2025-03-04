import json,sys
import logging
global logger 
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def filter_dict(obj_dict):
    """Filter out non-serializable elements from the object's __dict__."""
    filtered = {}
    for key, value in obj_dict.items():
        # Allow only JSON serializable types
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            filtered[key] = value
        elif isinstance(value, tuple):  
            filtered[key] = list(value)  # Convert tuples to lists for JSON
        elif isinstance(value, set):
            filtered[key] = list(value)  # Convert sets to lists for JSON
        # Skip functions, classes, and objects
    return filtered
          
def save_args_json(args,file_name):
    # filtered_dict = filter_dict(args.__dict__)
    encoded = json.dumps(args, indent=4, sort_keys=True,cls=NumpyEncoder)
    with open(file_name,'w') as f:
        f.write(encoded)

def load_args_from_json(file_name):
    with open(file_name,'r') as f:
        contents = f.read()
    encoded = json.loads(contents)
    return encoded

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