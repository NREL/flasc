import numpy as np

from floris.tools import FlorisInterface
from flasc.examples.models import load_floris_smarteole

# Function from 
# https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
def nested_get(dic, keys):    
    for key in keys:
        print(key)
        print(dic)
        print()
        dic = dic[key]
        print(dic)
        print('------------')
    return dic

# Function from 
# https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
def nested_set(dic, keys, value, idx=None):
    dic_in = dic.copy()

    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    if idx is None:
        # Parameter is a scaler
        dic[keys[-1]] = value
    else:
        # dic[keys[-1]] = value
        # Parameter is a list

        # # Get the underlying list
        par_list = nested_get(dic_in, keys)
        print('x')
        print(par_list)
        print('x')
        # par_list[idx] = value
        # dic[keys[-1]] = par_list

def set_fi_param(fi_in, param, value, idx=None):

    fi_dict_mod = fi_in.floris.as_dict()


    nested_set(fi_dict_mod, param, value, idx)

    return FlorisInterface(fi_dict_mod)


if __name__ == "__main__":
    
    fi, _ = load_floris_smarteole(wake_model="emgauss")
    fi_dict_mod = fi.floris.as_dict()

    param = ['wake','wake_velocity_parameters','empirical_gauss',\
                'wake_expansion_rates']
    
    print(fi_dict_mod)

    fi_2 = set_fi_param(fi, param, 7777777, idx=0)

    print('----')

    print(fi_2.floris.as_dict())

