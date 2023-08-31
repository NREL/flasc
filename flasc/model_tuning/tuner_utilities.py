import numpy as np

from typing import Any, Dict, List, Optional, Tuple, Union

from floris.tools import FlorisInterface



def nested_get(dic: Dict[str, Any],
                 keys: List[str]) -> Any:
    """Get a value from a nested dictionary using a list of keys.
    Based on: stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys

    Args:
        dic (Dict[str, Any]): The dictionary to get the value from.
        keys (List[str]): A list of keys to traverse the dictionary.

    Returns:
        Any: The value at the end of the key traversal.
    """
    for key in keys:
        dic = dic[key]
    return dic

def nested_set(dic: Dict[str, Any], 
                keys: List[str], 
                value: Any, 
                idx: Optional[int] = None) -> None:
    """Set a value in a nested dictionary using a list of keys.
    Based on: stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys

    Args:
        dic (Dict[str, Any]): The dictionary to set the value in.
        keys (List[str]): A list of keys to traverse the dictionary.
        value (Any): The value to set.
        idx (Optional[int], optional): If the value is an list, the index to change.
         Defaults to None.
    """
    dic_in = dic.copy()

    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    if idx is None:
        # Parameter is a scaler, set directly
        dic[keys[-1]] = value
    else:
        # Parameter is a list, need to first get the list, change the values at idx

        # # Get the underlying list
        par_list = nested_get(dic_in, keys)
        par_list[idx] = value
        dic[keys[-1]] = par_list


def set_fi_param(fi_in: FlorisInterface, 
        param: List[str], 
        value: Any, 
        idx: Optional[int] = None) -> FlorisInterface:
    """Set a parameter in a FlorisInterface object.

    Args:
        fi_in (FlorisInterface): The FlorisInterface object to modify.
        param (List[str]): A list of keys to traverse the FlorisInterface dictionary.
        value (Any): The value to set.
        idx (Optional[int], optional): The index to set the value at. Defaults to None.

    Returns:
        FlorisInterface: The modified FlorisInterface object.
    """
    fi_dict_mod = fi_in.floris.as_dict()
    nested_set(fi_dict_mod, param, value, idx)
    return FlorisInterface(fi_dict_mod)


from flasc.utilities_examples import load_floris_smarteole
if __name__ == "__main__":
    
    fi, _ = load_floris_smarteole(wake_model="emgauss")
    fi_dict_mod = fi.floris.as_dict()

    param = ['wake','wake_velocity_parameters','empirical_gauss',\
                'wake_expansion_rates']
    

    fi_2 = set_fi_param(fi, param, 7777777, idx=1)

    print(fi_dict_mod)
    print('******')
    print(fi_2.floris.as_dict())


