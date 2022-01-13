# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import numpy as np
import os
import pandas as pd


def batch_load_data(fn_path):
    ii = 0
    df_list = []
    while os.path.exists(fn_path + ".%d" % ii):
        fn_path_ii = fn_path + ".%d" % ii
        print("Found file %s. Loading..." % os.path.basename(fn_path_ii))
        df_list.append(pd.read_feather(fn_path_ii))
        ii += 1
    print("Loaded %d files. Concatenating." % ii)
    return pd.concat(df_list)


def batch_save_data(df, fn_path, no_rows_per_file=10000):
    N = df.shape[0]
    if 'time' in df.columns:
        df = df.reset_index(drop=True).copy()
    else:
        df = df.reset_index(drop=False).copy()

    splits = np.arange(0, N - 1, no_rows_per_file)
    splits = np.append(splits, N - 1)
    splits = np.unique(splits)
    for ii in range(len(splits) - 1):
        lb = splits[ii]
        ub = splits[ii+1]
        df_subset = df[lb:ub].reset_index(drop=True).copy()
        fn_path_ii = fn_path + ".%d" % ii
        print("Saving file to %s." % fn_path_ii)
        df_subset.to_feather(fn_path_ii)
