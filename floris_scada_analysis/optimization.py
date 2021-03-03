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
import scipy.stats as spst


def find_bias_x(x_1, y_1, x_2, y_2, search_range, search_dx):
    x_1 = np.array(x_1)
    y_2 = np.array(y_2)

    def errfunc(dx, x_1, x_2, y_1, y_2):
        y_1_cor = np.interp(x_2, x_1 - dx, y_1)
        y_1_isnan = np.isnan(y_1_cor)
        y_2_isnan = np.isnan(y_2)        

        # Clean up data
        clean_data = [~a and ~b for a, b in zip(y_1_isnan, y_2_isnan)]
        y_1_cor = y_1_cor[clean_data]
        y_2 = y_2[clean_data]

        if all(np.isnan(y_1_cor)) and all(np.isnan(y_2)):
            cost = np.nan
        else:
            cost = -1.0 * spst.pearsonr(y_1_cor, y_2)[0]
        return cost

    cost_min = 1.0e15
    success = False
    p1 = [0.0]
    for dx_opt in np.arange(search_range[0], search_range[1], search_dx):
        cost_eval = errfunc(dx_opt, x_1, x_2, y_1, y_2)
        if cost_eval <= cost_min:
            p1 = [dx_opt]
            cost_min = cost_eval
            success = True
    dx_opt = p1[0]

    return dx_opt, success