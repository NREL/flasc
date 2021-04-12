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


def find_sensor_stuck(measurement_array,
                      no_consecutive_measurements=6,
                      stddev_threshold=0.05,
                      index_array=None):

    # Create index array, if unspecified
    N = len(measurement_array)
    if index_array is None:
        index_array = np.array(range(N))

    # Ensure variable types
    index_array = np.array(index_array)
    measurement_array = np.array(measurement_array)

    # Remove nans from measurement array
    index_array = index_array[~np.isnan(measurement_array)]
    measurement_array = measurement_array[~np.isnan(measurement_array)]

    def format_array(array_in, row_length):
        array_in = np.array(array_in)
        Nm = row_length - 1
        C = array_in[0:-Nm]
        for ii in range(1, Nm):
            C = np.vstack([C, array_in[ii:-Nm+ii]])
        C = np.vstack([C, array_in[Nm::]]).T
        return C

    Cindex = format_array(index_array,
                          row_length=no_consecutive_measurements)
    Cmeas = format_array(measurement_array,
                         row_length=no_consecutive_measurements)

    # Get standard deviations and determine faults
    std_array = np.std(Cmeas, axis=1)
    indices_faulty = np.unique(Cindex[std_array < stddev_threshold, :])

    print('Found %d faulty measurements. This is %.3f %% of your dataset.'
          % (len(indices_faulty), 100*len(indices_faulty)/N))

    return indices_faulty
