# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#TODO: Do we want to follow FLORIS' method of keeping 3 dimensions for all matrices?

class TableAnalysis():

    def __init__(self, 
                 wd_edges = np.arange(0,  360 + 2.0, 2.0),
                 ws_edges = np.arange(0 - 0.5, 50, 1.0),
                 pow_ref_edges = np.arange(0,  15000 + 100.0, 100.0), #kW
                 use_pow_ref = False,
                 minutes_per_point=10.0,):
        
        # Check that the inputs are valid 
        if len(ws_edges) < 2:
            raise ValueError("ws_edges must have at least 2 elements")
        if len(wd_edges) < 2:
            raise ValueError("wd_edges must have at least 2 elements")
        if len(pow_ref_edges) < 2:
            raise ValueError("pow_ref_edges must have at least 2 elements")
        
        # Save the inputs
        self.wd_edges = wd_edges
        self.ws_edges = ws_edges
        self.pow_ref_edges = pow_ref_edges
        self.use_pow_ref = use_pow_ref
        self.minutes_per_point = minutes_per_point

        # Set the wind speed and wind direction and pow_ref bin centers
        self.wd_bin_centers = (self.wd_edges[:-1]+self.wd_edges[1:])/2
        self.ws_bin_centers = (self.ws_edges[:-1]+self.ws_edges[1:])/2
        self.pow_ref_bin_centers = (self.pow_ref_edges[:-1]+
                                    self.pow_ref_edges[1:])/2
    
        # Depending on use_pow_ref, set the number of bins
        if self.use_pow_ref:
            self.n_pow_ref_bins = len(self.pow_ref_bin_centers)
        else:
            self.n_ws_bins = len(self.ws_bin_centers)

        # Save the number of wind speed and wind direction bins
        self.n_wd_bins = len(self.wd_bin_centers)

        # Intialize list of matrices
        # Organize results in 3D matrix of wind speed, wind direction, and turbine
        # Put in lists for each dataframe
        self.mean_matrix_list = [] # Mean power per wind speed and wind direction and turbine bin
        self.median_matrix_list = [] # Median power per wind speed and wind direction and turbine bin
        self.count_matrix_list = []  # Count of power per wind speed and wind direction and turbine bin
        self.std_matrix_list = [] # Standard deviation of power per wind speed and wind direction and turbine bin
        self.se_matrix_list = [] # Standard error of power per wind speed and wind direction and turbine bin
        self.mu_matrix_list = [] # Guassian mu value3

        # Initialize user_defined_frequency_matrix to None
        self.user_defined_frequency_matrix = None

        # Initialize the number of cases to 0
        self.n_cases = 0

        # Initialize df list
        self.case_df_list = []
        self.case_names = []


    def add_df(self, case_df, case_name):

        # Make a copy of the dataframe
        df = case_df.copy()

        # Check that the dataframe has the correct columns
        if 'wd' not in df.columns:
            raise ValueError("Dataframe must have a column named 'wd'")
        if (not self.use_pow_ref) and 'ws' not in df.columns:
            raise ValueError("Dataframe must have a column named 'ws' when use_pow_ref is False")
        if self.use_pow_ref and 'pow_ref' not in df.columns:
            raise ValueError("Dataframe must have a column named 'pow_ref' when use_pow_ref is True")
        #TODO: Check that there is a pow column
        for c in df.columns:
            if self.use_pow_ref:
                if not ('wd' in c or 'pow_' in c or 'pow_ref' in c):
                        raise ValueError("Dataframe must only have columns named 'wd', 'ws', or 'pow_*'")
                if 'ws' in c:
                    raise ValueError("Dataframe must not have a column named 'ws' when use_pow_ref is True")
            else:
                if not ('wd' in c or 'ws' in c or 'pow_' in c):
                    raise ValueError("Dataframe must only have columns named 'wd', 'ws', or 'pow_*'")
                if 'pow_ref' in c:
                    raise ValueError("Dataframe must not have a column named 'pow_ref' when use_pow_ref is False")
        
        # Bin the wind speed and wind direction and drop original data
        if self.use_pow_ref:
            df['pow_ref_bin'] = pd.cut(df['pow_ref'], self.pow_ref_edges,labels=self.pow_ref_bin_centers)
            df = df.drop(columns=['pow_ref'])
        else:
            df['ws_bin'] = pd.cut(df['ws'], self.ws_edges,labels=self.ws_bin_centers)
            df = df.drop(columns=['ws'])

        df['wd_bin'] = pd.cut(df['wd'], self.wd_edges,labels=self.wd_bin_centers)
        df = df.drop(columns=['wd'])

        # Convert the turbine powers to a new column
        if self.use_pow_ref:
            df = df.melt(id_vars=['wd_bin', 'pow_ref_bin'], var_name='turbine', value_name='power')
        else:
            df = df.melt(id_vars=['wd_bin', 'ws_bin'], var_name='turbine', value_name='power')

        # Get a list of unique turbine names
        turbine_names = sorted(df['turbine'].unique())

        # Determine the number of turbines
        self.n_turbines = len(turbine_names)

        # Convert ws_bin and wd_bin to categorical with levels set to the bin centers
        # This enforces that the order of the bins is correct and the size of the matrix
        # matches the number of bins
        if self.use_pow_ref:
            df['pow_ref_bin'] = pd.Categorical(df['pow_ref_bin'], categories=self.pow_ref_bin_centers)
        else:
            df['ws_bin'] = pd.Categorical(df['ws_bin'], categories=self.ws_bin_centers)
        df['wd_bin'] = pd.Categorical(df['wd_bin'], categories=self.wd_bin_centers)
        
        df['turbine'] = pd.Categorical(df['turbine'], categories=turbine_names)

        #Save the dataframe and name
        self.case_df_list.append(df)
        self.case_names.append(case_name)

        # increment the number of cases
        self.n_cases += 1

        # # Get a matrix of mean values with dimensions: wd, ws|pow_ref,turbine whose shape
        # # is (len(wd_bin_centers), len(ws_bin_centers) | len(pow_ref_bin_centers), n_turbines)
        if self.use_pow_ref:
            mean_matrix = df.groupby(['wd_bin', 'pow_ref_bin', 'turbine'])['power'].mean().reset_index().power.to_numpy()
            mean_matrix = mean_matrix.reshape((len(self.wd_bin_centers), len(self.pow_ref_bin_centers), self.n_turbines))
        else:
            mean_matrix = df.groupby(['wd_bin', 'ws_bin', 'turbine'])['power'].mean().reset_index().power.to_numpy()
            mean_matrix = mean_matrix.reshape((len(self.wd_bin_centers), len(self.ws_bin_centers), self.n_turbines))

        # # Get a matrix of median values with dimensions: wd, ws|pow_ref,turbine whose shape
        # # is (len(wd_bin_centers), len(ws_bin_centers) | len(pow_ref_bin_centers), n_turbines)
        if self.use_pow_ref:
            median_matrix = df.groupby(['wd_bin', 'pow_ref_bin', 'turbine'])['power'].median().reset_index().power.to_numpy()
            median_matrix = median_matrix.reshape((len(self.wd_bin_centers), len(self.pow_ref_bin_centers), self.n_turbines))
        else:
            median_matrix = df.groupby(['wd_bin', 'ws_bin', 'turbine'])['power'].median().reset_index().power.to_numpy()
            median_matrix = median_matrix.reshape((len(self.wd_bin_centers), len(self.ws_bin_centers), self.n_turbines))

        # # Get a matrix of count values with dimensions: wd, ws|pow_ref,turbine whose shape
        # # is (len(wd_bin_centers), len(ws_bin_centers) | len(pow_ref_bin_centers), n_turbines)
        if self.use_pow_ref:
            count_matrix = df.groupby(['wd_bin', 'pow_ref_bin', 'turbine'])['power'].count().reset_index().power.to_numpy()
            count_matrix = count_matrix.reshape((len(self.wd_bin_centers), len(self.pow_ref_bin_centers), self.n_turbines))
        else:
            count_matrix = df.groupby(['wd_bin', 'ws_bin', 'turbine'])['power'].count().reset_index().power.to_numpy()
            count_matrix = count_matrix.reshape((len(self.wd_bin_centers), len(self.ws_bin_centers), self.n_turbines))
        


        #TODO For these matrices are what we want to build the new uncertainty bands off of but
        # I'm not sure yet how
        # Option 1: Is something like assume each bin has an uncertainty band defined by a gaussian however
        #    - assume some maximum value until a minimun number of points are in the bin

        # TODO For now just calculated the values likely used in such a formula

        # # Get a matrix of std values with dimensions: wd, ws|pow_ref,turbine whose shape
        # # is (len(wd_bin_centers), len(ws_bin_centers) | len(pow_ref_bin_centers), n_turbines)
        if self.use_pow_ref:
            std_matrix = df.groupby(['wd_bin', 'pow_ref_bin', 'turbine'])['power'].std().reset_index().power.to_numpy()
            std_matrix = std_matrix.reshape((len(self.wd_bin_centers), len(self.pow_ref_bin_centers), self.n_turbines))
        else:
            std_matrix = df.groupby(['wd_bin', 'ws_bin', 'turbine'])['power'].std().reset_index().power.to_numpy()
            std_matrix = std_matrix.reshape((len(self.wd_bin_centers), len(self.ws_bin_centers), self.n_turbines))


        # # Get a matrix of standard error values with dimensions: ws, wd, turbine whose shape
        # # is (len(wd_bin_centers), len(ws_bin_centers), n_turbines)
        se_matrix = std_matrix / np.sqrt(count_matrix)
        if self.use_pow_ref:
            se_matrix = se_matrix.reshape((len(self.wd_bin_centers), len(self.pow_ref_bin_centers), self.n_turbines))
        else:
            se_matrix = se_matrix.reshape((len(self.wd_bin_centers), len(self.ws_bin_centers), self.n_turbines))

        #TODO How to calculate mu?
        mu_matrix = np.zeros_like(mean_matrix)
        
        # Add the matrices to the list of matrices
        self.mean_matrix_list.append(mean_matrix)
        self.median_matrix_list.append(median_matrix)
        self.count_matrix_list.append(count_matrix)
        self.std_matrix_list.append(std_matrix)
        self.se_matrix_list.append(se_matrix)
        self.mu_matrix_list.append(mu_matrix)
        
        # # Fill missing values of the confidence interval matrix with the maximum value
        # #TODO: I don't know if this is correct my thinking is that if there is no data
        # # then the confidence interval is the maximum confidence interval
        # ci_matrix[np.isnan(ci_matrix)] = np.nanmax(ci_matrix)
        # self.ci_matrix_list.append(ci_matrix)

        # Update frequency matrix
        # TODO: Originaly I thought it made sense to update frequency matrix here
        # but now I think you need to know which turbine is being assessed so that
        # the matrix is 0 for bins where that turbine is missing values
        # self.update_frequency_matrix()


    def print_df_names(self):

        # Print the list of dataframes
        print('Dataframes in list:')
        for ii, df in enumerate(self.case_df_list):
            print('  %d: %s' % (ii, self.case_names[ii]))

    def get_overall_frequency_matrix(self):

        # Get the total matrix
        df_total = pd.concat(self.case_df_list)

        # Add a dummpy variable
        df_total['dummy'] = 1
        
        # Get a matrix of count values with dimensions: ws, wd | pow_ref
        if self.use_pow_ref:
            overall_frequency_matrix = df_total.groupby(['wd_bin', 'pow_ref_bin'])['dummy'].sum().reset_index().fillna(0).dummy.to_numpy()
            overall_frequency_matrix = overall_frequency_matrix.reshape((len(self.wd_bin_centers), len(self.pow_ref_bin_centers)))
        else:
            overall_frequency_matrix = df_total.groupby(['wd_bin', 'ws_bin'])['dummy'].sum().reset_index().fillna(0).dummy.to_numpy()
            overall_frequency_matrix = overall_frequency_matrix.reshape((len(self.wd_bin_centers), len(self.ws_bin_centers)))

        # normalize the frequency matrix
        overall_frequency_matrix = overall_frequency_matrix / np.sum(overall_frequency_matrix)

        return overall_frequency_matrix
    
    def get_uniform_frequency_matrix(self):

        # Get a matrix of count values with dimensions: ws, wd | pow_ref
        if self.use_pow_ref:
            uniform_frequency_matrix = np.ones((len(self.wd_bin_centers), len(self.pow_ref_bin_centers)))
        else:
            uniform_frequency_matrix = np.ones((len(self.wd_bin_centers), len(self.ws_bin_centers)))

        # normalize the frequency matrix
        uniform_frequency_matrix = uniform_frequency_matrix / np.sum(uniform_frequency_matrix)

        return uniform_frequency_matrix
    
    def set_user_defined_frequency_matrix(self, user_defined_frequency_matrix):

        # Confirm user_defined_frequency_matrix is a numpy array
        if not isinstance(user_defined_frequency_matrix, np.ndarray):
            raise ValueError('User defined frequency matrix must be a numpy array')

        # Confirm user_defined_frequency_matrix has dimensions: 
        # wd, ws | pow_ref
        if self.use_pow_ref:
            if user_defined_frequency_matrix.shape != (len(self.wd_bin_centers), len(self.pow_ref_bin_centers)):
                raise ValueError(f'User defined frequency matrix has incorrect dimensions, should be {len(self.wd_bin_centers)} x {len(self.pow_ref_bin_centers)}')
        else:
            if user_defined_frequency_matrix.shape != (len(self.wd_bin_centers), len(self.ws_bin_centers)):
                raise ValueError(f'User defined frequency matrix has incorrect dimensions, should be {len(self.wd_bin_centers)} x {len(self.ws_bin_centers)}')

        # normalize the frequency matrix
        user_defined_frequency_matrix = user_defined_frequency_matrix / np.sum(user_defined_frequency_matrix)

        self.user_defined_frequency_matrix = user_defined_frequency_matrix

    def get_user_defined_frequency_matrix(self):

        # Check that user defined frequency matrix has been set
        if self.user_defined_frequency_matrix is None:
            raise ValueError('User defined frequency matrix has not been set')

        return self.user_defined_frequency_matrix

    def get_turbine_availability_mask(self, turbine, min_points_per_bin=1):

        " Get a mask for a given turbine if each matrix has at least min_points_per_bin points in each bin"

        # turbine_availability_mask is true if each matrix has at least N points in each bin
        # Dimensions of turbine_mast are num_wd_bins x num_ws_bins
        # turbine_availability_mask = (np.min(self.count_matrix_list[:], axis=0)[:, :, turbine].squeeze() >= min_points_per_bin)
        turbine_availability_mask = (np.min(self.count_matrix_list[:], axis=0)[:, :, turbine] >= min_points_per_bin)

        return turbine_availability_mask

    def get_turbine_frequency_matrix(self, turbine):

        # turbine_frequency_matrix is the sum of the count matrix over all time
        # turbine_frequency_matrix = np.sum(self.count_matrix_list[:], axis=0)[:, :, turbine].squeeze()
        turbine_frequency_matrix = np.sum(self.count_matrix_list[:], axis=0)[:, :, turbine]

        # Normalize the frequency matrix
        turbine_frequency_matrix = turbine_frequency_matrix / np.sum(turbine_frequency_matrix)

        return turbine_frequency_matrix
    
    def get_frequency_matrix(self, frequency_matrix_type, turbine=None):

        # Get the frequency matrix
        if frequency_matrix_type == 'turbine':
            if turbine is None:
                raise ValueError('turbine must be specified if frequency_matrix_type is "turbine"')
            frequency_matrix = self.get_turbine_frequency_matrix(turbine)
        elif frequency_matrix_type == 'overall':
            frequency_matrix = self.get_overall_frequency_matrix()
        elif frequency_matrix_type == 'uniform':
            frequency_matrix = self.get_uniform_frequency_matrix()
        elif frequency_matrix_type == 'user_defined':
            frequency_matrix = self.get_user_defined_frequency_matrix()
        else:
            raise ValueError('frequency_matrix_type must be either "turbine", "overall", "uniform", or "user_defined"')
        
        return frequency_matrix

    def compute_expected_turbine_power(self, 
                            turbine, 
                            wd_range=None,
                            ws_range=None, 
                            pow_ref_range=None,
                            condition_on=None,
                            min_points_per_bin=1,
                            mean_or_median='mean',
                            frequency_matrix_type='turbine'
                            ):
        """Calculate the expected power.

        Args:

        Returns:
            _type_: _description_
        """            

        #TODO I think this right now returns the average power so needs to be scaled up
        # by number of hours to get energy

        # Check that condition_on is consisten with self.use_pow_ref
        if condition_on == 'pow_ref' and not self.use_pow_ref:
            raise ValueError('condition_on cannot be "pow_ref" if self.use_pow_ref is False')
        if condition_on == 'ws' and self.use_pow_ref:
            raise ValueError('condition_on cannot be "ws" if self.use_pow_ref is True')

        # Set the default values for the wind speed and wind direction bins
        if wd_range is None:
            wd_range = (self.wd_bin_centers[0], self.wd_bin_centers[-1])
        if self.use_pow_ref:
            if pow_ref_range is None:
                pow_ref_range = (self.pow_ref_bin_centers[0], self.pow_ref_bin_centers[-1])
        else:
            if ws_range is None:
                ws_range = (self.ws_bin_centers[0], self.ws_bin_centers[-1])

        # Get the base frequency matrix
        frequency_matrix = self.get_frequency_matrix(frequency_matrix_type, turbine)

        # Get the axis for summing over, initialize expected_value size
        if condition_on == None:
            sum_axis = None # Sum over both wind direction and wind speed | pow_ref
            expand_dims = (0, 1)
            expected_value = np.zeros((self.n_cases, 1, 1))
        elif condition_on == "wd":
            sum_axis = 1 # Sum over wind speeds
            expand_dims = (1)
            expected_value = np.zeros((self.n_cases, self.n_wd_bins, 1))
        elif (condition_on == "ws") | (condition_on == "pow_ref"):
            sum_axis = 0 # Sum over wind directions
            expand_dims = (0)
            if self.use_pow_ref:
                expected_value = np.zeros((self.n_cases, 1, self.n_pow_ref_bins))
            else:
                expected_value = np.zeros((self.n_cases, 1, self.n_ws_bins))
        else:
            raise ValueError("condition_on must be either 'wd', 'ws','pow_ref, or None.")

        # Get the base frequency matrix
        frequency_matrix = self.get_frequency_matrix(frequency_matrix_type, turbine)
        
        # Get indices of the wind speed and wind direction bins that are outside the range
        wd_ind_outside = np.where((self.wd_bin_centers < wd_range[0]) | 
                            (self.wd_bin_centers > wd_range[1]))[0]
        if self.use_pow_ref:
            pow_ref_ind_outside = np.where((self.pow_ref_bin_centers < pow_ref_range[0]) | 
                                      (self.pow_ref_bin_centers > pow_ref_range[1]))[0]
        else:
            ws_ind_outside = np.where((self.ws_bin_centers < ws_range[0]) | 
                                      (self.ws_bin_centers > ws_range[1]))[0]



        # Set the value to 0 for all bins that are not inside the range
        frequency_matrix[wd_ind_outside, :] = 0
        if self.use_pow_ref:
            frequency_matrix[:, pow_ref_ind_outside] = 0
        else:
            frequency_matrix[:, ws_ind_outside] = 0
        

        # Check that the frequency matrix is not all zeros
        if np.sum(frequency_matrix) == 0:
            raise ValueError('Frequency matrix is all zeros')

        # Re-normalize the frequency matrix, taking into account conditionals
        # np.divide allows us to handle 0/0 cases
        frequency_matrix = np.divide(frequency_matrix,
            np.sum(frequency_matrix, axis=sum_axis, keepdims=True), 
            out=np.zeros_like(frequency_matrix),
            where=frequency_matrix!=0
        ) 
        
        for i in range(self.n_cases):

            # Get the mean matrix for the wind speed and wind direction bins
            if mean_or_median == 'mean':
                power_matrix = self.mean_matrix_list[i][:, :, turbine].copy()
            elif mean_or_median == 'median':
                power_matrix = self.median_matrix_list[i][:, :, turbine].copy()
            else:
                raise ValueError('mean_or_median must be either "mean" or "median"')

            # Set to 0 all bins where the turbine availability mask is false
            turbine_availability_mask = self.get_turbine_availability_mask(turbine, min_points_per_bin=min_points_per_bin)

            power_matrix[~turbine_availability_mask] = np.nan # 0.0 #
            # Is this necessary? These are NaNs anyway, aren't they?

            # Check for NaNs in the power matrix
            product_matrix = power_matrix * frequency_matrix
            expected_value_temp = np.nansum(product_matrix, axis=sum_axis)

            if (condition_on == None):
                if (np.isnan(product_matrix).all(axis=sum_axis)):
                    expected_value_temp = np.nan
            else:
                nan_incidences = np.isnan(product_matrix).all(axis=sum_axis)
                expected_value_temp[nan_incidences] = np.nan

            expected_value[i, :, :] =  np.expand_dims(expected_value_temp, axis=expand_dims)
         
        return expected_value

    def compute_expected_power_across_turbines(self, 
                            turbine_list=None, 
                            wd_range=None,
                            ws_range=None, 
                            pow_ref_range=None,
                            condition_on=None,
                            min_points_per_bin=1,
                            mean_or_median='mean',
                            frequency_matrix_type='turbine'
                            ):
        """Calculate the energy in a range of wind speed and wind direction across the turbines
        in turbine list
        """

        # Check that condition_on is consisten with self.use_pow_ref
        if condition_on == 'pow_ref' and not self.use_pow_ref:
            raise ValueError('condition_on cannot be "pow_ref" if self.use_pow_ref is False')
        if condition_on == 'ws' and self.use_pow_ref:
            raise ValueError('condition_on cannot be "ws" if self.use_pow_ref is True')

        # If self.use_pow_ref is True, and the value of ws_range is not None, raise an error
        if self.use_pow_ref and ws_range is not None:
            raise ValueError('ws_range cannot be specified if self.use_pow_ref is True')
        
        # If self.use_pow_ref is False, and the value of pow_ref_range is not None, raise an error
        if not self.use_pow_ref and pow_ref_range is not None:
            raise ValueError('pow_ref_range cannot be specified if self.use_pow_ref is False')

        # Check that turbine_list is a list
        turbine_list = self._check_turbine_list(turbine_list)
            
        n_turbs = len(turbine_list)

        # Get the axis for summing over, initialize expected_value size
        if condition_on == None:
            expected_value_each = np.zeros((self.n_cases, 1, 1, n_turbs))
        elif condition_on == "wd":
            expected_value_each = np.zeros((self.n_cases, self.n_wd_bins, 1, n_turbs))
        elif condition_on == "ws":
            expected_value_each = np.zeros((self.n_cases, 1, self.n_ws_bins, n_turbs))
        elif condition_on == "pow_ref":
            expected_value_each = np.zeros((self.n_cases, 1, self.n_pow_ref_bins, n_turbs))
        else:
            raise ValueError("condition_on must be either 'wd', 'ws', 'pow_ref', or None.")
        
        for idx, t_i in enumerate(turbine_list):
            
            expected_value_each[:, :, :, idx] = self.compute_expected_turbine_power( 
                            turbine=t_i, 
                            ws_range=ws_range, 
                            wd_range=wd_range,
                            pow_ref_range=pow_ref_range,
                            condition_on=condition_on,
                            min_points_per_bin=min_points_per_bin,
                            mean_or_median=mean_or_median,
                            frequency_matrix_type=frequency_matrix_type
                            )

        # This code is more strict
        expected_value_total = np.sum(expected_value_each, axis=3)
 

        return expected_value_total

    # Redifine some wrappers for convenience
    def _check_turbine_list(self, turbine_list):
        
        if isinstance(turbine_list, int):
            turbine_list = [turbine_list]

        if turbine_list is not None:
            if not isinstance(turbine_list, list) and not isinstance(turbine_list, np.ndarray):
                raise ValueError('turbine_list must be a list')

        if turbine_list is None:
            turbine_list = list(range(self.n_turbines))

        if np.max(turbine_list) >= self.n_turbines:
            raise ValueError('turbine_list contains a turbine index that is greater than the number of turbines')

        return turbine_list

    def _case_index(self, case):
        if isinstance(case, int):
            return case
        elif isinstance(case, str):
            return self.case_names.index(case)

    def get_energy_in_range(self,
                            turbine_list=None,
                            wd_range=None,
                            ws_range=None,
                            pow_ref_range=None,
                            min_points_per_bin=1,
                            mean_or_median='mean',
                            frequency_matrix_type='turbine'
        ):

        energy = self.compute_expected_power_across_turbines(
            turbine_list=self._check_turbine_list(turbine_list),  
            wd_range=wd_range,
            ws_range=ws_range,
            pow_ref_range=pow_ref_range,
            condition_on=None,
            min_points_per_bin=min_points_per_bin,
            mean_or_median=mean_or_median,
            frequency_matrix_type=frequency_matrix_type
        )[:,0,0]# .squeeze()

        return energy

    def get_energy_per_wd_bin(self,
                              turbine_list=None,
                              ws_range=None,
                              pow_ref_range=None,
                              min_points_per_bin=1,
                              mean_or_median='mean',
                              frequency_matrix_type='turbine'):
        """Calculate the energy in a range of wind speed and wind direction across the turbines
        in turbine list
        """

        energy_per_wd_bin = self.compute_expected_power_across_turbines(
            turbine_list=self._check_turbine_list(turbine_list),  
            wd_range=None,
            ws_range=ws_range,
            pow_ref_range=pow_ref_range,
            condition_on="wd",
            min_points_per_bin=min_points_per_bin,
            mean_or_median=mean_or_median,
            frequency_matrix_type=frequency_matrix_type
        )[:,:,0]# .squeeze()

        return energy_per_wd_bin
    
    def get_energy_per_ws_bin(self,
                              turbine_list=None,
                              wd_range=None,
                              min_points_per_bin=1,
                              mean_or_median='mean',
                              frequency_matrix_type='turbine'):
        """Calculate the energy in a range of wind speed and wind direction across the turbines
        in turbine list
        """
        energy_per_ws_bin = self.compute_expected_power_across_turbines( 
            turbine_list=self._check_turbine_list(turbine_list), 
            wd_range=wd_range,
            ws_range=None, 
            pow_ref_range=None,
            condition_on="ws",
            min_points_per_bin=min_points_per_bin,
            mean_or_median=mean_or_median,
            frequency_matrix_type=frequency_matrix_type
        )[:,0,:]# .squeeze()

        return energy_per_ws_bin
    
    def get_energy_per_pow_ref_bin(self,
                              turbine_list=None,
                              wd_range=None,
                              min_points_per_bin=1,
                              mean_or_median='mean',
                              frequency_matrix_type='turbine'):
        """Calculate the energy in a range of wind speed and wind direction across the turbines
        in turbine list
        """
        energy_per_pow_ref_bin = self.compute_expected_power_across_turbines( 
            turbine_list=self._check_turbine_list(turbine_list), 
            wd_range=wd_range,
            ws_range=None, 
            pow_ref_range=None,
            condition_on="pow_ref",
            min_points_per_bin=min_points_per_bin,
            mean_or_median=mean_or_median,
            frequency_matrix_type=frequency_matrix_type
        )[:,0,:]# .squeeze()

        return energy_per_pow_ref_bin
    
    # Plot energy by wd bin
    def plot_energy_by_wd_bin(self, 
                                turbine_list=None, 
                                ws_range=None,
                                pow_ref_range=None,
                                min_points_per_bin=1,
                                mean_or_median='mean',
                                frequency_matrix_type='turbine',
                                ax=None, 
                                labels=None,
                                **kwargs):
        
        # Check if turbine list is a scalar
        if turbine_list is not None:
            if not isinstance(turbine_list, list) and not isinstance(turbine_list, np.ndarray):
                turbine_list = [turbine_list]
        
        # Check if labels are None, and if not that they have the right length
        if labels is None:
            labels = self.case_names
        else:
            if len(labels) != len(self.case_names):
                raise ValueError('labels must have the same length as case_names')


        # Get the energy list
        energy_list = self.get_energy_per_wd_bin(turbine_list,
                                                 ws_range, 
                                                 pow_ref_range,
                                                 min_points_per_bin,
                                                 mean_or_median,
                                                 frequency_matrix_type,)
        
        # Plot the energy list
        if ax is None:
            fig, ax = plt.subplots()
        for i in range(self.n_cases):
            ax.plot(self.wd_bin_centers, energy_list[i,:], **kwargs, label=labels[i])
        
        ax.set_xlabel('Wind direction [deg]')
        ax.set_ylabel('Energy [kWh]') #TODO Or expected power?
        ax.set_title('Energy by wind direction bin')
        ax.legend()
        ax.grid(True)       

        return ax

    # Plot energy by ws bin
    def plot_energy_by_ws_bin(self, 
                                turbine_list=None, 
                                wd_range=None,
                                min_points_per_bin=1,
                                mean_or_median='mean',
                                frequency_matrix_type='turbine',
                                ax=None, 
                                labels=None,
                                **kwargs):
        
        # Check if self.use_pow_ref is False
        if self.use_pow_ref:
            raise ValueError('Cannot plot energy per ws bin if use_pow_ref is True')
        
        # Check if turbine list is a scalar
        if turbine_list is not None:
            if not isinstance(turbine_list, list) and not isinstance(turbine_list, np.ndarray):
                turbine_list = [turbine_list]
        
        # Check if labels are None, and if not that they have the right length
        if labels is None:
            labels = self.case_names
        else:
            if len(labels) != len(self.case_names):
                raise ValueError('labels must have the same length as case_names')


        # Get the energy list
        energy_list = self.get_energy_per_ws_bin(turbine_list,
                                                 wd_range, 
                                                 min_points_per_bin,
                                                 mean_or_median,
                                                 frequency_matrix_type,)
        
        # Plot the energy list
        if ax is None:
            fig, ax = plt.subplots()
        for i in range(self.n_cases):
            ax.plot(self.ws_bin_centers, energy_list[i,:], **kwargs, label=labels[i])
        
        ax.set_xlabel('Wind speed [m/s]')
        ax.set_ylabel('Energy [kWh]') #TODO Or expected power?
        ax.set_title('Energy by wind direction bin')
        ax.legend()
        ax.grid(True)       

        return ax
    
    
    # Plot energy by pow_ref bin
    def plot_energy_by_pow_ref_bin(self, 
                                turbine_list=None, 
                                wd_range=None,
                                min_points_per_bin=1,
                                mean_or_median='mean',
                                frequency_matrix_type='turbine',
                                ax=None, 
                                labels=None,
                                **kwargs):
        
        # Check if self.use_pow_ref is True
        if not self.use_pow_ref:
            raise ValueError('Cannot plot energy per power reference bin if use_pow_ref is False')


        # Check if turbine list is a scalar
        turbine_list = self._check_turbine_list(turbine_list)
        
        # Check if labels are None, and if not that they have the right length
        if labels is None:
            labels = self.case_names
        else:
            if len(labels) != len(self.case_names):
                raise ValueError('labels must have the same length as case_names')


        # Get the energy list
        energy_list = self.get_energy_per_pow_ref_bin(turbine_list,
                                                 wd_range, 
                                                 min_points_per_bin,
                                                 mean_or_median,
                                                 frequency_matrix_type,)
        
        # Plot the energy list
        if ax is None:
            fig, ax = plt.subplots()
        for i in range(self.n_cases):
            ax.plot(self.pow_ref_bin_centers, energy_list[i,:], **kwargs, label=labels[i])
        
        ax.set_xlabel('Reference power [kW]')
        ax.set_ylabel('Energy [kWh]') #TODO Or expected power?
        ax.set_title('Energy by reference power')
        ax.legend()
        ax.grid(True)       

        return ax

    def visualize_power_ratio_per_bin(self, 
                                      turbine_list=None, 
                                      numerator_case=1,
                                      denominator_case=0,
                                      mean_or_median="mean",
                                      ax=None,
                                      **imshow_kwargs):

        ratio_matrix = self.simple_ratio(
            turbine_list,
            numerator_case,
            denominator_case,
            mean_or_median
        )
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if self.use_pow_ref:
            imshow_extent=(self.pow_ref_edges[0], self.pow_ref_edges[-1], 
                           self.wd_edges[-1], self.wd_edges[0])
        else:
            imshow_extent=(self.ws_edges[0], self.ws_edges[-1], 
                           self.wd_edges[-1], self.wd_edges[0])

        # Handle kwargs, defaults
        default_imshow_kwargs = {
            "aspect" : "auto", 
            "interpolation" : "none",
            "extent" : imshow_extent,
            "label" : None
        }
        imshow_kwargs = {**default_imshow_kwargs, **imshow_kwargs}
        
        pos = ax.imshow(ratio_matrix, **imshow_kwargs)
        cbar = fig.colorbar(pos, ax=ax)
        cbar.set_label("Power ratio")
        if self.use_pow_ref:
            ax.set_xlabel("Reference power")
        else:
            ax.set_xlabel("Wind speed")
        ax.set_ylabel("Wind direction [deg]")

        return ax, cbar

    def simple_ratio(self,
                     turbine_list=None,
                     numerator_case=1,
                     denominator_case=0,
                     mean_or_median="mean"
    ):
        """
        numerator_case and denominator_case can either by strings which match
        a case_name or integers for direct indexing.
        """
        turbine_list = np.array(self._check_turbine_list(turbine_list))

        if mean_or_median == "mean":
            A = self.mean_matrix_list[self._case_index(numerator_case)]\
                [:,:,turbine_list].sum(axis=2)
            B = self.mean_matrix_list[self._case_index(denominator_case)]\
                [:,:,turbine_list].sum(axis=2)
        elif mean_or_median == "median":
            # median(x+y+z) is not equal to median(x)+median(y)+median(z)!
            # Nonlinear operator.
            if len(turbine_list) > 1:
                raise ValueError("Cannot use 'median' with multiple turbines.")
            A = self.median_matrix_list[self._case_index(numerator_case)]\
                [:,:,turbine_list].sum(axis=2)
            B = self.median_matrix_list[self._case_index(denominator_case)]\
                [:,:,turbine_list].sum(axis=2)
        else:
            raise ValueError('mean_or_median must be either "mean" or "median"')

        # A NaN in any potion in A or B will produce NaN. 0 in denominator 
        # returns NaN

        ratio_matrix = np.divide(
            A, B, 
            out=np.nan*np.ones_like(A), 
            where=B!=0
        )

        return ratio_matrix
    
    def simple_ratio_with_confidence_interval(self, 
                                              turbine_list=None,
                                              numerator_case=1,
                                              denominator_case=0,
                                              confidence_level=0.90,
                                              mean_or_median="mean"
        ): 
        """
        From Harvey Motulsky text.
        """
        n_idx = self._case_index(numerator_case)
        d_idx = self._case_index(denominator_case)
        
        # For now, always uses the mean. Unsure how to use median.
        if mean_or_median == "median":
            raise NotImplementedError("Must use mean for confidence interval.")
        elif mean_or_median != "mean":
            raise ValueError("mean_or_median must be 'mean' for ratio with "+\
                "confidence interval calculaiton.")
        
        turbine_list = np.array(self._check_turbine_list(turbine_list))
        # if len(turbine_list) > 1:
        #     raise NotImplementedError("Confidence interval around ratio of "+\
        #         "sum power is not yet implemented.")

        # Extract values for convenience
        A = self.mean_matrix_list[n_idx][:,:,turbine_list].sum(axis=2)
        B = self.mean_matrix_list[d_idx][:,:,turbine_list].sum(axis=2)
        Q = self.simple_ratio(turbine_list, numerator_case, denominator_case)

        if len(turbine_list) == 1: # Take standard error directly
            se_A = self.se_matrix_list[self._case_index(numerator_case)]\
                [:,:,turbine_list].squeeze()
            se_B = self.se_matrix_list[self._case_index(denominator_case)]\
                [:,:,turbine_list].squeeze()
            count_A = self.count_matrix_list[self._case_index(numerator_case)]\
                [:,:,turbine_list].squeeze()
            count_B = self.count_matrix_list[self._case_index(denominator_case)]\
                [:,:,turbine_list].squeeze()
        else: # Compute new, summed standard error
            Var_A = (self.std_matrix_list[n_idx][:,:,turbine_list]**2 * 
                (self.count_matrix_list[n_idx][:,:,turbine_list]-1))\
                .sum(axis=2) /\
                (self.count_matrix_list[n_idx][:,:,turbine_list].sum(axis=2)-2)
            count_A = self.count_matrix_list[n_idx][:,:,turbine_list].sum(axis=2)
            se_A = np.sqrt(Var_A/count_A)

            Var_B = (self.std_matrix_list[d_idx][:,:,turbine_list]**2 * 
                (self.count_matrix_list[d_idx][:,:,turbine_list]-1))\
                .sum(axis=2) /\
                (self.count_matrix_list[d_idx][:,:,turbine_list].sum(axis=2)-2)
            count_B = self.count_matrix_list[d_idx][:,:,turbine_list].sum(axis=2)
            se_B = np.sqrt(Var_B/count_B)

        se_Q = Q * np.sqrt((se_A/A)**2 + (se_B/B)**2)

        t_score = stats.t.ppf((confidence_level+1)/2, count_A+count_B-2)

        ci_low = Q - t_score*se_Q
        ci_high = Q + t_score*se_Q

        return Q, ci_low, ci_high

    

if __name__ == '__main__':
    
    # WIND SPEED TESTS
    # Generate a test dataframe with columns: ws, wd, pow_000, pow_001, pow_002 
    # with N elements each ws are random numbers between 3 and 25
    # wd are random numbers between 0 and 360
    # pow_000 is random numbers between 0 and 1000
    np.random.seed(0)
    df_baseline = pd.DataFrame({
        'ws': np.random.uniform(3, 25, 1000),
        'wd': np.random.uniform(0, 360, 1000),
        'pow_000': np.random.uniform(0, 1000, 1000),
        'pow_001': np.random.uniform(0, 1000, 1000),
        'pow_002': np.random.uniform(0, 1000, 1000)
    })

    df_control = df_baseline.copy()
    df_control['pow_000'] = df_control['pow_000'] * 1.05
    df_control['pow_000'] = np.clip(df_control['pow_000'], 0, 1000)

    #ta = TableAnalysis(wd_step=60., ws_step=7)
    # ta.add_df(df_baseline, 'baseline')
    # ta.add_df(df_control, 'control')

    # print(ta.get_energy_in_range())

    # # ta.print_df_names()

    # # ta.get_overall_frequency_matrix()

    # # print(ta.get_energy_in_range(0))
    
    # print(ta.get_energy_per_ws_bin(0))

    # # ta.plot_energy_by_wd_bin()

    # # ta.plot_energy_by_ws_bin()

    # q, l, h = ta.simple_ratio_with_confidence_interval()
    # print(q.shape)
    # print(l.shape)
    # print(h.shape)

#    plt.show()
    ########################################################
    # POW REF TESTS
    print('~~~~POW REF TESTS~~~~')
    df_baseline = pd.DataFrame({
        # 'ws': np.random.uniform(3, 25, 1000),
        'pow_ref': np.random.uniform(0, 1000, 1000),
        'wd': np.random.uniform(0, 360, 1000),
        'pow_000': np.random.uniform(0, 1000, 1000),
        'pow_001': np.random.uniform(0, 1000, 1000),
        'pow_002': np.random.uniform(0, 1000, 1000)
    })

    df_control = df_baseline.copy()
    df_control['pow_000'] = df_control['pow_000'] * 1.05
    df_control['pow_000'] = np.clip(df_control['pow_000'], 0, 1000)

    ta = TableAnalysis(use_pow_ref=True)
    ta.add_df(df_baseline, 'baseline')
    ta.add_df(df_control, 'control')

    print(ta.get_energy_in_range())

    print(ta.get_energy_per_pow_ref_bin())

    ta.plot_energy_by_pow_ref_bin()

    ax, cbar = ta.visualize_power_ratio_per_bin()
    ax.set_xlim([0, 1000])

    plt.show()