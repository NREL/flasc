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
import matplotlib.pyplot as plt

#TODO: Do we want to follow FLORIS' method of keeping 3 dimensions for all matrices?

class TableAnalysis():

    def __init__(self, ws_step=1.0, wd_step=2.0, minutes_per_point=10.0,):
        
        # Set the wind speed and wind direction step sizes
        self.ws_step = ws_step
        self.wd_step = wd_step
        self.minutes_per_point = minutes_per_point

        # Set the wind speed and wind direction bins
        self.ws_bins = np.arange(0 - ws_step/2.0, 50, ws_step)
        self.wd_bins = np.arange(0,  360 + wd_step, wd_step)

        # Set the wind speed and wind direction bin centers
        self.ws_bin_centers = self.ws_bins[:-1] + ws_step/2
        self.wd_bin_centers = self.wd_bins[:-1] + wd_step/2

        # Save the number of wind speed and wind direction bins
        self.n_ws_bins = len(self.ws_bin_centers)
        self.n_wd_bins = len(self.wd_bin_centers)

        # Intialize list of matrices
        # Organize results in 3D matrix of wind speed, wind direction, and turbine
        # Put in lists for each dataframe
        self.mean_matrix_list = [] # Mean power per wind speed and wind direction and turbine bin
        self.median_matrix_list = [] # Median power per wind speed and wind direction and turbine bin
        self.count_matrix_list = []  # Count of power per wind speed and wind direction and turbine bin
        self.std_matrix_list = [] # Standard deviation of power per wind speed and wind direction and turbine bin
        self.ci_matrix_list = [] # Confidence interval of power per wind speed and wind direction and turbine bin
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
        if 'ws' not in df.columns:
            raise ValueError("Dataframe must have a column named 'ws'")
        #TODO: Check that there is a pow column
        for c in df.columns:
            if not ('wd' in c or 'ws' in c or 'pow_' in c):
                raise ValueError("Dataframe must only have columns named 'wd', 'ws', or 'pow_*'")
        
        # Bin the wind speed and wind direction
        df['ws_bin'] = pd.cut(df['ws'], self.ws_bins,labels=self.ws_bin_centers)
        df['wd_bin'] = pd.cut(df['wd'], self.wd_bins,labels=self.wd_bin_centers)

        # Drop ws and wd
        df = df.drop(columns=['ws', 'wd'])

        # Convert the turbines to a new column
        df = df.melt(id_vars=['wd_bin', 'ws_bin'], var_name='turbine', value_name='power')

        # Get a list of unique turbine names
        turbine_names = df['turbine'].unique()

        # Determine the number of turbines
        self.n_turbines = len(turbine_names)

        # Convert ws_bin and wd_bin to categorical with levels set to the bin centers
        # This enforces that the order of the bins is correct and the size of the matrix
        # matches the number of bins
        df['wd_bin'] = pd.Categorical(df['wd_bin'], categories=self.wd_bin_centers)
        df['ws_bin'] = pd.Categorical(df['ws_bin'], categories=self.ws_bin_centers)
        df['turbine'] = pd.Categorical(df['turbine'], categories=turbine_names)

        #Save the dataframe and name
        self.case_df_list.append(df)
        self.case_names.append(case_name)

        # increment the number of cases
        self.n_cases += 1

        # # Get a matrix of mean values with dimensions: ws, wd, turbine whose shape
        # # is (len(wd_bin_centers), len(ws_bin_centers), n_turbines)
        mean_matrix = df.groupby(['wd_bin', 'ws_bin', 'turbine'])['power'].mean().reset_index().power.to_numpy()
        mean_matrix = mean_matrix.reshape((len(self.wd_bin_centers), len(self.ws_bin_centers), self.n_turbines))
        
        # # Get a matrix of median values with dimensions: ws, wd, turbine whose shape
        # # is (len(wd_bin_centers), len(ws_bin_centers), n_turbines)
        median_matrix = df.groupby(['wd_bin', 'ws_bin', 'turbine'])['power'].median().reset_index().power.to_numpy()
        median_matrix = median_matrix.reshape((len(self.wd_bin_centers), len(self.ws_bin_centers), self.n_turbines))

        # # Get a matrix of count values with dimensions: ws, wd, turbine whose shape
        # # is (len(wd_bin_centers), len(ws_bin_centers), n_turbines)
        count_matrix = df.groupby(['wd_bin', 'ws_bin', 'turbine'])['power'].count().reset_index().fillna(0).power.to_numpy()
        count_matrix = count_matrix.reshape((len(self.wd_bin_centers), len(self.ws_bin_centers), self.n_turbines))

        #TODO For these matrices are what we want to build the new uncertainty bands off of but
        # I'm not sure yet how
        # Option 1: Is something like assume each bin has an uncertainty band defined by a gaussian however
        #    - assume some maximum value until a minimun number of points are in the bin

        # TODO For now just calculated the values likely used in such a formula

        # # Get a matrix of standard deviation values with dimensions: ws, wd, turbine whose shape
        # # is (len(wd_bin_centers), len(ws_bin_centers), n_turbines)
        std_matrix = df.groupby(['wd_bin', 'ws_bin', 'turbine'])['power'].std().reset_index().power.to_numpy()
        std_matrix = std_matrix.reshape((len(self.wd_bin_centers), len(self.ws_bin_centers), self.n_turbines))

        # # Get a matrix of confidence interval values with dimensions: ws, wd, turbine whose shape
        # # is (len(wd_bin_centers), len(ws_bin_centers), n_turbines)
        ci_matrix = std_matrix / np.sqrt(count_matrix)
        ci_matrix = ci_matrix.reshape((len(self.wd_bin_centers), len(self.ws_bin_centers), self.n_turbines))

        #TODO How to calculate mu?
        mu_matrix = np.zeros_like(mean_matrix)
        
        # Add the matrices to the list of matrices
        self.mean_matrix_list.append(mean_matrix)
        self.median_matrix_list.append(median_matrix)
        self.count_matrix_list.append(count_matrix)
        self.std_matrix_list.append(std_matrix)
        self.ci_matrix_list.append(ci_matrix)
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
        
        # Get a matrix of count values with dimensions: ws, wd
        overall_frequency_matrix = df_total.groupby(['wd_bin', 'ws_bin'])['dummy'].sum().reset_index().fillna(0).dummy.to_numpy()
        overall_frequency_matrix = overall_frequency_matrix.reshape((len(self.wd_bin_centers), len(self.ws_bin_centers)))

        # normalize the frequency matrix
        overall_frequency_matrix = overall_frequency_matrix / np.sum(overall_frequency_matrix)

        return overall_frequency_matrix
    
    def get_uniform_frequency_matrix(self):

        # Get a matrix of count values with dimensions: ws, wd
        uniform_frequency_matrix = np.ones((len(self.wd_bin_centers), len(self.ws_bin_centers)))

        # normalize the frequency matrix
        uniform_frequency_matrix = uniform_frequency_matrix / np.sum(uniform_frequency_matrix)

        return uniform_frequency_matrix
    
    def set_user_defined_frequency_matrix(self, user_defined_frequency_matrix):

        # normalize the frequency matrix
        user_defined_frequency_matrix = user_defined_frequency_matrix / np.sum(user_defined_frequency_matrix)

        self.set_user_defined_frequency_matrix = user_defined_frequency_matrix

    def get_user_defined_frequency_matrix(self):

        # Check that user defined frequency matrix has been set
        if self.user_defined_frequency_matrix is None:
            raise ValueError('User defined frequency matrix has not been set')

        return self.user_defined_frequency_matrix

    def get_turbine_availability_mask(self, turbine, min_points_per_bin=1):

        " Get a mask for a given turbine if each matrix has at least min_points_per_bin points in each bin"

        # turbine_availability_mask is true if each matrix has at least N points in each bin
        # Dimensions of turbine_mast are num_wd_bins x num_ws_bins
        turbine_availability_mask = (np.min(self.count_matrix_list[:], axis=0)[:, :, turbine].squeeze() >= min_points_per_bin)

        return turbine_availability_mask

    def get_turbine_frequency_matrix(self, turbine):

        # turbine_frequency_matrix is the sum of the count matrix over all time
        turbine_frequency_matrix = np.sum(self.count_matrix_list[:], axis=0)[:, :, turbine].squeeze()

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


    def get_energy_in_range_per_turbine(self, 
                            turbine, 
                            ws_min=None, 
                            ws_max=None, 
                            wd_min=None,
                            wd_max=None,
                            min_points_per_bin=1,
                            mean_or_median='mean',
                            frequency_matrix_type='turbine',):
        """Calculate the energy in a range of wind speed and wind direction.

        Args:
            turbine (_type_): _description_
            ws_min (_type_): _description_
            ws_max (_type_): _description_
            wd_min (_type_): _description_
            wd_max (_type_): _description_
            min_points_per_bin (_type_): _description_
            minutes_per_point (_type_): _description_
            mean_or_median (_type_): _description_
            

        Returns:
            _type_: _description_
        """            

        #TODO I think this right now returns the average power so needs to be scaled up
        # by number of hours to get energy

        # Set the default values for the wind speed and wind direction bins
        if ws_min is None:
            ws_min = self.ws_bin_centers[0]
        if ws_max is None:
            ws_max = self.ws_bin_centers[-1]
        if wd_min is None:
            wd_min = self.wd_bin_centers[0]
        if wd_max is None:
            wd_max = self.wd_bin_centers[-1]

        # Get the frequency matrix
        frequency_matrix = self.get_frequency_matrix(frequency_matrix_type, turbine)
        
        # Get indices of the wind speed and wind direction bins that are outside the range
        ws_ind_outside = np.where((self.ws_bin_centers < ws_min) | (self.ws_bin_centers > ws_max))[0]
        wd_ind_outside = np.where((self.wd_bin_centers < wd_min) | (self.wd_bin_centers > wd_max))[0]

        # Set the value to 0 for all bins that are not inside the range
        frequency_matrix[wd_ind_outside, :] = 0
        frequency_matrix[:, ws_ind_outside] = 0
        
        # Check that the frequency matrix is not all zeros
        if np.sum(frequency_matrix) == 0:
            raise ValueError('Frequency matrix is all zeros')

        # Re-normalize the frequency matrix
        frequency_matrix = frequency_matrix / np.sum(frequency_matrix)

        # Initialize the energy list
        energy_list_result = np.zeros(self.n_cases)
        
        for i in range(self.n_cases):

            # Get the mean matrix for the wind speed and wind direction bins
            if mean_or_median == 'mean':
                power_matrix = self.mean_matrix_list[i][:, :, turbine]
            elif mean_or_median == 'median':
                power_matrix = self.median_matrix_list[i][:, :, turbine]
            else:
                raise ValueError('mean_or_median must be either "mean" or "median"')

            # Set to 0 all bins where the turbine availability mask is false
            turbine_availability_mask = self.get_turbine_availability_mask(turbine, min_points_per_bin=min_points_per_bin)
            power_matrix[~turbine_availability_mask] = 0.0
    
            # Calculate the energy
            energy_list_result[i]  = np.nansum(frequency_matrix * power_matrix)
            

        return energy_list_result
    
    def get_energy_in_range_across_turbines(self,
                                            turbine_list= None,
                                            ws_min=None, 
                                            ws_max=None, 
                                            wd_min=None,
                                            wd_max=None,
                                            min_points_per_bin=1,
                                            mean_or_median='mean',
                                            frequency_matrix_type='turbine',):
        """Calculate the energy in a range of wind speed and wind direction across the turbines
        in turbine list
        """
        if turbine_list is not None:
            if not isinstance(turbine_list, list) and not isinstance(turbine_list, np.ndarray):
                raise ValueError('turbine_list must be a list')

        if turbine_list is None:
            turbine_list = list(range(self.n_turbines))

        # Initialize the energy list
        energy_list_result = np.zeros(self.n_cases)

        for t_i in turbine_list:
            
            energy_list = self.get_energy_in_range_per_turbine(t_i, 
                                                            ws_min=ws_min, 
                                                            ws_max=ws_max, 
                                                            wd_min=wd_min,
                                                            wd_max=wd_max,
                                                            min_points_per_bin=min_points_per_bin,
                                                            mean_or_median=mean_or_median,
                                                            frequency_matrix_type=frequency_matrix_type,)
            energy_list_result += energy_list

        return energy_list_result
            
            
        
    def get_energy_per_wd_bin_per_turbine(self, 
                            turbine, 
                            ws_min=None, 
                            ws_max=None, 
                            min_points_per_bin=1,
                            mean_or_median='mean',
                            frequency_matrix_type='turbine',):
        """Calculate the energy in a range of wind speed and wind direction.

        Args:
            turbine (_type_): _description_
            ws_min (_type_): _description_
            ws_max (_type_): _description_
            min_points_per_bin (_type_): _description_
            minutes_per_point (_type_): _description_
            mean_or_median (_type_): _description_
            

        Returns:
            _type_: _description_
        """            

        #TODO I think this right now returns the average power so needs to be scaled up
        # by number of hours to get energy

        # Set the default values for the wind speed and wind direction bins
        if ws_min is None:
            ws_min = self.ws_bin_centers[0]
        if ws_max is None:
            ws_max = self.ws_bin_centers[-1]


        # Get the frequency matrix
        frequency_matrix = self.get_frequency_matrix(frequency_matrix_type, turbine)
        
        # Get indices of the wind speed and wind direction bins that are outside the range
        ws_ind_outside = np.where((self.ws_bin_centers < ws_min) | (self.ws_bin_centers > ws_max))[0]

        # Set the value to 0 for all bins that are not inside the range
        frequency_matrix[:, ws_ind_outside] = 0
        
        # Check that the frequency matrix is not all zeros
        if np.sum(frequency_matrix) == 0:
            raise ValueError('Frequency matrix is all zeros')

        # Re-normalize the frequency matrix by wind direction bin so 
        # that the sum of the frequency matrix is 1 for each wind direction bin
        frequency_matrix = frequency_matrix / np.sum(frequency_matrix, axis=1)[:, None]

        # Initialize the energy list
        energy_list_result = np.zeros([self.n_cases, self.n_wd_bins])
        
        for i in range(self.n_cases):

            # Get the mean matrix for the wind speed and wind direction bins
            if mean_or_median == 'mean':
                power_matrix = self.mean_matrix_list[i][:, :, turbine]
            elif mean_or_median == 'median':
                power_matrix = self.median_matrix_list[i][:, :, turbine]
            else:
                raise ValueError('mean_or_median must be either "mean" or "median"')

            # Set to 0 all bins where the turbine availability mask is false
            turbine_availability_mask = self.get_turbine_availability_mask(turbine, min_points_per_bin=min_points_per_bin)
            power_matrix[~turbine_availability_mask] = 0.0

            # Calculate the energy
            energy_list_result[i, :]  = np.nansum(frequency_matrix * power_matrix, axis=1)
            

        return energy_list_result
    
    def get_energy_per_wd_bin_across_turbines(self,
                                            turbine_list= None,
                                            ws_min=None, 
                                            ws_max=None, 
                                            min_points_per_bin=1,
                                            mean_or_median='mean',
                                            frequency_matrix_type='turbine',):
        """Calculate the energy in a range of wind speed and wind direction across the turbines
        in turbine list
        """
        if turbine_list is not None:
            if not isinstance(turbine_list, list) and not isinstance(turbine_list, np.ndarray):
                raise ValueError('turbine_list must be a list')

        if turbine_list is None:
            turbine_list = list(range(self.n_turbines))

        # Initialize the energy list
        energy_list_result = np.zeros([self.n_cases, self.n_wd_bins])

        for t_i in turbine_list:
            
            energy_list = self.get_energy_per_wd_bin_per_turbine(t_i, 
                                                            ws_min=ws_min, 
                                                            ws_max=ws_max, 
                                                            min_points_per_bin=min_points_per_bin,
                                                            mean_or_median=mean_or_median,
                                                            frequency_matrix_type=frequency_matrix_type,)
            energy_list_result += energy_list

        return energy_list_result
    
    def get_energy_per_ws_bin_across_turbines(self,
                                            turbine_list= None,
                                            wd_min=None, 
                                            wd_max=None, 
                                            min_points_per_bin=1,
                                            mean_or_median='mean',
                                            frequency_matrix_type='turbine',):
        """Calculate the energy in a range of wind speed and wind direction across the turbines
        in turbine list
        """
        if turbine_list is not None:
            if not isinstance(turbine_list, list) and not isinstance(turbine_list, np.ndarray):
                raise ValueError('turbine_list must be a list')

        if turbine_list is None:
            turbine_list = list(range(self.n_turbines))

        # Initialize the energy list
        energy_list_result = np.zeros([self.n_cases, self.n_ws_bins])

        for t_i in turbine_list:
            
            energy_list = self.get_energy_per_ws_bin_per_turbine(t_i, 
                                                            wd_min=wd_min, 
                                                            wd_max=wd_max, 
                                                            min_points_per_bin=min_points_per_bin,
                                                            mean_or_median=mean_or_median,
                                                            frequency_matrix_type=frequency_matrix_type,)
            energy_list_result += energy_list

        return energy_list_result
    
    # Define the ws version of the above function
    def get_energy_per_ws_bin_per_turbine(self, 
                            turbine, 
                            wd_min=None, 
                            wd_max=None, 
                            min_points_per_bin=1,
                            mean_or_median='mean',
                            frequency_matrix_type='turbine',):
        """Calculate the energy in a range of wind speed and wind direction.

        Args:
            turbine (_type_): _description_
            ws_min (_type_): _description_
            ws_max (_type_): _description_
            min_points_per_bin (_type_): _description_
            minutes_per_point (_type_): _description_
            mean_or_median (_type_): _description_
            

        Returns:
            _type_: _description_
        """            

        #TODO I think this right now returns the average power so needs to be scaled up
        # by number of hours to get energy

        # Set the default values for the wind speed and wind direction bins
        if wd_min is None:
            wd_min = self.wd_bin_centers[0]
        if wd_max is None:
            wd_max = self.wd_bin_centers[-1]


        # Get the frequency matrix
        frequency_matrix = self.get_frequency_matrix(frequency_matrix_type, turbine)
        
        # Get indices of the wind speed and wind direction bins that are outside the range
        wd_ind_outside = np.where((self.wd_bin_centers < wd_min) | (self.wd_bin_centers > wd_max))[0]

        # Set the value to 0 for all bins that are not inside the range
        frequency_matrix[wd_ind_outside, :] = 0
        
        # Check that the frequency matrix is not all zeros
        if np.sum(frequency_matrix) == 0:
            raise ValueError('Frequency matrix is all zeros')

        # Re-normalize the frequency matrix by wind direction bin so 
        # that the sum of the frequency matrix is 1 for each wind direction bin
        frequency_matrix = frequency_matrix / np.sum(frequency_matrix, axis=0)[None, :]

        # Initialize the energy list
        energy_list_result = np.zeros([self.n_cases, self.n_ws_bins])
        
        for i in range(self.n_cases):

            # Get the mean matrix for the wind speed and wind direction bins
            if mean_or_median == 'mean':
                power_matrix = self.mean_matrix_list[i][:, :, turbine]
            elif mean_or_median == 'median':
                power_matrix = self.median_matrix_list[i][:, :, turbine]
            else:
                raise ValueError('mean_or_median must be either "mean" or "median"')

            # Set to 0 all bins where the turbine availability mask is false
            turbine_availability_mask = self.get_turbine_availability_mask(turbine, min_points_per_bin=min_points_per_bin)
            power_matrix[~turbine_availability_mask] = 0.0

            # Calculate the energy
            energy_list_result[i, :]  = np.nansum(frequency_matrix * power_matrix, axis=0)
            

        return energy_list_result
        

    
    def plot_energy_by_ws_bin(self, 
                                turbine_list=None, 
                                wd_min=None,
                                wd_max=None, 
                                min_points_per_bin=1,
                                mean_or_median='mean',
                                frequency_matrix_type='turbine',
                                ax=None, 
                                **kwargs):
        
        # Check if turbine list is a scalar
        if turbine_list is not None:
            if not isinstance(turbine_list, list) and not isinstance(turbine_list, np.ndarray):
                turbine_list = [turbine_list]
        
        # Get the energy list
        energy_list = self.get_energy_per_ws_bin_across_turbines(turbine_list,
                                                                  wd_min, 
                                                                  wd_max,
                                                                  min_points_per_bin,
                                                                  mean_or_median,
                                                                  frequency_matrix_type,)
        
        # Plot the energy list
        if ax is None:
            fig, ax = plt.subplots()
        for i in range(len(energy_list)):
            ax.plot(self.ws_bin_centers, energy_list[i,:], **kwargs, label=self.case_names[i])
        
        ax.set_xlabel('Wind speed [m/s]')
        ax.set_ylabel('Energy [kWh]') #TODO Or expected power?
        ax.set_title('Energy by wind speed bin')
        ax.legend()
        ax.grid(True)       

        return ax
    
    def plot_energy_by_wd_bin(self, 
                                turbine_list=None, 
                                ws_min=None,
                                ws_max=None, 
                                min_points_per_bin=1,
                                mean_or_median='mean',
                                frequency_matrix_type='turbine',
                                ax=None, 
                                **kwargs):
        
        # Check if turbine list is a scalar
        if turbine_list is not None:
            if not isinstance(turbine_list, list) and not isinstance(turbine_list, np.ndarray):
                turbine_list = [turbine_list]
        
        # Get the energy list
        energy_list = self.get_energy_per_wd_bin_across_turbines(turbine_list,
                                                                  ws_min, 
                                                                  ws_max,
                                                                  min_points_per_bin,
                                                                  mean_or_median,
                                                                  frequency_matrix_type,)
        
        # Plot the energy list
        if ax is None:
            fig, ax = plt.subplots()
        for i in range(len(energy_list)):
            ax.plot(self.wd_bin_centers, energy_list[i,:], **kwargs, label=self.case_names[i])
        
        ax.set_xlabel('Wind direction [deg]')
        ax.set_ylabel('Energy [kWh]') #TODO Or expected power?
        ax.set_title('Energy by wind direction bin')
        ax.legend()
        ax.grid(True)       

        return ax


    

if __name__ == '__main__':
    
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

    ta = TableAnalysis()
    ta.add_df(df_baseline, 'baseline')
    ta.add_df(df_control, 'control')

    ta.print_df_names()

    ta.get_overall_frequency_matrix()

    print(ta.get_energy_in_range_per_turbine(0))

    print(ta.get_energy_per_ws_bin_per_turbine(0))

    ta.plot_energy_by_wd_bin()

    ta.plot_energy_by_ws_bin()

    plt.show()

   

    # print(ta.get_energy_in_range(0, 8, 12, 0, 90))

    # print(ta.get_energy_by_wd_bin(0, 8, 12)[0].shape)

    # print(ta.get_energy_by_ws_bin(0, 0, 90)[0].shape)

    # ta.plot_energy_by_ws_bin(0, 0, 90)
