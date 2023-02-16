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

class TableAnalysis():

    def __init__(self, ws_step=1.0, wd_step=2.0):
        
        # Set the wind speed and wind direction step sizes
        self.ws_step = ws_step
        self.wd_step = wd_step

        # Set the wind speed and wind direction bins
        self.ws_bins = np.arange(0, 25, ws_step)
        self.wd_bins = np.arange(0, 360, wd_step)

        # Set the wind speed and wind direction bin centers
        self.ws_bin_centers = self.ws_bins[:-1] + ws_step/2
        self.wd_bin_centers = self.wd_bins[:-1] + wd_step/2

        # Initialize df list
        self.df_list = []
        self.df_names = []

        # Intialize list of matrices
        # Organize results in 3D matrix of wind speed, wind direction, and turbine
        # Put in lists for each dataframe
        self.mean_matrix_list = [] # Mean power per wind speed and wind direction and turbine bin
        self.median_matrix_list = [] # Median power per wind speed and wind direction and turbine bin
        self.count_matrix_list = []  # Count of power per wind speed and wind direction and turbine bin
        self.std_matrix_list = [] # Standard deviation of power per wind speed and wind direction and turbine bin
        self.ci_matrix_list = [] # Confidence interval of power per wind speed and wind direction and turbine bin
        self.mu_matrix_list = [] # Guassian mu value


    def add_df(self, df_in, name):

        # Make a copy of the dataframe
        df = df_in.copy()

        # Check that the dataframe has the correct columns
        if not 'wd' in df.columns:
            raise ValueError("Dataframe must have a column named 'wd'")
        if not 'ws' in df.columns:
            raise ValueError("Dataframe must have a column named 'ws'")
        for c in df.columns:
            if not ('wd' in c or 'ws' in c or 'pow_' in c):
                raise ValueError("Dataframe must only have columns named 'wd', 'ws', or 'pow_*'")
        

        #Add a dataframe to the list of dataframes
        self.df_list.append(df)
        self.df_names.append(name)

        # Bin the wind speed and wind direction
        df['ws_bin'] = pd.cut(df['ws'], self.ws_bins,labels=self.ws_bin_centers)
        df['wd_bin'] = pd.cut(df['wd'], self.wd_bins,labels=self.wd_bin_centers)

        # Drop ws and wd
        df = df.drop(columns=['ws', 'wd'])

        # Convert the turbines to a new column
        df = df.melt(id_vars=['ws_bin', 'wd_bin'], var_name='turbine', value_name='power')

        # Get a list of unique turbine names
        self.turbine_names = df['turbine'].unique()

        # Determine the number of turbines
        self.n_turbines = len(self.turbine_names)

        # Convert ws_bin and wd_bin to categorical with levels set to the bin centers
        # This enforces that the order of the bins is correct and the size of the matrix
        # matches the number of bins
        df['ws_bin'] = pd.Categorical(df['ws_bin'], categories=self.ws_bin_centers)
        df['wd_bin'] = pd.Categorical(df['wd_bin'], categories=self.wd_bin_centers)
        df['turbine'] = pd.Categorical(df['turbine'], categories=self.turbine_names)

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
        for ii, df in enumerate(self.df_list):
            print('  %d: %s' % (ii, self.df_names[ii]))

    def get_frequency_matrix(self, turbine):

        # TODO: Not positive but I think it's correct that the value of the frequency matrix
        # is the minimum number of points in a bin for all turbines
        # that way if a turbine is missing data in a bin then the frequency matrix
        # will be 0 for that bin

        # MS: I think that this should be the sum, not the min? But somehow 
        # it should deal with missing data bins, and I see how the min is good 
        # for that.

        # print(self.count_matrix_list[:][:, :, turbine])

        frequency_matrix = np.min(self.count_matrix_list[:], axis=0)[:, :, turbine].squeeze()

        # Normalize the frequency matrix
        frequency_matrix = frequency_matrix / np.sum(frequency_matrix)

        return frequency_matrix


    #TODO Could this name be more clear?
    def get_energy_in_range(self, 
                            turbine, 
                            ws_min=None, 
                            ws_max=None, 
                            wd_min=None,
                            wd_max=None,
                            minutes_per_point=10.0,
                            mean_or_median='mean',):
        """Calculate the energy in a range of wind speed and wind direction.

        Args:
            turbine (_type_): _description_
            ws_min (_type_): _description_
            ws_max (_type_): _description_
            wd_min (_type_): _description_
            wd_max (_type_): _description_
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

        if mean_or_median not in ['mean', 'median']:
            raise ValueError('mean_or_median must be either "mean" or "median"')

        
        # Get the indices of the wind speed and wind direction bins
        ws_ind = np.where((self.ws_bin_centers >= ws_min) & (self.ws_bin_centers <= ws_max))[0]
        wd_ind = np.where((self.wd_bin_centers >= wd_min) & (self.wd_bin_centers <= wd_max))[0]

        # Get the frequency matrix for this turbine
        freq_matrix = self.get_frequency_matrix(turbine)

        # Get the frequency matrix for the wind speed and wind direction bins
        freq_matrix = freq_matrix[wd_ind, :][:, ws_ind]

        # Normalize the frequency matrix
        freq_matrix = freq_matrix / np.sum(freq_matrix)

        # Initialize the energy list
        energy_list_mean = []
        

        for i in range(len(self.mean_matrix_list)):

            # Get the mean matrix for the wind speed and wind direction bins
            if mean_or_median == 'mean':
                power_matrix = self.mean_matrix_list[i][wd_ind, :][:, ws_ind, turbine]
            elif mean_or_median == 'median':
                power_matrix = self.median_matrix_list[i][wd_ind, :][:, ws_ind, turbine]
    
            # Calculate the energy
            energy = np.nansum(freq_matrix * power_matrix)
            
            # Append the energy to the list
            energy_list_mean.append(energy)

        return energy_list_mean

    def get_energy_by_wd_bin(self, turbine, ws_min=None, ws_max=None):
        
        # Initialize the energy list
        energy_list = []
        
        for i in range(len(self.mean_matrix_list)):

            # Get the wind speed bins
            if ws_min is None:
                ws_min = self.ws_bin_centers.min()
            if ws_max is None:
                ws_max = self.ws_bin_centers.max()

            # Get the indices of the wind speed  bins
            ws_ind = np.where((self.ws_bin_centers >= ws_min) & (self.ws_bin_centers <= ws_max))[0]

            # Get the frequency matrix for this turbine
            freq_matrix = self.get_frequency_matrix(turbine)

            # Get the frequency matrix for the wind speed bins
            # TODO: Check I do this right
            freq_matrix = freq_matrix[:, ws_ind]

            # Normalize the frequency matrix
            freq_matrix = freq_matrix / np.sum(freq_matrix)

            # Get the mean matrix for the wind speed bins
            mean_matrix = self.mean_matrix_list[i][:, ws_ind, turbine]

            # Calculate the energy
            energy = np.nansum(freq_matrix * mean_matrix, axis=1)

            # Append the energy to the list
            energy_list.append(energy)

        return energy_list
        
    def get_energy_by_ws_bin(self, turbine, wd_min=None, wd_max=None):
            
            # Initialize the energy list
            energy_list = []
            
            for i in range(len(self.mean_matrix_list)):
    
                # Get the wind direction bins
                if wd_min is None:
                    wd_min = self.wd_bin_centers.min()
                if wd_max is None:
                    wd_max = self.wd_bin_centers.max()
    
                # Get the indices of the wind direction  bins
                wd_ind = np.where((self.wd_bin_centers >= wd_min) & (self.wd_bin_centers <= wd_max))[0]
    
                # Get the frequency matrix for this turbine
                freq_matrix = self.get_frequency_matrix(turbine)
    
                # Get the frequency matrix for the wind direction bins
                freq_matrix = freq_matrix[wd_ind, :]
    
                # Normalize the frequency matrix
                freq_matrix = freq_matrix / np.sum(freq_matrix)
    
                # Get the mean matrix for the wind direction bins
                mean_matrix = self.mean_matrix_list[i][wd_ind, :, turbine]
    
                # Calculate the energy
                energy = np.nansum(freq_matrix * mean_matrix, axis=0)
    
                # Append the energy to the list
                energy_list.append(energy)
    
            return energy_list
            
            
    
    def plot_energy_by_ws_bin(self, turbine, wd_min=None, wd_max=None, ax=None, **kwargs):
        
        # Get the energy list
        energy_list = self.get_energy_by_ws_bin(turbine, wd_min, wd_max)
        
        # Plot the energy list
        if ax is None:
            fig, ax = plt.subplots()
        for i in range(len(energy_list)):
            ax.plot(self.ws_bin_centers, energy_list[i], **kwargs, label=self.df_names[i])
        
        ax.set_xlabel('Wind speed [m/s]')
        ax.set_ylabel('Energy [kWh]')
        ax.set_title('Energy by wind speed bin')
        ax.legend()
        ax.grid(True)       
        plt.show()

        return ax
    
    def plot_energy_diff_by_ws_bin(self, turbine, wd_min=None, wd_max=None, ax=None, **kwargs):
        
        # Get the energy list
        energy_list = self.get_energy_by_ws_bin(turbine, wd_min, wd_max)
        
        # Plot the energy list
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.plot(self.ws_bin_centers, energy_list[1] - energy_list[0], **kwargs, 
                label=f'{self.df_names[1]} - {self.df_names[0]}')
        
        ax.set_xlabel('Wind speed [m/s]')
        ax.set_ylabel('Energy [kWh]')
        ax.set_title('Difference energy by wind speed bin')
        ax.legend()
        ax.grid(True)       
        plt.show()

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

    print(ta.get_energy_in_range(0, 8, 12, 0, 90))

    print(ta.get_energy_by_wd_bin(0, 8, 12)[0].shape)

    print(ta.get_energy_by_ws_bin(0, 0, 90)[0].shape)

    ta.plot_energy_by_ws_bin(0, 0, 90)

    import ipdb; ipdb.set_trace()