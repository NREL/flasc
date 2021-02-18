import pandas as pd
import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np
import dill as pickle
from floris.utilities import wrap_180, wrap_360
# import floris_scada.utility as ut
# from floris.tools.plotting import data_plot
# import floris.tools.plotting as wpt
import seaborn as sns
from itertools import product
from matplotlib.patches import  Polygon
import matplotlib.patches as patches
import os

#Plotly stuff
import plotly_express as px
import plotly.io as pio
pio.templates.default = "plotly_white"




def compute_expectation(fx, p_X):
        """
        Compute expected value of f(X), X ~ p_X(x).
        Inputs:
            fx - pandas Series / 1-D numpy - array of possible outcomes.
            p_X - pandas Series / 1-D numpy - distribution of X.
                                              May be supplied as 
                                              nonnormalized frequencies.
        Outputs:
            (anon) - float - Expected value of f(X)
        """
        p_X = p_X/p_X.sum() # Make sure distribution is valid

        return fx @ p_X


class Energy_Analysis():
    """
    Top level analysis class for storing the data used across the analysis

    Args:

    Returns:
        TBD
    """

    def __init__(self, 
                fs, 
                test_turbines, 
                ref_turbines, 
                control_turbines=[], 
                dep_turbines = [], 
                wd_step=2,
                ws_step=1,
                full_filter_control_turbines=True, 
                full_filter_test_turbines=True, 
                full_filter_dep_turbines=True,
                full_filter_all=True,
                keep_unready_data=False,
                verbose=True,
                forced_wd_min=None,
                use_median=False,
                do_prints=True):
        
        # Instantiate the object using the passed in data

        # Make sure that test_turbines and ref_turbines are lists
        if not (type(test_turbines) is list):
            test_turbines = [test_turbines]
        if not (type(ref_turbines) is list):
            ref_turbines = [ref_turbines]


        # Save certain inputs
        #Bin Category Order
        # self.bin_cat_order = fs.bin_cat_order
        # Save the mean median choice
        self.use_median = use_median
        # Save the control turbines
        self.control_turbines = control_turbines


        # Determine turbine filtering levels
        # All subject to self filtering
        self_status_filter_list = test_turbines + ref_turbines + control_turbines + dep_turbines

        # Full filtering all depends
        full_status_filter_list = ref_turbines

        if full_filter_control_turbines:
            full_status_filter_list = full_status_filter_list + control_turbines

        if full_filter_test_turbines:
            full_status_filter_list = full_status_filter_list + test_turbines

        if full_filter_dep_turbines:
            full_status_filter_list = full_status_filter_list + dep_turbines

        if full_filter_all:
            full_status_filter_list = test_turbines + ref_turbines + control_turbines + dep_turbines

        # Show the lists
        if do_prints:
            print('Turbines to filter by self status:', self_status_filter_list)
            print('Turbines to filter by full status:', full_status_filter_list)

        # Set up the filtering masks
        self_status_mask = fs.get_self_status_mask(self_status_filter_list)
        full_status_mask = fs.get_self_status_mask(full_status_filter_list)
        ws_mask = fs.df['ws'].isnull() == False
        wd_mask = fs.df['wd'].isnull() == False

        # Vane mask 
        vane_mask = None
        for t in ref_turbines:
            if vane_mask is None:
                vane_mask = np.abs(fs.df['vane_cor_%03d' % t].values) < 25. 
            else:
                vane_mask = vane_mask & (np.abs(fs.df['vane_cor_%03d' % t].values) < 25.)

        if do_prints:
            print('The self status flag keeps %.1f%% of the data' % (100. * np.sum(self_status_mask)/len(self_status_mask)))
            print('The full status flag keeps %.1f%% of the data' % (100. * np.sum(full_status_mask)/len(full_status_mask)))
            print('The ws status flag keeps %.1f%% of the data' % (100. * np.sum(ws_mask)/len(ws_mask)))
            print('The wd status flag keeps %.1f%% of the data' % (100. * np.sum(wd_mask)/len(wd_mask)))
            print('The vane flag keeps %.1f%% of the data' % (100. * np.sum(vane_mask)/len(vane_mask)))

        if keep_unready_data:
            full_mask = full_status_mask & self_status_mask & ws_mask & wd_mask 
        else:
            unready_data_mask = fs.df.controller_ready==1
            if do_prints:
                print('The unready status flag keeps %.1f%% of the data' % (100. * np.sum(unready_data_mask)/len(unready_data_mask)))
            full_mask = full_status_mask & self_status_mask & ws_mask & wd_mask & unready_data_mask

        if do_prints:
            print('The full mask (no vane mask) keeps %.1f%% of the data' % (100. * np.sum(full_mask)/len(full_mask)))
        full_mask = full_mask & vane_mask
        if do_prints:
            print('The full mask (with vane mask) keeps %.1f%% of the data' % (100. * np.sum(full_mask)/len(full_mask)))

        # Now grab the data into a new dataframe
        df = fs.df.copy()
        df['ref_power'] = fs.get_column_mean_for_turbine_list('pow',ref_turbines)
        df['test_power'] = fs.get_column_mean_for_turbine_list('pow',test_turbines)

        # Now apply the filtering down
        if do_prints:
            print('Before filtering there are %d rows' % df.shape[0])
        df = df[full_mask]
        if do_prints:
            print('After filtering there are %d rows' % df.shape[0])
        
        # Create the binned versions of ws and wd
        if forced_wd_min is None:
            wd_min = np.floor(df.wd.min()) - wd_step/2.0
        else:
            wd_min = forced_wd_min

        # Save choice of wd min
        self.wd_min = wd_min

        # Save the steps
        self.ws_step = ws_step
        self.wd_step = wd_step

        # Save the dataframe
        self.df = df

        # Setup the directions
        self.setup_directions_and_frequencies()

        # Reset to simple index
        self.df = self.df.reset_index(drop=True)

    def setup_directions_and_frequencies(self):

        # Define some local values
        ws_step = self.ws_step
        wd_step = self.wd_step
        wd_min = self.wd_min

        ws_min = np.floor(self.df.ws.min()) - ws_step/2.0
        ws_max = np.ceil(self.df.ws.max()) + ws_step

        wd_max = np.ceil(self.df.wd.max())+ wd_step
        ws_edges = np.arange(ws_min,ws_max,ws_step)
        ws_labels = ws_edges[:-1] + ws_step / 2.0
        wd_edges = np.arange(wd_min,wd_max,wd_step)
        wd_labels = wd_edges[:-1] + wd_step / 2.0

        # Notice both bins labeled by the middle
        self.df['ws_bin'] = pd.cut(self.df.ws,ws_edges,labels=ws_labels)
        self.df['wd_bin'] = pd.cut(self.df.wd,wd_edges,labels=wd_labels)

        # Make sure a float
        self.df['ws_bin'] = self.df['ws_bin'].astype(float)
        self.df['wd_bin'] = self.df['wd_bin'].astype(float)

        # Self the labels, these are all possible values
        self.ws_labels = ws_labels
        self.wd_labels = wd_labels

        # Save the edges too
        self.ws_edges = ws_edges
        self.wd_edges = wd_edges

        # Build up the observed frequency table
        self.build_observed_freq()

        # Build up the annual frequency table
        self.build_annual_freq()


    def show_data_collection(self, ws_range, wd_range, ax=None,show_rect=True):
        
        # Use all the bins
        # ws_bins, wd_bins = self.get_in_range_bins(ws_range, wd_range)
        ws_bins = self.ws_labels
        wd_bins = self.wd_labels

        # Get the half steps
        ws_half = self.ws_step / 2.0
        wd_half = self.wd_step / 2.0

        # Get the in range data
        # df = self.limit_to_wsbin_wdbin(self.df, ws_range, wd_range)

        # Use full range
        df = self.df.copy()


        df = df[['ws_bin','wd_bin','control_mode']].copy()
        df['freq'] = 1
        df = df.groupby(['ws_bin','wd_bin','control_mode']).sum()


        indices = list(product(ws_bins,wd_bins,['baseline','controlled']))
        df = df.reindex(indices).fillna(0).reset_index()

        
        if ax is None:
            fig, ax = plt.subplots()

        for row in df.iterrows():
            v = row[1]
            ws = v.ws_bin
            wd = v.wd_bin
            if v.control_mode=='baseline':
                points = np.array([[wd - wd_half, ws+ws_half],[wd - wd_half, ws-ws_half],[wd + wd_half, ws-ws_half]])
            else:
                points = np.array([[wd - wd_half, ws+ws_half],[wd + wd_half, ws+ws_half],[wd + wd_half, ws-ws_half]])
            


            if v.freq == 0:
                color = 'r'
            elif v.freq <= 3:
                color = 'gray'
            elif v.freq > 3:
                color = 'green'
            polygon = Polygon(points, True,color=color)
            ax.add_patch(polygon)
        # ax.set_xlim(wd_range)
        ax.set_xlim(wd_bins.min(),wd_bins.max())
        ax.set_ylim(ws_bins.min(),ws_bins.max())

        if show_rect:
            # Draw the range rectangle
            rect = patches.Rectangle((wd_range[0]-wd_half,ws_range[0]-ws_half),wd_range[1] - wd_range[0],
                        ws_range[1] - ws_range[0],linewidth=3,edgecolor='c',facecolor='none')
            ax.add_patch(rect)
            

        # return ax


    def build_observed_freq(self, plot_distribution=False):
        if plot_distribution:
            fig = px.density_heatmap(self.df, x="wd_bin", y="ws_bin", marginal_x="histogram", marginal_y="histogram")
            fig.show()
        df_freq_observed = self.df[['ws_bin','wd_bin']].copy()
        df_freq_observed['freq'] = 1
        df_freq_observed = df_freq_observed.groupby(['ws_bin','wd_bin']).sum()

        indices = list(product(self.ws_labels,self.wd_labels))
        self.df_freq_observed = df_freq_observed.reindex(indices).fillna(0).reset_index()


    def build_annual_freq(self):

        if os.path.exists("wind_rose_annual_function.p"):

            # Load the saved annual frequency function
            annual_interp_function = pickle.load( open( "wind_rose_annual_function.p", "rb" ) )

            num_bins = len(self.ws_labels) * len(self.wd_labels)
            ws_array = np.zeros(num_bins)
            wd_array = np.zeros(num_bins)
            freq_array = np.zeros(num_bins) 

            for idx, (ws, wd) in enumerate(product(self.ws_labels, self.wd_labels)):
                ws_array[idx] = ws
                wd_array[idx] = wd
                freq_array[idx] = annual_interp_function(ws,wd)
        
            self.df_freq_annual = pd.DataFrame({'ws_bin':ws_array,'wd_bin':wd_array,'freq':freq_array})

        else:
            # print('No annnual function saved, using observed')
            self.df_freq_annual = self.df_freq_observed


    def limit_to_wsbin_wdbin(self, df, ws_range, wd_range):
        # Currently not including right, could chage
        # starting_rows = df.shape[0]
        df = df[df.ws_bin >= ws_range[0]]
        df = df[df.ws_bin < ws_range[1]]
        df = df[df.wd_bin >= wd_range[0]]
        df = df[df.wd_bin < wd_range[1]]
        # print('Wind range keeps %.1f%% of rows: %d' % (100 * df.shape[0]/starting_rows, df.shape[0]))

        return df

    def get_ws_only_distribution(self, ws_range, wd_range, use_observed_distribution=True):

        if use_observed_distribution:
            df_freq = self.df_freq_observed
        else:
            df_freq = self.df_freq_annual

        df_freq = self.limit_to_wsbin_wdbin(df_freq, ws_range, wd_range)
        df_freq = df_freq[['ws_bin','freq']].groupby('ws_bin').sum()
        df_freq = df_freq / df_freq.sum()
        df_freq = df_freq.reset_index()

        return df_freq

    def compare_ws_distribution_in_range(self, ws_range, wd_range):
        df_obs = self.get_ws_only_distribution(ws_range, wd_range, use_observed_distribution=True)
        df_obs['Distribution'] = 'Observed'
        df_obs.columns = ['Windspeed','Frequency','Distribution']
        df_an = self.get_ws_only_distribution(ws_range, wd_range, use_observed_distribution=False)
        df_an['Distribution'] = 'Annual'
        df_an.columns = ['Windspeed','Frequency','Distribution']

        df = pd.concat([df_obs,df_an])
        df = df.sort_values('Windspeed')

        g = sns.FacetGrid(df, hue='Distribution',aspect=2)
        g.map(plt.plot,'Windspeed','Frequency')
        g.add_legend()
        return g.fig

    def get_energy_ratio_frame(self, ws_range, wd_range, category):

        # Extract the interesting parts for the energy ratio frame
        df_sub = self.df[['ws_bin','wd_bin','ref_power','test_power','category']].copy()
        df_freq_sub = self.df_freq_observed.copy()
        df_freq_sub.columns = ['ws_bin','wd_bin','observed_freq']
        df_freq_sub = df_freq_sub.merge(self.df_freq_annual,how='left',on=['ws_bin','wd_bin'])
        df_freq_sub.columns = ['ws_bin','wd_bin','observed_freq','annual_freq']

        df_sub = self.limit_to_wsbin_wdbin(df_sub, ws_range, wd_range)
        df_freq_sub = self.limit_to_wsbin_wdbin(df_freq_sub, ws_range, wd_range)

        # Make the call to get that frame
        self.df_energy_frame =  df_sub.reset_index(drop=True)
        self.df_energy_frame_freq = df_freq_sub.reset_index(drop=True)

        # Grab the possible indices
        self.energy_frame = Energy_Frame(df_sub, df_freq_sub, category)

    def stack_eo(self, eo_in):
        # Only works if similar assumptions

        # Save the size before
        before  = self.df.shape[0]

        self.df = self.df.append(eo_in.df.copy())


        self.setup_directions_and_frequencies()

        # Reset to simple index
        self.df = self.df.reset_index(drop=True)
        
        # Print points
        print('*** AFTER STACKING***')
        print('Num points go from %d to %d' % (before,self.df.shape[0] ))

class Energy_Frame():
    """
    Top level analysis class for storing the data used across the analysis

    Args:

    Returns:
        TBD
    """

    def __init__(self, df, df_freq, category):
        
        self.df = df
        

        # Renormalize frequecies
        df_freq['observed_freq'] = df_freq.observed_freq / df_freq.observed_freq.sum()
        df_freq['annual_freq'] = df_freq.annual_freq / df_freq.annual_freq.sum()
        self.df_freq = df_freq

        # Save some relevent info
        self.wd_bin = np.array(sorted(df.wd_bin.unique()))
        self.ws_bin = np.array(sorted(df.ws_bin.unique()))
        self.category = category

    def get_column(self, wd):
        df_sub = self.df[self.df.wd_bin==wd].reset_index(drop=True).drop(['wd_bin'],axis='columns')
        df_freq_sub = self.df_freq[self.df_freq.wd_bin==wd].reset_index(drop=True).drop(['wd_bin'],axis='columns')
        return Energy_Column(df_sub, df_freq_sub, self.ws_bin, self.category)


    def get_1_cat_energy_ratio_array_with_range(self, use_observed_freq=True, N=100):

        result = np.zeros([len(self.wd_bin), 3])

        for wd_idx, wd in enumerate(self.wd_bin):
            # print(wd)
            ec = self.get_column(wd)
            result[wd_idx,:] = ec.get_1cat_energy_ratio_with_range(use_observed_freq = use_observed_freq,  N=N)
        
        df_res = pd.DataFrame(result, columns = ['baseline','baseline_l','baseline_u'])
        df_res['wd_bin'] = self.wd_bin

        return df_res

    def get_2_cat_energy_ratio_array_with_range(self, use_observed_freq=True, N=100):

        result = np.zeros([len(self.wd_bin), 12])

        for wd_idx, wd in enumerate(self.wd_bin):
            # print(wd)
            ec = self.get_column(wd)
            result[wd_idx,:] = ec.get_2cat_energy_ratio_with_range(use_observed_freq = use_observed_freq,  N=N)
        
        df_res = pd.DataFrame(result, columns = ['baseline','baseline_l','baseline_u','controlled','controlled_l','controlled_u','diff','diff_l','diff_u','per','per_l','per_u'])
        df_res['wd_bin'] = self.wd_bin

        return df_res




class Energy_Column():

    def __init__(self, df, df_freq, ws_bin, category):

        self.df = df

        # Renormalize frequecies
        df_freq['observed_freq'] = df_freq.observed_freq / df_freq.observed_freq.sum()
        df_freq['annual_freq'] = df_freq.annual_freq / df_freq.annual_freq.sum()
        self.df_freq = df_freq

        # Save the bins
        self.ws_bin = ws_bin
        self.category = category

    def get_energy_ratio(self, use_observed_freq = True, randomize_df = False):

        # Local copies
        df = self.df.copy()
        df_freq = self.df_freq.copy()
        ws_bin = self.ws_bin
        category = self.category

        # If resampling for boot-strapping
        if randomize_df:
            df = df.sample(frac=1,replace=True)


        # Remove bins with unmatched categories
        for ws in ws_bin:
            for cg_idx, cg in enumerate(category):
                if not(((df.ws_bin == ws) & (df.category==cg)).any()):
                    # print('Cat: %s is missing %d m/s, removing' % (cg, ws))
                    df = df[df.ws_bin!=ws]
                    df_freq = df_freq[df_freq.ws_bin!=ws]
        
        # Check for empty frame
        if df.shape[0]==0:
            return np.zeros(len(category)) * np.nan

        df_group = df[['ws_bin','category','ref_power','test_power']].groupby(['ws_bin','category']).mean()
        df_ref = df_group[['ref_power']].unstack()
        df_ref.columns = [c[1] for c in df_ref.columns]
        df_test = df_group[['test_power']].unstack()
        df_test.columns = [c[1] for c in df_test.columns]

        df_freq = df_freq[['ws_bin','observed_freq','annual_freq']].groupby(['ws_bin']).sum()

        results = np.zeros(len(category))
        if use_observed_freq:
            freq_signal = 'observed_freq'
        else:
            freq_signal = 'annual_freq'
        for catg_idx, catg in enumerate(category):
            ref_energy = compute_expectation(df_ref[catg],df_freq[freq_signal] )
            test_energy = compute_expectation(df_test[catg],df_freq[freq_signal] )
            results[catg_idx] = test_energy / ref_energy

        return results

    def get_1cat_energy_ratio(self, use_observed_freq = True, randomize_df = False):

        results = self.get_energy_ratio(use_observed_freq = use_observed_freq,  randomize_df = randomize_df)

        # Return with the difference as a convience
        return np.array([results[0]])

    def get_1cat_energy_ratio_with_range(self, use_observed_freq = True, N=100, percentiles = [10,90]):

        # Get central results
        results = self.get_1cat_energy_ratio(use_observed_freq = use_observed_freq, randomize_df = False)

        # Get a bootstrap sample of range
        bootstrap_results = np.zeros([N,1])
        for i in range(N):
            bootstrap_results[i,:] = self.get_1cat_energy_ratio(use_observed_freq = use_observed_freq, randomize_df = True)
        
        # Return the results in the order used in previous versions
        results_array = np.array([
            results[0],
            np.nanpercentile(bootstrap_results[:,0], percentiles)[0],
            np.nanpercentile(bootstrap_results[:,0], percentiles)[1]
        ])
        return(results_array)


    def get_2cat_energy_ratio(self, use_observed_freq = True, randomize_df = False):

        results = self.get_energy_ratio(use_observed_freq = use_observed_freq,  randomize_df = randomize_df)

        # Return with the difference as a convience
        return np.array([results[0], results[1], results[1] - results[0], 100 * (results[1] - results[0]) / results[0]])


    def get_2cat_energy_ratio_with_range(self, use_observed_freq = True, N=100, percentiles = [10,90]):

        # Get central results
        results = self.get_2cat_energy_ratio(use_observed_freq = use_observed_freq, randomize_df = False)

        # Get a bootstrap sample of range
        bootstrap_results = np.zeros([N,4])
        for i in range(N):
            bootstrap_results[i,:] = self.get_2cat_energy_ratio(use_observed_freq = use_observed_freq, randomize_df = True)
        
        # Return the results in the order used in previous versions
        results_array = np.array([
            results[0],
            np.nanpercentile(bootstrap_results[:,0], percentiles)[0],
            np.nanpercentile(bootstrap_results[:,0], percentiles)[1],
            results[1],
            np.nanpercentile(bootstrap_results[:,1], percentiles)[0],
            np.nanpercentile(bootstrap_results[:,1], percentiles)[1],
            results[2],
            np.nanpercentile(bootstrap_results[:,2], percentiles)[0],
            np.nanpercentile(bootstrap_results[:,2], percentiles)[1],
            results[3],
            np.nanpercentile(bootstrap_results[:,3], percentiles)[0],
            np.nanpercentile(bootstrap_results[:,3], percentiles)[1],
        ])

        return(results_array)

