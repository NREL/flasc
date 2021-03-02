import os
import pandas as pd
# import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np
import pickle
from floris.utilities import wrap_180, wrap_360
import floris.tools.plotting as wpt
# import seaborn as sns
import copy
from .layout import visualize_layout



class Floris_Scada():
    """
    Top-level class that contains a Floris-SCADA analysis



    Args:
        fi: A floris interface to model of SCADA farm
        scada_data_file: The name of the file containing the formatted SCADA data
        original_turbine_names: Names of the turbines according to original data, used in conjunction to FLORIS 0-index names

    Returns:
        TBD
    """

    def __init__(self, fi, scada_data_file='scada_data.p', original_turbine_names=None, base_floris_name='floris_original'):



        # Store FLORIS interface within class
        # The base case is strored in the first element of the array
        self.num_fi = 1
        self.fi_array = [fi]
        self.fi_name_array = [base_floris_name]
        self.fi_marker_array = ['.']

        # Save D for later conviencce
        self.D = self.fi_array[0].floris.farm.turbines[0].rotor_diameter

        # Save the original turbine names
        self.original_turbine_names = original_turbine_names

        # Floris turbine names are a simple range
        self.floris_turbine_names = np.arange(0,len(fi.layout_x),1)

        # Read in the scada data
        self.df = pickle.load(open(scada_data_file, "rb" ))
        self.df = self.df.reset_index(drop=True)

        # In case not yet defined, df_other needs to include a bin_cat column, just make empty for now
        self.df['bin_cat'] = np.nan

        # Similarly a dict of category order, must exist but is initially empty
        self.bin_cat_order = {}

        # Confirm num turbines match
        num_turbines_scada = self.get_num_turbines()
        num_turbines_floris = len(fi.layout_x)
        print(num_turbines_scada, num_turbines_floris)
        if not (num_turbines_scada == num_turbines_floris):
            raise ValueError("FLORIS and SCADA have different number of turbines")
        else:
            self.num_turbines = num_turbines_floris

        # If original turbine names suplied, likewise confirm match
        if original_turbine_names is not None:
            if not (len(original_turbine_names) == num_turbines_floris):
                raise ValueError("FLORIS and Original Turbine Names have different number of turbines")

        # Save the number of cases
        # self.num_samples = self.get_num_samples()

        # Build the wake table
        self.build_wake_table()

        # Finish
        print('Case of %d turbines and %d samples completed initialization' % (self.num_turbines,self.get_num_samples()))

    def get_num_turbines(self):
        num_turbines = len([c for c in self.df.columns if 'yaw_cor_' in c])
        return num_turbines

    def get_num_samples(self):
        return self.df.shape[0]

    def add_floris_input(self, fi, name, marker=None):

        # Probably should add a check the name is not repeated

        # Add an additional floris case
        self.num_fi = self.num_fi + 1
        self.fi_array.append(fi)
        self.fi_name_array.append(name)

        if marker is None:
            marker = ['*','s','o','^'][self.num_fi - 1]
        self.fi_marker_array.append(marker)

    def calc_floris(self):

        # Start by ensuring simple index for df
        self.df = self.df.reset_index(drop=True)

        for f_idx, fi in enumerate(self.fi_array):
            f_name = self.fi_name_array[f_idx]
            print('Processing FLORIS (%s) case %d of %d' % (f_name,f_idx + 1, self.num_fi))

            #  For now just do the baseline
            floris_pow = np.zeros([self.get_num_samples(), self.num_turbines])
            ws_array = self.df['ws']
            wd_array = self.df['wd']

            # Calculate the yaw angle as the difference from the corrected
            # Yaw position and the farm wind direction
            # df_yaw_offset = self.df_yaw_cor.copy(deep=True)
            df_yaw_offset = pd.DataFrame()
            for t in range(self.num_turbines):
                df_yaw_offset[t] = wrap_180(wd_array - self.df['yaw_cor_%03d' % t])

            df_yaw_offset = df_yaw_offset.fillna(0).clip(-30,30)

            # Loop over cases
            print('...Looping over cases')
            for idx in range(self.get_num_samples()):
                ws = ws_array[idx]
                wd = wd_array[idx]
                yaw = df_yaw_offset.iloc[idx].values
                # print('--',idx,ws,wd,yaw)
                fi.reinitialize_flow_field(wind_speed=ws,wind_direction=wd)
                fi.calculate_wake(yaw_angles=yaw)
                floris_pow[idx,:] = np.array(fi.get_turbine_power())/1000.
            print('...Done')

            # Append the FLORIS cases
            prefix = f_name + '_'
            for t in range(self.num_turbines):
                self.df[prefix + '%03d' % t] = floris_pow[:,t]

    # Define an approximate calc_floris() function
    def calc_floris_approx(self, ws_step=0.5, wd_step=1.0, ti_step=None):
        # Start by ensuring simple index for df
        self.df = self.df.reset_index(drop=True)

        for f_idx, fi in enumerate(self.fi_array):
            f_name = self.fi_name_array[f_idx]
            print('Processing FLORIS (%s) case %d of %d' % (f_name,f_idx + 1, self.num_fi))

            # Calculate mean of bins
            ws_array = self.df['ws']
            wd_array = self.df['wd']
            ws_array_approx = np.arange(np.round(np.min(ws_array)-ws_step/2.,1), np.round(np.max(ws_array)+1.5*ws_step,1), ws_step)
            wd_array_approx = np.arange(np.round(np.min(wd_array)-wd_step/2.,1), np.round(np.max(wd_array)+1.5*wd_step,1), wd_step)
            floris_pow_table = np.zeros((len(ws_array_approx), len(wd_array_approx), self.num_turbines))
            np.repeat([[np.array(np.repeat(0.0, len(wd_array_approx)))]], len(ws_array_approx), axis=1)

            N_raw = self.df.shape[0]
            N_approx = len(ws_array_approx)*len(wd_array_approx)

            if N_raw <= N_approx:
                print('Approximation would not reduce number of cases with the current settings (N_raw=' +str(N_raw)+', N_approx='+str(N_approx)+')')
                print('Calculating the exact solutions for this dataset. Avoiding any approximations.')
                self.calc_floris()
                return

            print('Reducing calculations from ' + str(self.df.shape[0]) + ' to ' + str(len(ws_array_approx)*len(wd_array_approx)) + ' cases.')

            print('...Looping over cases')
            for idx_ws in range(len(ws_array_approx)):
                ws = ws_array_approx[idx_ws]
                print('Calculating bins for ws = ' + str(ws) + ' m/s and ' + str(len(wd_array_approx)) + ' wind directions ['+str(idx_ws)+'/'+str(len(ws_array_approx))+'].')
                for idx_wd in range(len(wd_array_approx)):
                    wd = wd_array_approx[idx_wd]

                    # Check if any files in this bin
                    if any(
                        [a and b for a, b in zip(
                            [a and b for a, b in zip(ws_array >= ws-ws_step/2., ws_array < ws+ws_step/2.)],
                            [a and b for a, b in zip(wd_array >= wd-wd_step/2., wd_array < wd+wd_step/2.)]
                            )]
                    ):
                        fi.reinitialize_flow_field(wind_speed=ws, wind_direction=wd)
                        fi.calculate_wake()
                        floris_pow_table[idx_ws][idx_wd] = np.array(fi.get_turbine_power())/1000.
                    else:
                        floris_pow_table[idx_ws][idx_wd] = np.repeat(np.nan, self.num_turbines)

            # Now map individual data entries to full DataFrame
            print('Now mapping the precalculated solutions from FLORIS to the dataframe entries...')
            floris_pow =  np.nan * np.empty((len(ws_array), self.num_turbines))
            for si in range(len(ws_array_approx)):
                print('  Processing data with wind speeds in bin [%0.1f, %0.1f].' % (ws_array_approx[si]-ws_step/2., ws_array_approx[si]+ws_step/2.))
                diff_ws = ws_array - ws_array_approx[si]
                idx_df = np.where([a and b for a, b in zip(diff_ws >= -ws_step/2., diff_ws < ws_step/2.)])[0]
                if len(idx_df) > 0:
                    df_wd_array_subset = wd_array[idx_df]
                    for di in range(len(wd_array_approx)):
                        diff_wd = df_wd_array_subset - wd_array_approx[di]
                        idx_subset = np.where([a and b for a, b in zip(diff_wd >= -wd_step/2., diff_wd < wd_step/2.)])[0]
                        df_ids = idx_df[idx_subset]
                        floris_pow[df_ids] = floris_pow_table[si][di]
            print('Finished mapping')

            # Append the FLORIS cases
            prefix = f_name + '_'
            for t in range(self.num_turbines):
                self.df[prefix + '%03d' % t] = floris_pow[:, t]
        return

    def apply_category(self, category, mask_name='category', bin_cat_order = None):

        # Assign to df_other
        self.df['bin_cat'] = category

        # Build a mask around category
        mask = self.df['bin_cat'].notnull()
        self.apply_mask(mask, mask_name=mask_name)

        # Make the category order if not provided
        if bin_cat_order is None:
            # Use alphebetical order
            cats = sorted(self.df['bin_cat'].unique())
            bin_cat_order = dict(zip(cats,range(len(cats))))

        # Save the category order
        self.bin_cat_order = bin_cat_order
            

    def apply_mask(self, mask, mask_name = ''):
        print('...The mask %s keeps %.1f%% of the data' % (mask_name, 100. * np.sum(mask)/len(mask)))

        # Apply the filter
        self.df = self.df[mask]
        self.df = self.df.reset_index(drop=True)

    def apply_is_in_filter(self, col, limit_to, mask_name='is_in_filter'):

        print('Limiting data to when df.%s is in:' % col, limit_to)

        # Build up a mask
        mask = self.df[col].isin(limit_to)
        self.apply_mask(mask, mask_name=mask_name)

    def get_column_list(self, signal, turbine_list):

        # If last part is underscore, remove it
        if signal[-1] == '_':
            signal = signal[:-1]
        
        return [signal + "_%03d" % t for t in turbine_list]

    def define_ws_by_turbine_list(self, turbine_list):
        self.df['ws'] = self.df[self.get_column_list('ws',turbine_list)].mean(axis=1)

    def define_wd_by_turbine_list(self, turbine_list):
        wd_all = self.df[self.get_column_list('wd_cor', turbine_list)]
        wds_x_sum = np.cos(wd_all * np.pi / 180.).sum(axis=1)
        wds_y_sum = np.sin(wd_all * np.pi / 180.).sum(axis=1)
        wds_mean = wrap_360(np.arctan2(wds_y_sum, wds_x_sum) * 180. / np.pi)
        self.df['wd'] = wds_mean

    def get_status_mask(self, turbine_list):
        return self.df[self.get_column_list('status',turbine_list)].min(axis=1) == 1

    def get_self_status_mask(self, turbine_list):
        return self.df[self.get_column_list('self_status', turbine_list)].min(axis=1) == 1

    def get_column_mean_for_turbine_list(self, column_prefix, turbine_list):
        return self.df[self.get_column_list(column_prefix,turbine_list)].mean(axis=1)

    def get_column_subset_for_turbine_list(self, column_prefix, turbine_list):
        return self.df[self.get_column_list(column_prefix,turbine_list)]

    # def get_mean_power_by_turbine_list(self, turbine_list):
    #     return self.df[self.get_column_list('pow',turbine_list)].mean(axis=1)

    # def get_mean_power_by_turbine_list_floris(self, floris_prefix, turbine_list):

    #     return self.df[self.get_column_list(floris_prefix,turbine_list)].mean(axis=1)


    def show_layout(self,ax=None,show_wake_lines=False,limit_dist=None,turbine_face_north=False,show_wake_count_direction=None):

        fi = self.fi_array[0]

        # Grab the wake_array
        if show_wake_count_direction is not None:
            wake_array = fi.floris.farm.turbine_map.number_of_wakes_iec(show_wake_count_direction)
            wake_array = [w[1] for w in wake_array]
        else:
            wake_array = None

        visualize_layout(fi.layout_x,fi.layout_y,self.original_turbine_names, self.D,
                        ax=ax,
                        show_wake_lines=show_wake_lines,
                        limit_dist=limit_dist,
                        turbine_face_north=turbine_face_north,
                        show_wake_count_direction=show_wake_count_direction,
                        wake_array=wake_array)

    def build_wake_table(self):

        fi = self.fi_array[0]

        print('...Building wake table')
        
        # Initialize wake table
        self.wake_table = dict()

        # Loop over some wind direction
        wind_dirs = np.arange(0,361,1.)

        for wd_idx, wd in enumerate(wind_dirs):

            wake_array = fi.floris.farm.turbine_map.number_of_wakes_iec(wd)
            self.wake_table[wd] = np.array([w[1] for w in wake_array])



    # def dump_to_csv_wide(self,filename='data.csv'):
    #     # Function for dumping the SCADA data to CSV, can be useful for sandstorm inspection

    #     df_yaw = self.df_yaw.copy(deep=True)
    #     df_yaw.columns = ['yaw_%02d' % t for t in df_yaw.columns]

    #     df_pow = self.df_pow.copy(deep=True)
    #     df_pow.columns = ['pow_%02d' % t for t in df_pow.columns]

    #     df_ws = self.df_ws.copy(deep=True)
    #     df_ws.columns = ['ws_%02d' % t for t in df_ws.columns]

    #     df_wd = self.df_wd.copy(deep=True)
    #     df_wd.columns = ['wd_%02d' % t for t in df_wd.columns]

        
    #     df_ti = self.df_ti.copy(deep=True)
    #     print(df_ti.columns)
    #     df_ti.columns = ['ti_%02d' % t for t in df_ti.columns]

    #     df_vane = self.df_vane.copy(deep=True)
    #     df_vane.columns = ['vane_%02d' % t for t in df_vane.columns]

    #     df_met = self.df_met.copy(deep=True)


    #     df = pd.concat([df_yaw,df_pow,df_ws,df_wd,df_ti,df_vane,df_met],axis=1)
    #     df.to_csv(filename)

    # def dump_to_csv_tall(self,filename='data.csv'):
    #     # Function for dumping the SCADA data to CSV, can be useful for sandstorm inspection

    #     df_yaw = self.df_yaw.stack().reset_index()
    #     df_yaw.columns = ['sample','turbine','yaw']
        
    #     df_pow = self.df_pow.stack().reset_index()
    #     df_pow.columns = ['sample','turbine','power']

    #     df_ws = self.df_ws.stack().reset_index()
    #     df_ws.columns = ['sample','turbine','ws']       
        
    #     df_wd = self.df_wd.stack().reset_index()
    #     df_wd.columns = ['sample','turbine','wd']

    #     df_ti = self.df_ti.stack().reset_index()
    #     df_ti.columns = ['sample','turbine','ti']

    #     df_vane = self.df_vane.stack().reset_index()
    #     df_vane.columns = ['sample','turbine','vane']

    #     df_met = self.df_met.copy(deep=True)
    #     df_met.columns = ['met_ws','met_wd','met_ti']
    #     df_met.index.name = 'sample'
    #     df_met = df_met.reset_index()


    #     df = pd.merge(df_yaw,df_pow,how='outer',on=['sample','turbine'])
    #     df = pd.merge(df,df_ws,how='outer',on=['sample','turbine'])
    #     df = pd.merge(df,df_wd,how='outer',on=['sample','turbine'])
    #     df = pd.merge(df,df_ti,how='outer',on=['sample','turbine'])
    #     df = pd.merge(df,df_vane,how='outer',on=['sample','turbine'])
    #     df = pd.merge(df,df_met,how='outer',on=['sample'])


    #     df['turbine'] = 'f_' + df.turbine.astype(str)
    #     print(df.head(20))

    #     # df = pd.concat([df_yaw,df_pow,df_ws,df_wd,df_ti,df_vane,df_met],axis=1)
    #     df.to_csv(filename)