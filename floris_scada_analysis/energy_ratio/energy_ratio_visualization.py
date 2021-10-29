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
import pandas as pd
import os

import matplotlib.pyplot as plt

from floris import tools as wfct


def plot(energy_ratios, labels=None):
    """This function plots energy ratios against the reference wind
    direction. The plot may or may not include uncertainty bounds,
    depending on the information contained in the provided energy ratio
    dataframes.

    Args:
        energy_ratios ([iteratible]): List of Pandas DataFrames containing
                the energy ratios for each dataset, respectively. Each
                entry in this list is a Dataframe containing the found
                energy ratios under the prespecified settings, contains the
                columns:
                    * wd_bin: The mean wind direction for this bin
                    * N_bin: Number of data entries in this bin
                    * baseline: Nominal energy ratio value (without UQ)
                    * baseline_l: Lower bound for energy ratio. This
                        value is equal to baseline without UQ and lower
                        with UQ.
                    * baseline_u: Upper bound for energy ratio. This
                        value is equal to baseline without UQ and higher
                        with UQ.
        labels ([iteratible], optional): Label for each of the energy ratio
            dataframes. Defaults to None.

    Returns:
        fig ([plt.Figure]): Figure in which energy ratios are plotted.
        ax ([iteratible]): List of axes in the figure with length 2.
    """
    # Format inputs if single case is inserted vs. lists
    if not isinstance(energy_ratios, (list, tuple)):
        energy_ratios = [energy_ratios]
        if isinstance(labels, str):
            labels = [labels]

    if labels is None:
        labels = ["Nominal" for _ in energy_ratios]
        uq_labels = ["Confidence bounds" for _ in energy_ratios]
    else:
        uq_labels = ["%s confidence bounds" % lb for lb in labels]

    N = len(energy_ratios)
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 5))

    # Calculate bar width for bin counts
    bar_width = (0.7 / N) * np.min(
        [np.diff(er["wd_bin"])[0] for er in energy_ratios]
    )

    for ii, df in enumerate(energy_ratios):
        df = df.copy()

        # Get x-axis values
        x = np.array(df["wd_bin"], dtype=float)

        # Add NaNs to avoid connecting plots over gaps
        dwd = np.min(x[1::] - x[0:-1])
        jumps = np.where(np.diff(x) > dwd * 1.50)[0]
        if len(jumps) > 0:
            df = df.append(
                pd.DataFrame(
                    {
                        'wd_bin': x[jumps] + dwd / 2.0,
                        "N_bin": [0] * len(jumps),
                    }
                )
            )
            df = df.iloc[np.argsort(df['wd_bin'])].reset_index(drop=True)
            x = np.array(df["wd_bin"], dtype=float)

        # Plot horizontal black line at 1.
        xlims = [np.min(x) - 4., np.max(x) + 4.]
        ax[0].plot(xlims, [1., 1.], color='black')

        # Plot energy ratios
        ax[0].plot(x, df["baseline"], '-o', markersize=3., label=labels[ii])

        # Plot uncertainty bounds from bootstrapping, if applicable
        has_uq = (np.max(np.abs(df['baseline'] - df['baseline_l'])) > 0.001)
        if has_uq:
            ax[0].fill_between(x, df["baseline_l"], df["baseline_u"],
                               alpha=0.25, label=uq_labels[ii])

        # Plot the bin count
        ax[1].bar(x-(ii-N/2)*bar_width, df["N_bin"], width=bar_width)

    # Format the energy ratio plot
    ax[0].set_ylabel('Energy ratio (-)')
    ax[0].legend()
    ax[0].grid(b=True, which='major', axis='both', color='gray')
    ax[0].grid(b=True, which='minor', axis='both', color='lightgray')
    ax[0].minorticks_on()
    plt.grid(True)

    if labels[0] is not None:
        ax[0].legend()

    # Format the bin count plot
    ax[1].grid(b=True, which='major', axis='both', color='gray')
    ax[1].grid(b=True, which='minor', axis='both', color='lightgray')
    ax[1].set_xlabel('Wind direction (deg)')
    ax[1].set_ylabel('Number of data points (-)')

    # Enforce a tight layout
    plt.tight_layout()

    return fig, ax


def table_analysis(df_list, name_list, t_list, wd_bins, ws_bins, excel_filename, fi=None):
    
    
    # Save some useful info
    header_row = 2
    first_data_row = header_row + 1
    first_data_col = 1
    
    # If t_list is none put in all possible turbines
    # TODO
    
    # Save the basename, always assumed to be the first dataframe
    basename = name_list[0]
    
    # Stitch the dataframes together
    df_full = pd.DataFrame()
    for dfx, namex in zip(df_list, name_list):
        df_temp = dfx.copy()
        df_temp['name'] = namex
        df_full = df_full.append(df_temp)
        
    
    # Cut wd and ws into bins
    df_full['wd_bin'] = pd.cut(df_full.wd,wd_bins)
    df_full['ws_bin'] = pd.cut(df_full.ws,ws_bins)
    
    # Save the original bins
    df_save_bin = df_full[['wd_bin','ws_bin']]
    
    # Drop out of range
    df_full = df_full.dropna(subset=['wd_bin','ws_bin'])
    
    # Sort by bins
    df_full = df_full.sort_values(['name','wd_bin','ws_bin'])
    
    # Add a bin count column
    df_full['bin_count'] = 1
    
    # So we don't lose precision multiply TI by 100
    df_full['ti'] = 100 * df_full['ti']
    
    # Convert all to sums and means
    df_group = df_full.groupby(['wd_bin','ws_bin','name']).agg([np.sum,np.mean])
    
    # Flatten the columns
    df_group.columns = ["_".join(c) for c in df_group.columns]
    
    # Spin the name out to the columns
    df_group = df_group.unstack()
    
    # Flatten the columns
    df_group.columns = ["_".join(c) for c in df_group.columns]
    
    # Round the numerical columns to one decimal place
    df_group = df_group.round(1)
    
    # Reset the index
    df_group = df_group.reset_index()
    
    # Put together the final table
    df_table = df_group[['wd_bin','ws_bin']].copy()
    
    # Add the bin counts, 
    for n in name_list:
        df_table['bin_count_sum_%s' % n] = df_group['bin_count_sum_%s' % n]

    # Add a balanced bin count, the sum of the others, unless any are 0, in which case 0

    # Actually let's try it this way, use the minimum
    bin_cols = ['bin_count_sum_%s' % n for n in name_list]
    df_table['bin_balanced'] = df_table[bin_cols].min(axis=1)


    # Add mean wind speeds, and reference power
    for n in name_list:
        df_table['ws_mean_%s' % n] = df_group['ws_mean_%s' % n]
    for n in name_list:
        df_table['ti_mean_%s' % n] = df_group['ti_mean_%s' % n] / 100. # Back to decimal
    
    # Reference energy and power, and balanced energy
    for n in name_list:
        df_table['ref_pow_%s' % n] = df_group['pow_ref_sum_%s' % n] / df_group['bin_count_sum_%s' % n]
        df_table['ref_energy_%s' % n] = df_group['pow_ref_sum_%s' % n]
        df_table['ref_energy_balanced_%s' % n] = df_table['ref_pow_%s' % n] * df_table['bin_balanced']
    
    # Add an empty column
    df_table['___'] = None
    
    # Add the rest via turbine
    for t in t_list:
        for n in name_list:

            # Add the power
            df_table['pow_%03d_%s' % (t,n)] = df_group['pow_%03d_sum_%s' % (t,n)] / df_table['bin_count_sum_%s' % n]

            # Add the energy
            df_table['energy_%03d_%s' % (t,n)] = df_group['pow_%03d_sum_%s' % (t,n)]

            # Add the balanced enegy
            df_table['energy_balanced_%03d_%s' % (t,n)] = df_table['pow_%03d_%s' % (t,n)] * df_table['bin_balanced']
            
            # Add the energy ratio
            df_table['er_%03d_%s' % (t,n)] = np.round(df_table['energy_%03d_%s' % (t,n)] / df_table['ref_energy_%s' % n],3)

            # Add the balanced energy ratio
            df_table['er_balanced_%03d_%s' % (t,n)] = np.round(df_table['energy_balanced_%03d_%s' % (t,n)] / df_table['ref_energy_balanced_%s' % n],3)
    
            # Only do this if not first
            if not n == name_list[0]:
                # Add the change in energy ratio from baseline (whichever name is first)
                df_table['er_change_%03d_%s' % (t,n)] = np.round(1 * (df_table['er_%03d_%s' % (t,n)] - df_table['er_%03d_%s' % (t,basename)]) / df_table['er_%03d_%s' % (t,basename)],3) 
                # Add the change in energy ratio from baseline (whichever name is first)
                df_table['er_change_balanced_%03d_%s' % (t,n)] = np.round(1 * (df_table['er_balanced_%03d_%s' % (t,n)] - df_table['er_balanced_%03d_%s' % (t,basename)]) / df_table['er_balanced_%03d_%s' % (t,basename)],3) 
    
        # Add an empty column
        df_table['___%d' % t] = None
        
    
    
    # Add the totals by direction
    df_table_final = pd.DataFrame()
    for wd_bin in df_table.wd_bin.unique():
        df_sub_pre = df_table[df_table.wd_bin == wd_bin]
        
        # Make a new df which is a sum of the other
        df_sub = df_sub_pre.append(df_sub_pre.sum(numeric_only=True), ignore_index=True)#.to_frame()
        
        # Go through the columns of this frame and fix the last row
        last_row = df_sub.shape[0] -1
        
        # Fix ws and wd
        df_sub.loc[last_row,'wd_bin'] = df_sub.loc[last_row-1,'wd_bin']
        df_sub['ws_bin'] = df_sub['ws_bin'].astype(str)
        df_sub.loc[last_row,'ws_bin'] = 'TOTALS'

        # Remove the total for reference power 
        for n in name_list:
            df_sub.loc[last_row,'ref_pow_%s' % n] = np.nan
        
        # Correct ws and ti to be a mean
        for n in name_list:
            df_sub.loc[last_row,'ws_mean_%s' % n] = np.round(np.sum(df_sub_pre['bin_count_sum_%s' % n] * df_sub_pre['ws_mean_%s' % n]) / df_sub_pre['bin_count_sum_%s' % n].sum(),1)
            df_sub.loc[last_row,'ti_mean_%s' % n] = np.round(np.sum(df_sub_pre['bin_count_sum_%s' % n] * df_sub_pre['ti_mean_%s' % n]) / df_sub_pre['bin_count_sum_%s' % n].sum(),3)
            

        # Correct the energy ratios
        for t in t_list:
            for n in name_list:

                # Not sure what the total power should be so just to nan
                df_sub.loc[last_row,'pow_%03d_%s' % (t,n)] = np.nan
                
                # Recompute the energy ratio (overwrite the sum)
                df_sub['er_%03d_%s' % (t,n)] = np.round(df_sub['energy_%03d_%s' % (t,n)] / df_sub['ref_energy_%s' % n],3)
                df_sub['er_balanced_%03d_%s' % (t,n)] = np.round(df_sub['energy_balanced_%03d_%s' % (t,n)] / df_sub['ref_energy_balanced_%s' % n],3)
                # df_sub.loc[last_row,'er_%03d_%s' % (t,n)] = df_sub.loc[last_row,'energy_%03d_%s' % (t,n)] / df_sub.loc[last_row,'ref_energy_%s' % n]
        
                # Only do this if not first
                if not n == name_list[0]:
                    # Recompute the change in energy ratio (overwrite the sum)
                    df_sub['er_change_%03d_%s' % (t,n)] = np.round(1 * (df_sub['er_%03d_%s' % (t,n)] - df_sub['er_%03d_%s' % (t,basename)]) / df_sub['er_%03d_%s' % (t,basename)],3) 
                    df_sub['er_change_balanced_%03d_%s' % (t,n)] = np.round(1 * (df_sub['er_balanced_%03d_%s' % (t,n)] - df_sub['er_balanced_%03d_%s' % (t,basename)]) / df_sub['er_balanced_%03d_%s' % (t,basename)],3) 

        # Add an empty row
        df_sub = df_sub.append(pd.DataFrame([[''] * len(df_sub.columns)], columns=df_sub.columns))
        
        # Append to the final
        df_table_final = df_table_final.append(df_sub)
        
    
    # Write out the dataframe with xslxwriter
    writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
    df_table_final.to_excel(writer , index=False, sheet_name= 'results',startcol=first_data_col,startrow=header_row)
    workbook = writer.book
    worksheet = writer.sheets['results']
    
    # Set some formatting
    
    # Make change and TI into a percentage
    # Adding percentage format.
    fmt_rate = workbook.add_format({
     "num_format" : '%0.0' , "bold" : False
    })
    cols = df_table_final.columns
    change_list = [i for i in range(len(cols)) if ('change' in cols[i]) or ('ti_' in cols[i])]
    for c in change_list:
        worksheet.set_column(c+first_data_col,c+first_data_col,10,fmt_rate)
        
    # Make the seperator columns very narrow and black
    fmt_black = workbook.add_format({
     "fg_color" : '#000000'
    })
    change_list = [i for i in range(len(cols)) if '___' in cols[i]]
    for c in change_list:
        worksheet.set_column(c+first_data_col,c+first_data_col,1,fmt_black)
        
    # Add data bars to the bins counts
    change_list = [i for i in range(len(cols)) if 'bin' in cols[i]]
    for c in change_list:
        worksheet.conditional_format(first_data_row,c+first_data_col,df_table_final.shape[0]+first_data_row,c+first_data_col, {'type': 'data_bar','max_value':100})
        
    # Add color to the change columns
    change_list = [i for i in range(len(cols)) if 'change' in cols[i]]

    for c in change_list:
        #worksheet.conditional_format(first_data_row,c+first_data_col,df_table_final.shape[0]+first_data_row,c+first_data_col, {'type': 'data_bar','bar_axis_position': 'middle','bar_negative_border_color_same': True})

        worksheet.conditional_format(first_data_row,c+first_data_col,df_table_final.shape[0]+first_data_row,c+first_data_col, {'type': '3_color_scale',
                                                                     'min_value': -1.0,
                                                                     'min_type':'num',
                                                                     'max_value': 1.0,
                                                                     'mid_value':0.0,
                                                                     'mid_type':'num',
                                                                     'min_color':'#FF0000',
                                                                     'mid_color':'#FFFFFF',
                                                                     'max_color':'#00FF00',
                                                                     'max_type':'num'})

    # Add color to energy ratios
    change_list = [i for i in range(len(cols)) if ('er_' in cols[i]) and not ('change' in cols[i])]
    for c in change_list:
        # worksheet.conditional_format(first_data_row,c+first_data_col,df_table_final.shape[0]+first_data_row,c+first_data_col, {'type': '3_color_scale',
        #                                                              'min_value': 0.25,
        #                                                              'min_type':'num',
        #                                                              'max_value': 1.0,
        #                                                              'max_type':'num'})
        worksheet.conditional_format(first_data_row,c+first_data_col,df_table_final.shape[0]+first_data_row,c+first_data_col, {'type': '3_color_scale',
                                                                     'min_value': 0.25,
                                                                     'min_type':'num',
                                                                     'max_value': 2.0,
                                                                     'mid_value':1.0,
                                                                     'mid_type':'num',
                                                                     'min_color':'#0000FF',
                                                                     'mid_color':'#FFFFFF',
                                                                     'max_color':'#00FF00',
                                                                     'max_type':'num'})
    
    
    # Header
    # Adding formats for header row.
    fmt_header = workbook.add_format({
     'bold': True,
     'text_wrap': True,
     'valign': 'top',
     'fg_color': '#5DADE2',
     'font_color': '#FFFFFF',
     'border': 1})
    for col , value in enumerate(df_table_final.columns.values):
         worksheet.write(header_row, col+first_data_col, value, fmt_header)
            
            
    # If an fi is provided, use it to make layout images to help with directions of things
    if fi is not None:
        
        # Use FI to show the direction
        
        # Make that first colum wide
        worksheet.set_column('A:A', 30)
        
        #check image folder
        if not os.path.exists('images'):
            os.makedirs('images')
            
        # For each bin were checking make an image
        sort_df = df_save_bin.sort_values(['wd_bin','ws_bin']).dropna()
        num_ws_bin = len(sort_df.ws_bin.unique())
        for wdb_idx, wdb in enumerate(sort_df.wd_bin.unique()):
            wd_arrow = wdb.mid # Put arrow in middle of bin
            fig, ax = plt.subplots(figsize=(2,2))
            fi.reinitialize_flow_field(wind_direction=wd_arrow,wind_speed=8.)
            fi.calculate_wake()
            hor_plane = fi.get_hor_plane()
            wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
            im_name = os.path.join('images','wd_%03d.png' % wd_arrow)
            fig.savefig(im_name,bbox_inches='tight')
            
            # Insert the figure
            worksheet.insert_image(first_data_row + wdb_idx * (num_ws_bin + 2),0, im_name)
            
        # Next use FI to indicate the turbine
        
        # Make the first row bigger
        worksheet.set_row(0, 120)
        
        # Get a list of blank columns indicating turbine starts
        blank_cols = [i for i in range(len(cols)) if '___' in cols[i]]
        for t_idx, t in enumerate(t_list):
            
            # Plot the layout
            fig, ax = plt.subplots(figsize=(3,2))
            fi.vis_layout(ax=ax)
            ax.plot(fi.layout_x[t],fi.layout_y[t],'mo',ms=25)
            im_name = os.path.join('images','layout_%03d.png' % t)
            fig.savefig(im_name,bbox_inches='tight')
            
            # Find the column
            bc = blank_cols[t_idx]
            worksheet.insert_image(0,bc+1, im_name)
            
    
    # Freeze the panes
    worksheet.freeze_panes(first_data_row,first_data_col)
    
    writer.save()
    
