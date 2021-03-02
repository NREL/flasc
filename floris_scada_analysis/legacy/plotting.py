import pandas as pd
import matplotlib.pyplot as plt
# import floris.tools as wfct
import numpy as np
# import dill as pickle
# from floris.utilities import wrap_180, wrap_360
# import floris_scada.utility as ut
# from floris.tools.plotting import data_plot
# import floris.tools.plotting as wpt
import seaborn as sns
from itertools import product
from matplotlib.patches import  Polygon
import matplotlib.patches as patches

# #Plotly stuff
# import plotly_express as px
# import plotly.io as pio
# pio.templates.default = "plotly_white"

# # Define some colors
# base_color_fixed='#332288'#0F2080'
# con_color_fixed='#117733'  #'#A95AA1'

# base_color_light='#88CCEE'
# con_color_light='#44AA99'

# opt_color='#CC6677'

# Define some colors
base_color_fixed='#4477AA'#0F2080'
con_color_fixed='#228833'  #'#A95AA1'

base_color_light='#66CCEE'
con_color_light='#CCBB44'

opt_color='#EE6677'


def plot_energy_ratio(df, channel, ax, label, color='k', ls='-'):

    ax.plot(df.wd_bin, df[channel], ls = ls, color=color, label=label)

def plot_energy_ratio_range(df, channel, ax, label='_nolegend_', color='k'):

    ax.fill_between(df.wd_bin, df[channel + '_l'], df[channel + '_u'],color=color, alpha=0.5,label=label)

def plot_baseline(df, ax, base_color=base_color_fixed, ls='-', baseline_label='baseline'):
    plot_energy_ratio(df, 'baseline',ax,baseline_label,base_color,ls)
    plot_energy_ratio_range(df, 'baseline',ax,'_nolegend_',base_color)


def plot_baseline_and_con(df, ax, base_color=base_color_fixed, con_color=con_color_fixed,ls='-',
     baseline_label='baseline', controlled_label='controlled'):

    plot_energy_ratio(df, 'baseline',ax,baseline_label,base_color,ls)
    plot_energy_ratio(df, 'controlled',ax,controlled_label,con_color,ls)
    plot_energy_ratio_range(df, 'baseline',ax,'_nolegend_',base_color)
    plot_energy_ratio_range(df, 'controlled',ax,'_nolegend_',con_color)

def plot_diff(df, ax, label='Difference', color='k'):
    plot_energy_ratio(df, 'diff',ax,label=label, color=color)
    plot_energy_ratio_range(df, 'diff',ax,label='_nolegend_', color=color)

def plot_percent_diff(df, ax, label='Percent Difference', color='k'):
    plot_energy_ratio(df, 'per',ax,label=label, color=color)
    plot_energy_ratio_range(df, 'per',ax,label='_nolegend_', color=color)

def band_plot(x, y, color, label="_nolegend_", use_median=False):
    
    df = pd.DataFrame({"x": x, "y": y})
    x_values = np.array(sorted(df.x.unique()))
    y_values = np.zeros_like(x_values)
    std_values = np.zeros_like(x_values)
    count_values = np.zeros_like(x_values)
    for idx, x in enumerate(x_values):
        df_sub = df[df.x==x]
        count_values[idx] = df.shape[0]
        if df.shape[0] == 0:
            std_values[idx] = 0
            y_values[idx] = np.nan
        else:
            std_values[idx] = np.std(df_sub.y)
            if use_median:
                y_values[idx] = np.median(df_sub.y)
            else:
                y_values[idx] = np.mean(df_sub.y)

    plt.scatter(
        x_values, y_values, color=color, s=count_values,   label="_nolegend_" ,alpha=0.6,
    )
    plt.plot(x_values, y_values, label=label, color=color)
    plt.fill_between(
            x_values, y_values-std_values, y_values+std_values, alpha=0.2, color=color, label="_nolegend_"
    )



    

