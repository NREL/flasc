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


def plot(energy_ratios, labels=None, hide_uq_labels=True):
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
        hide_uq_labels (bool, optional): If true, do not specifically label
            the confidence intervals in the plot

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

    if hide_uq_labels:
        uq_labels = ['_nolegend_' for l in uq_labels]

    N = len(energy_ratios)
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 5))

    # Calculate bar width for bin counts
    bar_width = (0.7 / N) * np.min(
        [er["wd_bin"].diff().min() for er in energy_ratios]
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
                        "wd_bin": x[jumps] + dwd / 2.0,
                        "N_bin": [0] * len(jumps),
                    }
                )
            )
            df = df.iloc[np.argsort(df["wd_bin"])].reset_index(drop=True)
            x = np.array(df["wd_bin"], dtype=float)

        # Plot horizontal black line at 1.
        xlims = [np.min(x) - 4.0, np.max(x) + 4.0]
        ax[0].plot(xlims, [1.0, 1.0], color="black")

        # Plot energy ratios
        ax[0].plot(x, df["baseline"], "-o", markersize=3.0, label=labels[ii])

        # Plot uncertainty bounds from bootstrapping, if applicable
        has_uq = np.max(np.abs(df["baseline"] - df["baseline_lb"])) > 0.001
        if has_uq:
            ax[0].fill_between(
                x,
                df["baseline_lb"],
                df["baseline_ub"],
                alpha=0.25,
                label=uq_labels[ii],
            )

        # Plot the bin count
        ax[1].bar(x - (ii - N / 2) * bar_width, df["bin_count"], width=bar_width)

    # Format the energy ratio plot
    ax[0].set_ylabel("Energy ratio (-)")
    ax[0].legend()
    ax[0].grid(b=True, which="major", axis="both", color="gray")
    ax[0].grid(b=True, which="minor", axis="both", color="lightgray")
    ax[0].minorticks_on()
    plt.grid(True)

    if labels[0] is not None:
        ax[0].legend()

    # Format the bin count plot
    ax[1].grid(b=True, which="major", axis="both", color="gray")
    ax[1].grid(b=True, which="minor", axis="both", color="lightgray")
    ax[1].set_xlabel("Wind direction (deg)")
    ax[1].set_ylabel("Number of data points (-)")

    # Enforce a tight layout
    plt.tight_layout()

    return fig, ax


def table_analysis(
    df_list,
    fout_xlsx,
    hide_bin_count_columns=False,
    hide_ws_ti_columns=False,
    hide_pow_columns=False,
    hide_unbalanced_cols=True,
    fi=None
    ):
    # Save some useful info
    header_row = 2
    first_data_row = header_row + 1
    first_data_col = 1

    # Extract variables
    ws_step = df_list[0]["er_ws_step"]
    wd_bin_width = df_list[0]["er_wd_bin_width"]

    # Extract relevant details
    name_list = [df["name"] for df in df_list]
    df_per_wd_bin_list = [
        d["er_results_info_dict"]["df_per_wd_bin"]for d in df_list
    ]
    df_per_ws_bin_list = [
        d["er_results_info_dict"]["df_per_ws_bin"] for d in df_list
    ]
    test_turbines = np.array(df_list[0]["er_test_turbines"], dtype=int)

    # Append column names with data label for df_per_wd_bin
    for ii, df in enumerate(df_per_wd_bin_list):
        name = name_list[ii]
        df.columns = ["{:s}_{:s}".format(c, name) for c in df.columns]

    # Append column names with data label for df_per_ws_bin
    for ii, df in enumerate(df_per_ws_bin_list):
        name = name_list[ii]
        df = df.reset_index(drop=False).set_index(["wd_bin", "ws_bin"])
        df.columns = ["{:s}_{:s}".format(c, name) for c in df.columns]
        df_per_ws_bin_list[ii] = df

    # Concatenate information from different dataframes
    df_per_wd_bin = pd.concat(df_per_wd_bin_list, axis=1)
    df_per_ws_bin = pd.concat(df_per_ws_bin_list, axis=1)

    # Merge df_per_wd_bin information into df_per_ws_bin
    df_per_ws_bin = df_per_ws_bin.reset_index(drop=False).set_index("wd_bin")
    df_per_wd_bin["ws_bin"] = 999999.9  # very high number
    df_merged = pd.concat([df_per_ws_bin, df_per_wd_bin])
    df_merged = df_merged.sort_values(by=["wd_bin", "ws_bin"])
    df_merged = df_merged.reset_index(drop=False)

    wd_intervals = [pd.Interval(a, b, "left") for a, b in zip(
        df_merged["wd_bin"] - wd_bin_width / 2.0,
        df_merged["wd_bin"] + wd_bin_width / 2.0
        )
    ]
    ws_intervals = [pd.Interval(a, b, "left") for a, b in zip(
        df_merged["ws_bin"] - ws_step / 2.0,
        df_merged["ws_bin"] + ws_step / 2.0
        )
    ]

    df_table = pd.DataFrame(
        {
            "wd_bin": wd_intervals,
            "ws_bin": ws_intervals,
        }
    )

    # Overwrite placeholder large numbers with new interval
    total_ids = [i.right >= 999999.9 for i in ws_intervals]
    df_table.loc[total_ids, "ws_bin"] = "TOTALS"
    df_merged.loc[total_ids, "ws_bin"] = "TOTALS"

    # Add bin counts for the dataframes
    cols = ["bin_count_{:s}".format(n) for n in name_list]
    for c in cols:
        df_table[c] = df_merged[c].fillna(0).astype(int)

    # Add balanced bin count per ws and in total
    df_table["bin_count_balanced"] = df_table[cols].min(axis=1).astype(int)
    df_table.loc[total_ids, "bin_count_balanced"] = int(0)
    Ntot = df_table.groupby(["wd_bin"])["bin_count_balanced"].sum()
    df_table.loc[total_ids, "bin_count_balanced"] = np.array(Ntot, dtype=int)

    df_merged["bin_count_balanced_tot"] = df_table.loc[total_ids, "bin_count_balanced"]
    df_merged["bin_count_balanced_tot"] = df_merged["bin_count_balanced_tot"].bfill().astype(int)

    # add ws_mean and ti_mean for all dataframes
    for col in ["ws_mean", "ti_mean"]:
        for n in name_list:
            c = "{:s}_{:s}".format(col, n)
            if c in df_merged.columns:
                df_table[c] = df_merged[c]

    # Add reference power and energy
    bin_totals = np.array(df_merged["bin_count_balanced_tot"])
    for n in name_list:
        pow_mean = df_merged["pow_ref_mean_{:s}".format(n)]
        energy_unbal = df_merged["energy_ref_unbalanced_{:s}".format(n)]
        energy_bal_norm = df_merged["energy_ref_balanced_norm_{:s}".format(n)]
        energy_bal = bin_totals * energy_bal_norm

        df_table["ref_pow_{:s}".format(n)] = pow_mean
        df_table["ref_energy_unbalanced_{:s}".format(n)] = energy_unbal
        df_table["ref_energy_balanced_{:s}".format(n)] = energy_bal

        # Fill empty entries with 0.0 for energy
        df_table["ref_energy_unbalanced_{:s}".format(n)] = (
            df_table["ref_energy_unbalanced_{:s}".format(n)].fillna(0.0)
        )
        df_table["ref_energy_balanced_{:s}".format(n)] = (
            df_table["ref_energy_balanced_{:s}".format(n)].fillna(0.0)
        )

    # Add empty column/spacer
    df_table["___"] = None

    # Add test power and energy
    bin_totals = np.array(df_merged["bin_count_balanced_tot"])
    for n in name_list:
        pow_mean = df_merged["pow_test_mean_{:s}".format(n)]
        energy_unbal = df_merged["energy_test_unbalanced_{:s}".format(n)]
        energy_bal_norm = df_merged["energy_test_balanced_norm_{:s}".format(n)]
        energy_bal = bin_totals * energy_bal_norm
        energy_ratio = df_merged["energy_ratio_unbalanced_{:s}".format(n)]
        energy_ratio_bal = df_merged["energy_ratio_balanced_{:s}".format(n)]

        df_table["test_pow_{:s}".format(n)] = pow_mean
        df_table["test_energy_unbalanced_{:s}".format(n)] = energy_unbal
        df_table["test_energy_balanced_{:s}".format(n)] = energy_bal
        df_table["energy_ratio_unbalanced_{:s}".format(n)] = energy_ratio
        df_table["energy_ratio_balanced_{:s}".format(n)] = energy_ratio_bal

        # Fill empty entries with 0.0 for energy
        df_table["test_energy_unbalanced_{:s}".format(n)] = (
            df_table["test_energy_unbalanced_{:s}".format(n)].fillna(0.0)
        )
        df_table["test_energy_balanced_{:s}".format(n)] = (
            df_table["test_energy_balanced_{:s}".format(n)].fillna(0.0)
        )

    # Define change in unbalanced and balanced energy ratios
    bl = df_table["energy_ratio_unbalanced_{:s}".format(name_list[0])]
    bl_bal = df_table["energy_ratio_balanced_{:s}".format(name_list[0])]

    for n in name_list[1::]:
        df_table["change_energy_ratio_unbalanced_{:s}".format(n)] = (
            (df_table["energy_ratio_unbalanced_{:s}".format(n)] - bl) / bl
        )
        df_table["change_energy_ratio_balanced_{:s}".format(n)] = (
            (df_table["energy_ratio_balanced_{:s}".format(n)] - bl_bal)
            / bl_bal
        )

    # Add empty column/spacer
    df_table["___0"] = None

    # Add empty rows in df_table after each wd_bin
    df_empty = pd.DataFrame([None])
    df_array = []
    splits = np.where(total_ids)[0]
    splits = np.hstack([0, splits + 1])  # Add zero

    for ii in range(len(splits) - 1):
        lb = splits[ii]
        ub = splits[ii+1]
        df_array.append(df_table[lb:ub])
        df_array.append(df_empty)

    df_table_spaced = pd.concat(df_array, axis=0, ignore_index=True)
    df_table_spaced = df_table_spaced[df_table.columns]
    df_table = df_table_spaced

    # Write out the dataframe with xslxwriter
    writer = pd.ExcelWriter(fout_xlsx, engine="xlsxwriter")
    df_table.to_excel(
        writer,
        index=False,
        sheet_name="results",
        startcol=first_data_col,
        startrow=header_row,
    )
    workbook = writer.book
    worksheet = writer.sheets["results"]

    # FORMATTING

    # Format large numbers to 2 decimal
    fmt_rate = workbook.add_format({"num_format": "0.00", "bold": False})
    cols = df_table.columns
    change_list = [
        i
        for i in range(len(cols))
        if ("_mean_" in cols[i]) or ("_pow_" in cols[i]) or (
            ("_energy_" in cols[i]) and not ("energy_ratio_" in cols[i])
        )
    ]
    for c in change_list:
        worksheet.set_column(
            c + first_data_col, c + first_data_col, 10, fmt_rate
        )

    # Format energy ratios to 3 decimal
    fmt_rate = workbook.add_format({"num_format": "0.000", "bold": False})
    cols = df_table.columns
    change_list = [i for i in range(len(cols)) if "energy_ratio" in cols[i]]
    for c in change_list:
        worksheet.set_column(
            c + first_data_col, c + first_data_col, 10, fmt_rate
        )

    # Format change and TI into a percentage
    fmt_rate = workbook.add_format({"num_format": "%0.0", "bold": False})
    cols = df_table.columns
    change_list = [
        i
        for i in range(len(cols))
        if ("change" in cols[i]) or ("ti_" in cols[i])
    ]
    for c in change_list:
        worksheet.set_column(
            c + first_data_col, c + first_data_col, 10, fmt_rate
        )

    # # Make "totals" rows bold
    # bold_format = workbook.add_format({'bold': True})
    # total_ids = np.where(df_table["ws_bin"] == "TOTALS")[0]
    # for ri in total_ids:
    #     worksheet.set_row(first_data_row + ri, 15, bold_format)

    # Make the seperator columns very narrow and black
    fmt_black = workbook.add_format({"fg_color": "#000000"})
    change_list = [i for i in range(len(cols)) if "___" in cols[i]]
    for c in change_list:
        worksheet.set_column(
            c + first_data_col, c + first_data_col, 1, fmt_black
        )

    # Add data bars to the bins counts
    change_list = [i for i in range(len(cols)) if "bin" in cols[i]]
    for c in change_list:
        worksheet.conditional_format(
            first_data_row,
            c + first_data_col,
            df_table.shape[0] + first_data_row,
            c + first_data_col,
            {"type": "data_bar", "max_value": 100},
        )

    # Add color to the change columns
    change_list = [i for i in range(len(cols)) if "change" in cols[i]]

    for c in change_list:
        worksheet.conditional_format(
            first_data_row,
            c + first_data_col,
            df_table.shape[0] + first_data_row,
            c + first_data_col,
            {
                "type": "3_color_scale",
                "min_value": -1.0,
                "min_type": "num",
                "max_value": 1.0,
                "mid_value": 0.0,
                "mid_type": "num",
                "min_color": "#FF0000",
                "mid_color": "#FFFFFF",
                "max_color": "#00FF00",
                "max_type": "num",
            },
        )

    # Add color to energy ratios
    change_list = [
        i
        for i in range(len(cols))
        if ("er_" in cols[i]) and not ("change" in cols[i])
    ]
    for c in change_list:
        worksheet.conditional_format(
            first_data_row,
            c + first_data_col,
            df_table.shape[0] + first_data_row,
            c + first_data_col,
            {
                "type": "3_color_scale",
                "min_value": 0.25,
                "min_type": "num",
                "max_value": 2.0,
                "mid_value": 1.0,
                "mid_type": "num",
                "min_color": "#0000FF",
                "mid_color": "#FFFFFF",
                "max_color": "#00FF00",
                "max_type": "num",
            },
        )

    # Header
    # Adding formats for header row.
    fmt_header = workbook.add_format(
        {
            "bold": True,
            "text_wrap": True,
            "valign": "top",
            "fg_color": "#5DADE2",
            "font_color": "#FFFFFF",
            "border": 1,
        }
    )
    for col, value in enumerate(df_table.columns.values):
        worksheet.write(header_row, col + first_data_col, value, fmt_header)

    # If a floris model is provided, use it to make layout images
    if fi is not None:
        # Make that first colum wide
        worksheet.set_column("A:A", 30)

        # Create image folder
        img_folder = os.path.join(os.path.dirname(fout_xlsx), "xlsx_images")
        if not os.path.exists(img_folder):
            os.makedirs(img_folder, exist_ok=True)

        # For each bin were checking, make image of wake scenarios
        sort_df = df_table[["wd_bin", "ws_bin"]].copy()
        sort_df = sort_df.sort_values(["wd_bin", "ws_bin"]).dropna()
        for wdb in sort_df.wd_bin.unique():
            row_top_ws_bin = sort_df.index[wdb == sort_df["wd_bin"]][0]
            wd_arrow = wdb.mid  # Put arrow in middle of bin
            fig, ax = plt.subplots(figsize=(2, 2))
            fi.reinitialize_flow_field(
                wind_direction=wd_arrow, wind_speed=8.0
            )
            fi.calculate_wake()
            hor_plane = fi.get_hor_plane()
            hor_plane = wfct.cut_plane.change_resolution(
                hor_plane,
                resolution=(200, 200),
            )
            wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
            im_name = os.path.join(img_folder, "wd_%03d.png" % wd_arrow)
            fig.savefig(im_name, bbox_inches="tight")

            # Insert the figure
            worksheet.insert_image(first_data_row + row_top_ws_bin, 0, im_name)

        # Make the first row bigger
        worksheet.set_row(0, 120)

        # Get a list of blank columns indicating turbine starts
        blank_cols = [i for i in range(len(cols)) if "___" in cols[i]]

        # Plot the layout on top
        fig, ax = plt.subplots(figsize=(3, 2))
        fi.vis_layout(ax=ax)
        xt = np.array(fi.layout_x)
        yt = np.array(fi.layout_y)
        ax.plot(xt[test_turbines], yt[test_turbines], "mo", ms=25)
        im_name = os.path.join(img_folder, "layout.png")
        fig.savefig(im_name, bbox_inches="tight")
        worksheet.insert_image(0, blank_cols[0] + 1, im_name)

    # Hide columns if necessary
    if hide_bin_count_columns:
        cols = [i for i, c in enumerate(df_table.columns) if "bin_count" in c]
        for ii in cols:
            worksheet.set_column(ii + 1, ii + 1, None, None, {'hidden': 1})

    if hide_ws_ti_columns:
        cols = [
            i for i, c in enumerate(df_table.columns) if (
                ("ws_mean_" in c) or ("ws_std_" in c) or
                ("ti_mean_" in c) or ("ti_std_" in c)
            )
        ]
        for ii in cols:
            worksheet.set_column(ii + 1, ii + 1, None, None, {'hidden': 1})

    if hide_pow_columns:
        cols = [
            i for i, c in enumerate(df_table.columns) if (
                ("ref_pow_" in c) or ("test_pow_" in c)
            )
        ]
        for ii in cols:
            worksheet.set_column(ii + 1, ii + 1, None, None, {'hidden': 1})

    if hide_unbalanced_cols:
        cols = [i for i, c in enumerate(df_table.columns) if "unbalance" in c]
        for ii in cols:
            worksheet.set_column(ii + 1, ii + 1, None, None, {'hidden': 1})

    # Freeze the panes
    worksheet.freeze_panes(first_data_row, first_data_col)

    writer.save()
    print("File successfully written to {:s}.".format(fout_xlsx))
