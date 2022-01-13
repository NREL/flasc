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
from datetime import timedelta as td
import re

from ..dataframe_operations import dataframe_manipulations as dfm
from .. import utilities as fsut


def fix_csv_contents(csv_contents, line_format_str):
    """Check the contents of the raw database .csv file and ensure each row
       fits a predefined formatting. This can pick out irregularities in rows,
       such as a missing or deformed time entry in a row.

    Args:
        csv_contents ([str]): Contents of the preprocessed .csv file

    Returns:
        csv_contents ([str]): Contents of the postprocessed .csv file
    """

    # Specify required row format in each csv file
    line_format = re.compile(line_format_str)

    # Split large string into separate line strings
    csv_contents = csv_contents.split("\n")

    # Remove last line if empty
    if len(csv_contents[-1]) == 0:
        csv_contents = csv_contents[0:-1]  # Skip last line

    # Check and fix formatting of each line
    pop_ids = []  # List with to-be-removed row ids
    for i in range(1, len(csv_contents)):  # Skip first line
        csv_line = csv_contents[i]
        if line_format.match(csv_line) is None:
            print(
                "    Ignoring this row due to incorrect format: '"
                + csv_line
                + "'"
            )
            pop_ids.append(i)

    for i in pop_ids[::-1]:  # Back to front to avoid data shifts
        csv_contents.pop(i)  # Remove lines

    csv_contents = "\n".join(csv_contents) + "\n"
    return csv_contents


def read_raw_scada_files(files, single_file_reader_func,
                         channel_definitions_filename,
                         channel_definitions_sheetname,
                         ffill_missing_data=False,
                         missing_data_buffer=None):
    """ Read multiple SCADA datafiles and process them in preparation for
        uploading to the SQL database. Processing steps include merging
        multiple dataframes, removing/merging duplicate time entries, up-
        sampling data to the 1 second frequency, and finding large time
        gaps in the dataset and filling it with empty placeholder values.

    Args:
        files ([list]): List containing file paths for raw .csv files
        fn_channel_defs ([str]): Path to channel_definitions.xlsx

    Returns:
        df_db[pd.DataFrame]: Dataframe with the formatted database
    """
    # Read channel definitions file
    df_defs = pd.read_excel(io=channel_definitions_filename,
                            sheet_name=channel_definitions_sheetname)

    # Convert files to list if necessary
    if isinstance(files, str):
        files = [files]

    # Read all datafiles and merge them together
    df = single_file_reader_func(files[0], df_defs)
    for fn in files[1::]:
        df = df.append(single_file_reader_func(fn, df_defs))

    # Sort dataset by time and fix duplicate entries
    df = dfm.df_sort_and_fix_duplicates(df)

    if ffill_missing_data:
        dt = fsut.estimate_dt(df['time'])
        if missing_data_buffer is None:
            missing_data_buffer = dt + td(seconds=1)

        # Find large gaps of missing data and fill it with 'missing'
        df = dfm.df_find_and_fill_data_gaps_with_missing(df, missing_data_buffer)

        # Upsample dataset with forward fill (ZOH)
        print("Upsampling df with forward fill...")
        df = df.set_index('time')
        df = df.resample(dt).ffill().ffill()  # Forward fill()
        df = df.reset_index(drop=False)

        print("Replacing all 'missing' rows in the upsampled df with np.nan...")
        for c in df.columns:
            df[c] = df[c].replace(['missing'], np.nan)

    # Drop all rows that only have nan values
    df = dfm.df_drop_nan_rows(df)

    return df
