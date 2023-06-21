# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import os

import pandas as pd


if __name__ == "__main__":
    # Placeholder function that normally downloads data from a repository,
    # potentially by usage of the SQL database manager.

    # All raw SCADA, met mast, lidar, buoy data is collected and then
    # formatted into a .ftr (feather) file for sake of disk space usage.

    # For our sake, we just copy the data from the "demo_dataset" folder
    # over to a local folder.
    root_path = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(root_path, "..", "demo_dataset")
    df_scada = pd.read_feather(
        os.path.join(source_path, "demo_dataset_scada_60s.ftr")
    )
    df_metmast = pd.read_feather(
        os.path.join(source_path, "demo_dataset_metmast_60s.ftr")
    )

    # Now save to local directory
    data_path = os.path.join(root_path, "data", "00_raw_data")
    os.makedirs(data_path, exist_ok=True)

    df_scada.to_feather(os.path.join(data_path, "df_scada_60s.ftr"))
    df_metmast.to_feather(os.path.join(data_path, "df_metmast_60s.ftr"))

    print("All data is downloaded and saved to {:s}.".format(data_path))
