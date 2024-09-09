"""FLASC DataFrame module."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame
from wind_up.constants import (
    DataColumns,
    RAW_DOWNTIME_S_COL,
    RAW_POWER_COL,
    RAW_WINDSPEED_COL,
    RAW_YAWDIR_COL,
    TIMESTAMP_COL,
)


# Create a new DataFrame subclass
class FlascDataFrame(DataFrame):
    """Subclass of pandas.DataFrame for working with FLASC data.

    I think it makes most sense to store it as FLASC expects it:
    - with the correct column names
    - in wide format

    Then, can offer a transformation to export as the user would like it, for them to work on it
    further. How, then, would I revert it back to the needed format


    Two possible types of data we should try to handle:
    1. Semiwide:
    - One column for time stamp
    - One column for turbine id
    - Many data channel columns
    2. Long:
    - One column for time stamp
    - One column for variable name
    - One column for value

    FLASC format is wide, i.e.
    - One column for time stamp
    - One column for each channel for each turbine

    Want handling to go between long and wide and semiwide and wide.
    """

    # Attributes to pickle must be in this list
    _metadata = ["name_map", "_user_format"]

    def __init__(self, *args, name_map=None, **kwargs):
        """Initialize the FlascDataFrame class, a subclass of pandas.DataFrame.

        Args:
            *args: arguments to pass to the DataFrame constructor
            name_map (dict): Dictionary of column names to map from the user format to the FLASC
                format, where the key string is the user format and the value string is the FLASC
                equivalent. Defaults to None.
            **kwargs: keyword arguments to pass to the DataFrame constructor
        """
        super().__init__(*args, **kwargs)

        self._user_format = "wide"  # or "long" or "semiwide"

        # check that name_map dictionary is valid
        if name_map is not None:
            if not isinstance(name_map, dict):
                raise ValueError("name_map must be a dictionary")
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in name_map.items()):
                raise ValueError("name_map must be a dictionary of strings")
        self.name_map = name_map
        # Apply the name_map
        self.convert_to_flasc_format(inplace=True)  # Do we want to do this here?

    @property
    def _constructor(self):
        return FlascDataFrame

    def __str__(self):
        """Printout when calling print(df)."""
        if self._in_flasc_format:
            return "FlascDataFrame in FLASC format\n" + super().__str__()
        else:
            return "FlascDataFrame in user format\n" + super().__str__()

    @property
    def n_turbines(self):
        """Return the number of turbines in the dataset."""
        self.check_flasc_format()

        nt = 0
        while ("pow_%03d" % nt) in self.columns:
            nt += 1
        return nt

    def check_flasc_format(self):
        """Raise an error if the data is not in FLASC format."""
        if not self._in_flasc_format:
            raise ValueError(
                (
                    "Data must be in FLASC format to perform this operation."
                    "Call df.convert_to_flasc_format() to convert the data to FLASC format."
                )
            )
        else:
            pass

    def convert_to_user_format(self, inplace=False):
        """Convert the DataFrame to the format that the user expects, given the name_map."""
        # Convert the format
        if self._user_format == "long":
            self._convert_wide_to_long()  # Should this be assigned to something?
        elif self._user_format == "semiwide":
            self._convert_wide_to_semiwide()  # Should this be assigned to something?
        elif self._user_format == "wide":
            pass

        # Set the flag
        self._in_flasc_format = False

        # Convert column names and return
        if self.name_map is not None:
            return self.rename(columns={v: k for k, v in self.name_map.items()}, inplace=inplace)
        else:
            return None if inplace else self.copy()

    def convert_to_flasc_format(self, inplace=False):
        """Convert the DataFrame to the format that FLASC expects."""
        # Convert the format
        if self._user_format == "long":
            self._convert_long_to_wide()  # Should this be assigned to something?
        elif self._user_format == "semiwide":
            self._convert_semiwide_to_wide()  # Should this be assigned to something?
        elif self._user_format == "wide":
            pass

        # Set the flag
        self._in_flasc_format = True

        # Convert column names and return
        if self.name_map is not None:
            return self.rename(columns=self.name_map, inplace=inplace)
        else:
            return None if inplace else self.copy()

    def _convert_long_to_wide(self):
        """Convert a long format DataFrame to a wide format DataFrame."""
        # raise NotImplementedError("TO DO")
        pass

    def _convert_semiwide_to_wide(self):
        """Convert a semiwide format DataFrame to a wide format DataFrame."""
        raise NotImplementedError("TO DO")

    def _convert_wide_to_long(self):
        """Convert a wide format DataFrame to a long format DataFrame."""
        if "time" not in self.columns:
            raise ValueError("Column 'time' must be present in the DataFrame")

        return self.melt(id_vars="time", var_name="variable", value_name="value")

    def _convert_wide_to_semiwide(self):
        """Convert a wide format DataFrame to a semiwide format DataFrame."""
        if "time" not in self.columns:
            raise ValueError("Column 'time' must be present in the DataFrame")

        raise NotImplementedError("TO DO")
        # Should have columns:
        # time
        # turbine_id (as specified by the user)
        # variable
        # value

    def to_feather(self, path, **kwargs):
        """Raise warning about lost information and save to feather format."""
        print(
            "Dataframe will be saved as a pandas DataFrame. "
            "Extra attributes from FlascDataFrame will be lost. "
            "We recommend using df.to_pickle() and pd.read_pickle() instead, "
            "as this will retain FlascDataFrame attributes."
        )
        return super().to_feather(path, **kwargs)

    def convert_to_windup_format(
        self,
        turbine_names: list[str] | None = None,
        time_col: str = "time",
        power_col: str = "pow",
        windspeed_col: str = "ws",
        winddirection_col: str = "wd",
        normal_operation_col: str | None = None,
        pitchangle_col: str | None = None,
        genrpm_col: str | None = None,
        downtimecounter_col: str | None = None,
        turbine_num_digits: int = 3,
    ):
        """Convert the DataFrame to the format that wind-up expects."""
        # figure out how many turbines there are from columns
        nt = sum(
            [
                1
                for col in self.columns
                if col.startswith(f"{power_col}_") and col[-turbine_num_digits:].isdigit()
            ]
        )
        # if turbine_names provided check it matches
        if turbine_names is not None:
            if not len(turbine_names) == nt:
                msg = (
                    f"Number of names in turbine_names, {len(turbine_names)}, "
                    f"does not match number of turbines in SCADA data, {nt}."
                )
                raise ValueError(msg)
        # build a new dataframe one turbine at a time
        turbine_num_format = f"0{turbine_num_digits}d"
        scada_df = pd.DataFrame()
        for i in range(nt):
            wtg_cols = [col for col in self.columns if col.endswith(f"_{i:{turbine_num_format}}")]
            wtg_df = pd.DataFrame(self[[time_col, *wtg_cols]]).__finalize__(None)
            wtg_df.columns = [time_col, *[x[: -(turbine_num_digits + 1)] for x in wtg_cols]]
            wtg_df[DataColumns.turbine_name] = (
                turbine_names[i] if turbine_names is not None else f"{i:{turbine_num_format}}"
            )
            scada_df = pd.concat([scada_df, wtg_df])
        scada_df = scada_df.set_index(time_col)
        scada_df.index.name = (
            TIMESTAMP_COL  # assumption is that flasc timestamps are UTC start format
        )
        scada_df = scada_df.rename(
            columns={
                power_col: RAW_POWER_COL,  # DataColumns.active_power_mean,
                windspeed_col: RAW_WINDSPEED_COL,  # DataColumns.wind_speed_mean,
                winddirection_col: RAW_YAWDIR_COL,  # DataColumns.yaw_angle_mean,
            }
        )

        if pitchangle_col is None:
            scada_df[DataColumns.pitch_angle_mean] = 0
        else:
            scada_df = scada_df.rename(columns={pitchangle_col: DataColumns.pitch_angle_mean})
        if genrpm_col is None:
            scada_df[DataColumns.gen_rpm_mean] = 1000
        else:
            scada_df = scada_df.rename(columns={genrpm_col: DataColumns.gen_rpm_mean})
        if downtimecounter_col is None:
            scada_df[RAW_DOWNTIME_S_COL] = 0
        else:
            scada_df = scada_df.rename(columns={downtimecounter_col: DataColumns.shutdown_duration})

        scada_df[DataColumns.active_power_mean] = scada_df[RAW_POWER_COL]
        scada_df[DataColumns.wind_speed_mean] = scada_df[RAW_WINDSPEED_COL]
        scada_df[DataColumns.yaw_angle_mean] = scada_df[RAW_YAWDIR_COL]
        scada_df[DataColumns.shutdown_duration] = scada_df[RAW_DOWNTIME_S_COL]
        if normal_operation_col is not None:
            cols_to_filter = [
                col
                for col in scada_df.columns
                if col != normal_operation_col
                and "raw_" not in col
                and col != DataColumns.turbine_name
            ]
            scada_df.loc[~scada_df[normal_operation_col].isin([True]), cols_to_filter] = pd.NA
        return scada_df
