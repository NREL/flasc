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

    Stores data in preferred Flasc format, or user format, with option to convert between the two.

    Want handling to go between long and wide.
    """

    # Attributes to pickle must be in this list
    _metadata = [
        "channel_name_map",
        "_user_format",
        "_long_data_columns",
    ]

    def __init__(self, *args, channel_name_map=None, long_data_columns=None, **kwargs):
        """Initialize the FlascDataFrame class, a subclass of pandas.DataFrame.

        Args:
            *args: arguments to pass to the DataFrame constructor
            channel_name_map (dict): Dictionary of column names to map from the user format to the
                FLASC format, where the key string is the user format and the value string is the
                FLASC equivalent. Defaults to None.
            long_data_columns (dict): Dictionary of column names for long format data. Defaults to
                {"variable_column": "variable", "value_column": "value"}.  If
                not provided, user data format assumed to be wide.
            **kwargs: keyword arguments to pass to the DataFrame constructor
        """
        super().__init__(*args, **kwargs)

        # check that name_map dictionary is valid
        if channel_name_map is not None:
            if not isinstance(channel_name_map, dict):
                raise ValueError("channel_name_map must be a dictionary")
            if not all(
                isinstance(k, str) and isinstance(v, str) for k, v in channel_name_map.items()
            ):
                raise ValueError("channel_name_map must be a dictionary of strings")
        self.channel_name_map = channel_name_map

        # Determine the user format
        if long_data_columns is None:
            self._user_format = "wide"
            self._long_data_columns = None
        else:
            self._user_format = "long"

            # Confirm the long_data_columns is a dictionary with the correct keys
            if not isinstance(long_data_columns, dict):
                raise ValueError("long_data_columns must be a dictionary")
            if not all(col in long_data_columns for col in ["variable_column", "value_column"]):
                raise ValueError(
                    "long_data_columns must contain keys 'variable_column', " "and 'value_column'"
                )
            self._long_data_columns = long_data_columns

    @property
    def in_flasc_format(self):
        """Return True if the data is in FLASC format, False otherwise."""
        if ("time" in self.columns) and ("pow_000" in self.columns):
            return True
        else:
            return False

    @property
    def _constructor(self):
        return FlascDataFrame

    def __str__(self):
        """Printout when calling print(df)."""
        if self.in_flasc_format:
            return "FlascDataFrame in FLASC format\n" + super().__str__()
        else:
            return f"FlascDataFrame in user ({self._user_format}) format\n" + super().__str__()

    def _repr_html_(self):
        """Printout when displaying results in jupyter notebook."""
        if self.in_flasc_format:
            return "FlascDataFrame in FLASC format\n" + super()._repr_html_()
        else:
            return f"FlascDataFrame in user ({self._user_format}) format\n" + super()._repr_html_()

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
        if not self.in_flasc_format:
            raise ValueError(
                (
                    "Data must be in FLASC format to perform this operation."
                    "Call df.convert_to_flasc_format() to convert the data to FLASC format."
                )
            )
        else:
            pass

    def copy_metadata(self, other):
        """Copy metadata from another FlascDataFrame to self.

        Args:
            other (FlascDataFrame): DataFrame to copy metadata from.
        """
        for attr in self._metadata:
            setattr(self, attr, getattr(other, attr))

    def convert_to_user_format(self, inplace=False):
        """Convert the DataFrame to the format that the user expects, given the channel_name_map.

        Args:
            inplace (bool): If True, modify the DataFrame in place.
                If False, return a new DataFrame.

        Returns:
            FlascDataFrame: FlascDataFrame in user format if inplace is False, None otherwise.

        """
        # Check if already in user format
        if not self.in_flasc_format:
            if inplace:
                return
            else:
                return self.copy()

        # Make a copy of self
        df_user = self.copy()

        # Rename the channel columns to user-specified names
        if self.channel_name_map is not None:
            df_user.rename(columns={v: k for k, v in self.channel_name_map.items()}, inplace=True)

        # Convert the format to long if _user_format is long
        if self._user_format == "long":
            df_user = self._convert_wide_to_long(df_user)

        # Assign to self or return
        if inplace:
            self.__init__(
                df_user,
                channel_name_map=self.channel_name_map,
                long_data_columns=self._long_data_columns,
            )
        else:
            return df_user

    def convert_time_to_datetime(self, inplace=False):
        """Convert the time column to a datetime representation.

        Args:
            inplace (bool): If True, modify the DataFrame in place. If False,
                return a new DataFrame.

        Returns:
            FlascDataFrame: FlascDataFrame with time column as datetime object if inplace is False,
            None otherwise
        """
        if "time" not in self.columns:
            raise KeyError("Column 'time' must be present in the DataFrame")

        if inplace:
            self["time"] = pd.to_datetime(self["time"])
        else:
            df = self.copy()
            df["time"] = pd.to_datetime(df["time"])
            return df

    def convert_to_flasc_format(self, inplace=False):
        """Convert the DataFrame to the format that FLASC expects.

        Args:
            inplace (bool): If True, modify the DataFrame in place. If False,
                return a new DataFrame.

        Returns:
            FlascDataFrame: FlascDataFrame in FLASC format if inplace is False, None otherwise

        # TODO: could consider converting "time" to datetime type here. If so, will want to keep
        # the original "time" column for back-conversion if needed.
        # Similarly, we could sort on time, but perhaps both are too meddlesome
        """
        # Check if already in flasc format
        if self.in_flasc_format:
            if inplace:
                return
            else:
                return self.copy()

        # Make a copy of self
        df_flasc = self.copy()

        # Convert back from long if necessary
        if self._user_format == "long":
            df_flasc = self._convert_long_to_wide(df_flasc)

        # Rename the channel columns to flasc-naming convention
        if self.channel_name_map is not None:
            df_flasc.rename(columns=self.channel_name_map, inplace=True)

        # Assign to self or return
        if inplace:
            self.__init__(
                df_flasc,
                channel_name_map=self.channel_name_map,
                long_data_columns=self._long_data_columns,
            )
        else:
            return df_flasc

    def _convert_long_to_wide(self, df_):
        """Convert a long format DataFrame to a wide format DataFrame.

        Args:
            df_ (FlascDataFrame): Long format FlascDataFrame

        Returns:
            FlascDataFrame: Wide format FlascDataFrame
        """
        # Pivot the table so the variable column becomes the column names with time
        # kept as the first column and value as the values
        df_ = df_.pivot(
            index="time",
            columns=self._long_data_columns["variable_column"],
            values=self._long_data_columns["value_column"],
        ).reset_index()

        # Remove the name
        df_.columns.name = None

        # Reset the index to make the time column a regular column
        return FlascDataFrame(
            df_,
            channel_name_map=self.channel_name_map,
            long_data_columns=self._long_data_columns,
        )

    def _convert_wide_to_long(self, df_):
        """Convert a wide format DataFrame to a long format DataFrame.

        Args:
            df_ (FlascDataFrame): Wide format FlascDataFrame

        Returns:
            FlascDataFrame: Long format FlascDataFrame

        """
        df_ = df_.melt(
            id_vars="time",
            var_name=self._long_data_columns["variable_column"],
            value_name=self._long_data_columns["value_column"],
        ).sort_values(["time", self._long_data_columns["variable_column"]])

        # Reset index for cleanliness
        df_ = df_.reset_index(drop=True)

        return FlascDataFrame(
            df_,
            channel_name_map=self.channel_name_map,
            long_data_columns=self._long_data_columns,
        )

    def to_feather(self, path, **kwargs):
        """Raise warning about lost information and save to feather format."""
        print(
            "Dataframe will be saved as a pandas DataFrame. "
            "Extra attributes from FlascDataFrame will be lost. "
            "We recommend using df.to_pickle() and pd.read_pickle() instead, "
            "as this will retain FlascDataFrame attributes."
        )
        return super().to_feather(path, **kwargs)

    def export_to_windup_format(
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
