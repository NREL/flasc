"""FLASC DataFrame module."""

from pandas import DataFrame


# Create a new DataFrame subclass
class FlascDataFrame(DataFrame):
    """Subclass of pandas.DataFrame for working with FLASC data.

    Stores data in preferred Flasc format, or user format, with option to convert between the two.

    Want handling to go between long and wide.
    """

    # Attributes to pickle must be in this list
    _metadata = [
        "channel_name_map",
        "_channel_name_map_to_user",
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

        # Check that the time column is present
        if "time" not in self.columns:
            raise ValueError("Column 'time' must be present in the DataFrame")

        # check that name_map dictionary is valid
        if channel_name_map is not None:
            if not isinstance(channel_name_map, dict):
                raise ValueError("channel_name_map must be a dictionary")
            if not all(
                isinstance(k, str) and isinstance(v, str) for k, v in channel_name_map.items()
            ):
                raise ValueError("channel_name_map must be a dictionary of strings")
        self.channel_name_map = channel_name_map

        # Save the reversed name_map (to go to user_format)
        self._channel_name_map_to_user = (
            {v: k for k, v in channel_name_map.items()} if channel_name_map is not None else None
        )

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

        # Convert the format
        if self._user_format == "long":
            df_user = self._convert_wide_to_long()
        elif self._user_format == "wide":
            df_user = self.copy()

            # In wide to wide conversion, only need to rename the columns
            if self.channel_name_map is not None:
                df_user.rename(columns=self._channel_name_map_to_user, inplace=True)

        # Assign to self or return
        if inplace:
            self.__init__(
                df_user,
                channel_name_map=self.channel_name_map,
                long_data_columns=self._long_data_columns,
            )
        else:
            return df_user

    def convert_to_flasc_format(self, inplace=False):
        """Convert the DataFrame to the format that FLASC expects.

        Args:
            inplace (bool): If True, modify the DataFrame in place. If False,
                return a new DataFrame.

        Returns:
            FlascDataFrame: FlascDataFrame in FLASC format if inplace is False, None otherwise

        """
        # Check if already in flasc format
        if self.in_flasc_format:
            if inplace:
                return
            else:
                return self.copy()

        # Convert the format
        if self._user_format == "long":
            df_flasc = self._convert_long_to_wide()  # Should this be assigned to something?
        elif self._user_format == "wide":
            df_flasc = self.copy()

            # In wide to wide conversion, only need to rename the columns
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

    def _convert_long_to_wide(self):
        """Convert a long format DataFrame to a wide format DataFrame.

        Returns:
            FlascDataFrame: Wide format FlascDataFrame
        """
        # Start by converting the variable names
        df_wide = self.copy()
        if df_wide.channel_name_map is not None:
            df_wide[self._long_data_columns["variable_column"]] = df_wide[
                self._long_data_columns["variable_column"]
            ].map(df_wide.channel_name_map)

        # Pivot the table so the variable column becomes the column names with time
        # kept as the first column and value as the values
        df_wide = df_wide.pivot(
            index="time",
            columns=self._long_data_columns["variable_column"],
            values=self._long_data_columns["value_column"],
        ).reset_index()

        # Remove the name
        df_wide.columns.name = None

        # Reset the index to make the time column a regular column
        return FlascDataFrame(
            df_wide,
            channel_name_map=self.channel_name_map,
            long_data_columns=self._long_data_columns,
        )

    def _convert_wide_to_long(self):
        """Convert a wide format DataFrame to a long format DataFrame.

        Returns:
            FlascDataFrame: Long format FlascDataFrame

        """
        df_long = self.melt(
            id_vars="time",
            var_name=self._long_data_columns["variable_column"],
            value_name=self._long_data_columns["value_column"],
        ).sort_values(["time", self._long_data_columns["variable_column"]])

        if self.channel_name_map is not None:
            df_long[self._long_data_columns["variable_column"]] = df_long[
                self._long_data_columns["variable_column"]
            ].map(self._channel_name_map_to_user)

        # Reset index for cleanliness
        df_long = df_long.reset_index(drop=True)

        return FlascDataFrame(
            df_long,
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
