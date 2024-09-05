"""FLASC DataFrame module."""

from pandas import DataFrame


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

    def __init__(self, *args, name_map=None, in_flasc_format=True, user_format="wide", **kwargs):
        """Initialize the FlascDataFrame class, a subclass of pandas.DataFrame.

        Args:
            *args: arguments to pass to the DataFrame constructor
            name_map (dict): Dictionary of column names to map from the user format to the FLASC
                format, where the key string is the user format and the value string is the FLASC
                equivalent. Defaults to None.
            in_flasc_format (bool): Whether the data is in FLASC format. Defaults to True.
            user_format (str): The format that the user expects the data to be in. Must be one of
                'long', 'semiwide', or 'wide'. Defaults to 'wide'.
            **kwargs: keyword arguments to pass to the DataFrame constructor
        """
        super().__init__(*args, **kwargs)

        # check that name_map dictionary is valid
        if name_map is not None:
            if not isinstance(name_map, dict):
                raise ValueError("name_map must be a dictionary")
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in name_map.items()):
                raise ValueError("name_map must be a dictionary of strings")
        self.name_map = name_map

        # Save the reversed name_map (to go to user_format)
        self._name_map_to_user = (
            {v: k for k, v in name_map.items()} if name_map is not None else None
        )

        # Set the format
        self._in_flasc_format = in_flasc_format

        # Save the user format
        if user_format not in ["long", "semiwide", "wide"]:
            raise ValueError("user_format must be one of 'long', 'semiwide', 'wide'")
        self._user_format = user_format

        # I think we should not convert to allow to stay in user format
        # # Convert to flasc format if not already
        # if not in_flasc_format:
        #     self.convert_to_flasc_format(inplace=True)
        # else:
        #     self._in_flasc_format = True

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
            df_user = self._convert_wide_to_long()
        elif self._user_format == "semiwide":
            df_user = self._convert_wide_to_semiwide()
        elif self._user_format == "wide":
            df_user = self.copy()

            # In wide to wide conversion, only need to rename the columns
            if self.name_map is not None:
                df_user.rename(self._name_map_to_user, inplace=inplace)

        # Assign to self or return
        if inplace:
            self.__init__(
                df_user,
                name_map=self.name_map,
                in_flasc_format=False,
                user_format=self._user_format,
            )
        else:
            # Force in flasc format to False
            df_user._in_flasc_format = False

            return df_user

    def convert_to_flasc_format(self, inplace=False):
        """Convert the DataFrame to the format that FLASC expects."""
        # Convert the format
        if self._user_format == "long":
            df_flasc = self._convert_long_to_wide()  # Should this be assigned to something?
        elif self._user_format == "semiwide":
            df_flasc = self._convert_semiwide_to_wide()  # Should this be assigned to something?
        elif self._user_format == "wide":
            df_flasc = self.copy()

            # In wide to wide conversion, only need to rename the columns
            if self.name_map is not None:
                df_flasc.rename(columns=self.name_map, inplace=inplace)

        # Assign to self or return
        if inplace:
            self.__init__(
                df_flasc,
                name_map=self.name_map,
                in_flasc_format=True,
                user_format=self._user_format,
            )
        else:
            # Force in flasc format to True
            df_flasc._in_flasc_format = True

            return df_flasc

    def _convert_long_to_wide(self):
        """Convert a long format DataFrame to a wide format DataFrame."""
        # Start by converting the variable names
        df_wide = self.copy()
        if df_wide.name_map is not None:
            df_wide["variable"] = df_wide["variable"].map(df_wide.name_map)

        # Pivot the table so the variable column becomes the column names with time
        # kept as the first column and value as the values
        df_wide = df_wide.pivot(index="time", columns="variable", values="value").reset_index()

        # Remove the name
        df_wide.columns.name = None

        # Reset the index to make the time column a regular column
        return FlascDataFrame(
            df_wide,
            name_map=self.name_map,
            in_flasc_format=self._in_flasc_format,
            user_format=self._user_format,
        )

    def _convert_semiwide_to_wide(self):
        """Convert a semiwide format DataFrame to a wide format DataFrame."""
        raise NotImplementedError("TO DO")

    def _convert_wide_to_long(self):
        """Convert a wide format DataFrame to a long format DataFrame.

        Returns:
            FlascDataFrame: Long format FlascDataFrame

        """
        if "time" not in self.columns:
            raise ValueError("Column 'time' must be present in the DataFrame")

        df_long = self.melt(id_vars="time", var_name="variable", value_name="value").sort_values(
            ["time", "variable"]
        )

        if self.name_map is not None:
            df_long["variable"] = df_long["variable"].map(self._name_map_to_user)

        # Reset index for cleanliness
        df_long = df_long.reset_index(drop=True)

        return FlascDataFrame(
            df_long,
            name_map=self.name_map,
            in_flasc_format=self._in_flasc_format,
            user_format=self._user_format,
        )

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
