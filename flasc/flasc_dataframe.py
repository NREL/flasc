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
    """

    # Add to list, an initialize with Nones or similar
    _metadata = ["new_property", "name_map", "newnew_property"]

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

        self._flasc = True
        # add an attribute here, make sure it's in the metadata
        self.new_property = 23

        self._user_format = "wide"  # or "long" or "semiwide"

        # check that name_map dictionary is valid
        if name_map is not None:
            if not isinstance(name_map, dict):
                raise ValueError("name_map must be a dictionary")
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in name_map.items()):
                raise ValueError("name_map must be a dictionary of strings")
        self.name_map = name_map
        # Apply the name_map
        self.convert_to_flasc_format(inplace=True)

    @property
    def _constructor(self):
        return FlascDataFrame

    def __str__(self):
        """Printout when calling print(df)."""
        return "This is a FlascDataFrame!\n" + super().__str__()

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
            raise ValueError((
                "Data must be in FLASC format to perform this operation."
                "Call df.convert_to_flasc_format() to convert the data to FLASC format."
            ))
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
        pass

    def _convert_semiwide_to_wide(self):
        """Convert a semiwide format DataFrame to a wide format DataFrame."""
        pass

    def _convert_wide_to_long(self):
        """Convert a wide format DataFrame to a long format DataFrame."""
        if "time" not in self.columns:
            raise ValueError("Column 'time' must be present in the DataFrame")

        return self.melt(id_vars="time", var_name="variable", value_name="value")

    def _convert_wide_to_semiwide(self):
        """Convert a wide format DataFrame to a semiwide format DataFrame."""
        if "time" not in self.columns:
            raise ValueError("Column 'time' must be present in the DataFrame")

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


# Likely this will be used for testing, later but it's convenient for prototyping here
if __name__ == "__main__":
    df = FlascDataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, name_map={"a": "AA"})
    print(df)
    print(df.new_property)
    df.flasc_method()  # Assigns newnew_property
    print(df._flasc)
    print(df._metadata)

    # Check that modifying df still returns an FlascDataFrame
    print(type(df))
    df.new_property = 42
    df.name_map = 6
    df = df.drop(columns="c")  # Modifies the dataframe, returns a copy
    print(df)
    print(df.new_property)
    print(df.name_map)
    # Not retained with copy, unless in _metadata. If in _metadata, retained!
    print(df.newnew_property)

    # Try out the convert methods (seem good)
    data = {"AA": [1, 2, 3], "BB": [4, 5, 6], "CC": [7, 8, 9]}
    df = FlascDataFrame(data, name_map={"AA": "a", "BB": "b", "CC": "c"})
    print(df)
    df2 = df.convert_to_user_format()
    print(df2)
    df.convert_to_user_format(inplace=True)
    print(df)

    # Drop a column, convert back
    df = df.drop(columns="CC")
    df.convert_to_flasc_format(inplace=True)
    print(df)
    # Works great!

    # Next, the long format conversion... more complicated

    """
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

    Converting between semiwide and wide should be relatively straightforward.
    Actually, neither of these should be too bad
    """
