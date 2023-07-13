class EnergyRatioResult:
    """  This class is used to store the results of the energy ratio calculations
    and provide convenient methods for plotting and saving the results.
    """
    def __init__(self,
                    df_result, 
                    df_names,
                    energy_table,
                    ref_cols,
                    test_cols,
                    wd_cols,
                    ws_cols,
                    wd_step,
                    wd_min,
                    wd_max,
                    ws_step,
                    ws_min,
                    ws_max,
                    bin_cols_in,
                    N
                  ):

        self.df_result = df_result
        self.df_names = df_names
        self.energy_table = energy_table
        self.num_df = len(df_names)
        self.ref_cols = ref_cols
        self.test_cols = test_cols
        self.wd_cols = wd_cols
        self.ws_cols = ws_cols
        self.wd_step = wd_step
        self.wd_min = wd_min
        self.wd_max = wd_max
        self.ws_step = ws_step
        self.ws_min = ws_min
        self.ws_max = ws_max
        self.bin_cols_in = bin_cols_in
        self.N = N