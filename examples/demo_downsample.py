import os
import pandas as pd
import numpy as np

from flasc.dataframe_operations.dataframe_manipulations import df_reduce_precision


def load_data():
    # Load dataframe with scada data
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ftr_path = os.path.join(
        root_dir,  "demo_dataset", "demo_dataset_scada_60s.ftr"
    )
    if not os.path.exists(ftr_path):
        raise FileNotFoundError(
            "Please run ./examples/demo_dataset/"
            + "generate_demo_dataset.py before try"
            + "ing any of the other examples."
        )
    df = pd.read_feather(ftr_path)
    return df




if __name__ == "__main__":
    # Set a random seed
    np.random.seed(0)

    # Load data and FLORIS
    df = load_data()

    # Downsample the types
    df_out = df_reduce_precision(df, verbose=True)

    print()
    print('---------------')
    print('Memory Usage of original: ', df.memory_usage(deep=True).sum(), 'bytes')
    print('Memory Usage after reduction: ', df_out.memory_usage(deep=True).sum(), 'bytes', '(%.1f%%)' % (100 * df_out.memory_usage(deep=True).sum()/df.memory_usage(deep=True).sum()))
    