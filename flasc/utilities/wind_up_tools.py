
import pandas as pd
from wind_up.constants import DataColumns, TIMESTAMP_COL


def df_scada_to_wind_up(
    df_scada: pd.DataFrame, turbine_names: list[str] | None = None
) -> pd.DataFrame:
    # figure out how many turbines there are from columns
    nt = sum(
        [
            1
            for col in df_scada.columns
            if col.startswith("is_operation_normal_") and col[-3:].isdigit()
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
    scada_df = pd.DataFrame()
    for i in range(nt):
        wtg_cols = [col for col in df_scada.columns if col.endswith(f"_{i:03d}")]
        wtg_df = df_scada[["time", *wtg_cols]].copy()
        wtg_df.columns = ["time", *[x[:-4] for x in wtg_cols]]
        wtg_df["TurbineName"] = turbine_names[i] if turbine_names is not None else f"{i:03d}"
        scada_df = pd.concat([scada_df, wtg_df])
    scada_df = scada_df.set_index("time")
    scada_df.index.name = TIMESTAMP_COL  # assumption is that flasc timestamps are UTC start format
    scada_df = scada_df.rename(
        columns={
            "pow": DataColumns.active_power_mean,
            "ws": DataColumns.wind_speed_mean,
            "wd": DataColumns.yaw_angle_mean,
        }
    )
    # fill in other columns with placeholding values
    scada_df[DataColumns.pitch_angle_mean] = 0
    scada_df[DataColumns.gen_rpm_mean] = 1000
    scada_df[DataColumns.shutdown_duration] = 0
    return scada_df
