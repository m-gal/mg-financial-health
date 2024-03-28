"""
    LOAD data preprocessed1
    TRANSFORM w/ time's factor for model #2
    SAVE for futher modeling [./project/data/processed].

    to show progress bar for apply functions add to code
        import tqdm
        tqdm.notebook.tqdm().pandas()
    and change .apply(func) <-> .progress_apply(get_ddate_seq)
    @author: mikhail.galkin
"""

# %% Load libraries -------------------------------------------------------------
import sys
import winsound
import json
import numpy as np
import pandas as pd


# %% Load project"s stuff -------------------------------------------------------
# os.environ["NUMEXPR_MAX_THREADS"] = "48"
sys.path.extend(["../.."])

from fhealth.config import data_processed_dir
from fhealth.utils import pd_set_options
from fhealth.utils import pd_reset_options
from fhealth.data.explore_data import make_dataprep_report


def get_ddate_seq(x, quarters_lag_back=1):
    """Draw consistent sequances of statements w/ predefined back lag of quarters

    Args:
        x (pd.DataFrame): pd.DataFrame of pd.DataFrame.GroupBy object

    Returns:
        pd.DataFrame:
    """
    s = 1  # rank of statement's ddate sequances
    seq = []  # list of consistent sequances' numbers
    # for each row in df
    for i in range(len(x)):
        row = x.iloc[i]

        if row["quarters_lag_back"] <= quarters_lag_back:
            seq.append(s)
        else:
            s += 1
            seq.append(s)

    x["seq"] = seq
    return x


def df_transform_to_dflow(df, n_statements=3):
    # Define needless columns
    cols_inf = ["ddate", "seq", "seq_rank", "countryba", "sic"]
    cols_x = [x for x in list(df) if x not in (cols_inf + ["cik"])]

    # Initiate statements flows's dataframe
    dflow = df.copy()
    cols_00 = dict(zip(dflow[cols_x].columns, "yy_" + dflow[cols_x].columns))
    dflow.rename(columns=cols_00, inplace=True)

    #! Create new dataframe
    for i in range(-1, -n_statements, -1):
        print(f"Shift to {i} ddate")
        dshift = df.drop(cols_inf, axis=1).groupby(by=["cik"]).shift(-i).fillna(0)
        dshift.columns = "_" + str("%02d" % (-i - 1)) + "_" + dshift.columns
        dflow = pd.concat([dshift, dflow], axis=1)

    # # Drop rows: sequances w/ lenth less then 3
    idx_drop = dflow[dflow["seq_rank"] < n_statements].index
    print(f"Drop the {len(idx_drop)} of incompleted statements' sequances ...")
    dflow.drop(idx_drop, inplace=True)

    # Reorder the features
    dflow = dflow[["cik"] + cols_inf].join(dflow.drop(["cik"] + cols_inf, axis=1))
    dflow.rename(columns={"countryba": "country"}, inplace=True)
    # # sort columns alphabetically
    # dflow = dflow[sorted(list(dflow), reverse=True)]
    print(f"Transormation was done!\nThe new DF's shape: {dflow.shape}")
    return dflow


# %% Main -----------------------------------------------------------------------
def main(
    data_file="fhealth_dev_sic.csv",
    make_report=True,
):
    print(f"Loading the data processed for model #2 ... ----------------------")
    pd_set_options()
    # * Data
    dev_periods = [
        "2018-03-31",
        "2018-06-30",
        "2018-09-30",
        "2018-12-31",
        "2019-03-31",
        "2019-06-30",
        "2019-09-30",
        "2019-12-31",
        "2020-03-31",
        "2020-06-30",
        "2020-09-30",
        "2020-12-31",
        "2021-03-31",
    ]
    cols = [
        "sic",
        "cik",
        "countryba",
        "ddate",
        "solvency_debt_ratio",
        "liquid_current_ratio",
        "liquid_quick_ratio",
        "liquid_cash_ratio",
        "profit_net_margin",
        "profit_roa",
        # "active_dio",
        "active_acp",
    ]
    print(f"\nLoad data:\n\tDir: {data_processed_dir}\n\tFile: {data_file}")
    df = pd.read_csv(
        data_processed_dir / data_file,
        header=0,
        usecols=cols,
        parse_dates=["ddate"],
    )

    # A dictionary of the categorical features and their vocabulary for NN.
    CATEGORICAL_FEATURES_WITH_VOCABULARY = {
        "sic": [int(x) for x in sorted(list(df["sic"].unique()))],
        "country": sorted(list(df["countryba"].unique())),
    }
    # Save dictionary of categorical features
    with open(data_processed_dir / "model2_cat_feats_vocab.json", "w") as fp:
        json.dump(CATEGORICAL_FEATURES_WITH_VOCABULARY, fp)

    # Trim time period
    df = df.loc[df["ddate"].isin(dev_periods)].reset_index(drop=True)
    print(f"The DF's shape: {df.shape}")
    df.sort_values(["cik", "ddate"], ignore_index=True, inplace=True)

    print(f"\nFind out the consistent sequances in statements' ddate ---------")
    # # Calculate time lag in days between statements inside each 'cik'
    # df["ddate_lag_back"] = (
    #     df.groupby(["cik"])["ddate"]
    #     .diff(1)  #
    #     .apply(lambda x: x / np.timedelta64(1, "D"))
    #     .fillna(0)
    #     .astype("int64")
    # )
    # Calculate time lag in quaters between statements inside each 'cik'
    df["quarters_lag_back"] = (
        df.groupby(["cik"])["ddate"]
        .diff(1)
        .apply(lambda x: (x / np.timedelta64(1, "D")) / 90)
        .fillna(0)
        .astype("int64")
    )
    df = df.groupby(["cik"]).apply(get_ddate_seq)
    # Get rank of statement inside the sequances
    df["seq_rank"] = df.groupby(["cik", "seq"])["seq"].rank("first").astype("int64")
    # Get lenth of sequances
    df["seq_len"] = df.groupby(["cik", "seq"])["seq"].transform(len)
    # Drop rows: sequances w/ lenth less then 3
    idx_drop = df[df["seq_len"] < 3].index
    print(f"Drop the {len(idx_drop)} of sequances w/ lenth less then 3...")
    df.drop(idx_drop, inplace=True)
    # Drop columns
    df.drop(["quarters_lag_back", "seq_len"], axis=1, inplace=True)

    print(f"\nTransform the original 2D dataset to 'statements flows' format -")
    df = df_transform_to_dflow(df, n_statements=3)

    print(f"\nCalculate changing at last statement ---------------------------")
    df["y_solvency_debt_ratio"] = df.apply(
        lambda x: 1 if x["yy_solvency_debt_ratio"] < x["_00_solvency_debt_ratio"] else 0,
        axis=1,
    )
    df["y_liquid_current_ratio"] = df.apply(
        lambda x: 1 if x["yy_liquid_current_ratio"] >= x["_00_liquid_current_ratio"] else 0,
        axis=1,
    )
    df["y_liquid_quick_ratio"] = df.apply(
        lambda x: 1 if x["yy_liquid_quick_ratio"] >= x["_00_liquid_quick_ratio"] else 0,
        axis=1,
    )
    df["y_liquid_cash_ratio"] = df.apply(
        lambda x: 1 if x["yy_liquid_cash_ratio"] >= x["_00_liquid_cash_ratio"] else 0,
        axis=1,
    )
    df["y_profit_net_margin"] = df.apply(
        lambda x: 1 if x["yy_profit_net_margin"] >= x["_00_profit_net_margin"] else 0,
        axis=1,
    )
    df["y_profit_roa"] = df.apply(
        lambda x: 1 if x["yy_profit_roa"] >= x["_00_profit_roa"] else 0,
        axis=1,
    )
    df["y_active_acp"] = df.apply(
        lambda x: 1 if x["yy_active_acp"] < x["_00_active_acp"] else 0,
        axis=1,
    )

    if make_report:
        print(f"Type#2: Generating a report ... ------------------------------")
        # Create DataPrep reports
        report_name = f"model2_dev"
        report_title = f"DATA: Processed - Type#2: Forecasting"
        make_dataprep_report(df, report_name, report_title)

    pd_reset_options()
    winsound.Beep(frequency=3000, duration=300)

    return df


# %% RUN ========================================================================
if __name__ == "__main__":
    df = main()
    df.to_csv(data_processed_dir / "model2_dev.csv", header=True, index=False)
    print(f"\n!!! DONE: Dev set for model #2 !!!")

# %%
