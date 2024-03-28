"""
    LOAD precleaned data
    MAKE some processing
    SAVE data for futher modeling [./project/data/processed].

    @author: mikhail.galkin
"""

# %% Load libraries -------------------------------------------------------------
import sys
import warnings
import time

import numpy as np
import pandas as pd

from pathlib import Path
from IPython.display import display


# %% Load project"s stuff -------------------------------------------------------
# os.environ["NUMEXPR_MAX_THREADS"] = "48"
sys.path.extend(["../.."])

from fhealth.config import data_precleaned_dir
from fhealth.config import data_processed_dir
from fhealth.utils import pd_set_options
from fhealth.utils import pd_reset_options
from fhealth.utils import df_get_glimpse
from fhealth.utils import cols_get_na
from fhealth.utils import outliers_get_quantiles
from fhealth.utils import outliers_get_zscores
from fhealth.utils import outliers_ridoff
from fhealth.data.explore_data import make_dataprep_report
from fhealth.data.explore_data import make_sweetviz_analyze


# %% Main -----------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")
    pd_set_options()
    # reports' points for dev set
    dev_periods = [
        "2012-03-31",
        "2012-06-30",
        "2012-09-30",
        "2012-12-31",
        "2013-03-31",
        "2013-06-30",
        "2013-09-30",
        "2013-12-31",
        "2014-03-31",
        "2014-06-30",
        "2014-09-30",
        "2014-12-31",
        "2015-03-31",
        "2015-06-30",
        "2015-09-30",
        "2015-12-31",
        "2016-03-31",
        "2016-06-30",
        "2016-09-30",
        "2016-12-31",
        "2017-03-31",
        "2017-06-30",
        "2017-09-30",
        "2017-12-31",
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
    cols_base = [
        "sic",
        "cik",
        "countryba",
        "name",
        "ddate",
    ]
    cols_coeff = [
        "solvency_debt_ratio",
        # "solvency_debt_to_equity",
        # "solvency_irc",
        "liquid_current_ratio",
        "liquid_quick_ratio",
        "liquid_cash_ratio",  # 12962 -> 12948
        # "profit_gross_margin",
        "profit_net_margin",
        # "profit_operating_margin",
        # "profit_roe",
        "profit_roa",
        # "active_dio", # 12948 -> 7329
        "active_acp",
    ]
    # ratios which are then higher than better
    cols_coeff_good_above = [
        "liquid_current_ratio",
        "liquid_quick_ratio",
        "liquid_cash_ratio",
        "profit_net_margin",
        "profit_roa",
    ]
    # ratios which are then lower than better
    cols_coeff_good_under = [
        "solvency_debt_ratio",
        "active_acp",
    ]
    # groups for median calculation
    groups = [
        # "ddate",
        "sic",
        # "countryba",
    ]

    print(f"Loading the precleaned data ... ----------------------------------")
    df = pd.read_csv(
        data_precleaned_dir / "2012q1_2021q2_precleaned.csv",
        header=0,
        usecols=cols_base + cols_coeff,
        parse_dates=["ddate"],
    )  # 333107

    print(f"Handling data: periods, Inf, NA's & nonsensical ... --------------")
    df = df.loc[df["ddate"].isin(dev_periods)]  # 236748
    df["countryba"] = df["countryba"].fillna("US")
    df = df.replace([np.inf, -np.inf], np.nan)  # 236748
    df = df.dropna(axis=0, how="any", subset=cols_coeff)  # 65163
    # drop nonsensical
    df = df.loc[
        (
            df[
                [
                    "solvency_debt_ratio",
                    "liquid_current_ratio",
                    "liquid_quick_ratio",
                    "liquid_cash_ratio",
                    # "active_dio",
                    "active_acp",
                ]
            ]
            >= 0
        ).all(axis=1)
    ]  # 64485
    df = df.loc[(df[["profit_net_margin"]] < 2).all(axis=1)]  # 63506
    df = df.loc[(df[["profit_net_margin"]] > -2).all(axis=1)]  # 52923

    print(f"Handling outliers ... --------------------------------------------")
    # Drop incorrect values
    idx_drop = df[df["profit_roa"] < -10].index
    df.drop(idx_drop, inplace=True)

    # One-side outliers by positive quantile
    df_q = outliers_get_quantiles(
        df,
        cols_to_check=[
            "solvency_debt_ratio",
            "liquid_current_ratio",
            "liquid_quick_ratio",
            "liquid_cash_ratio",
            # "active_dio",
            "active_acp",
        ],
        treshold=0.99,
    )
    df = outliers_ridoff(df, df_q)

    # Two-sides outliers by z-score
    # df_z = outliers_get_zscores(
    #     df,
    #     cols_to_check=["profit_net_margin", "profit_roa"],
    #     sigma=2.5,
    # )
    # df = outliers_ridoff(df, df_z)

    # Sort df
    df = df.sort_values(["cik", "ddate"], ignore_index=True)

    print(f"Type#1: Labeling the data ... ------------------------------------")
    # Calc group's medians for coeffitients
    cols_med = ["med_" + col for col in cols_coeff]
    df[cols_med] = df.groupby(by=groups)[cols_coeff].transform("median")

    # Get columns name for labels' calculating
    cols_y_above = ["y_" + col for col in cols_coeff_good_above]
    cols_y_under = ["y_" + col for col in cols_coeff_good_under]
    cols_med_above = ["med_" + col for col in cols_coeff_good_above]
    cols_med_under = ["med_" + col for col in cols_coeff_good_under]

    # Calc labels for coeffitients
    df[cols_y_above] = pd.DataFrame(
        np.where(df[cols_coeff_good_above].values >= df[cols_med_above].values, 1, 0)
    )
    df[cols_y_under] = pd.DataFrame(
        np.where(df[cols_coeff_good_under].values < df[cols_med_under].values, 1, 0)
    )

    # Calc resulting label
    df["y_ttl"] = df[cols_y_above + cols_y_under].sum(axis=1)
    df["y_ttl_avg"] = df.groupby(by=groups)["y_ttl"].transform("mean")
    df["y"] = pd.DataFrame(np.where(df["y_ttl"].values >= df["y_ttl_avg"].values, 1, 0))

    df_get_glimpse(df)
    # Get vacabularies for categorical features to use in NN later
    {}

    print(f"Type#1: Generating a report ... ----------------------------------")
    # Create DataPrep reports
    report_name = f"fhealth_dev_{groups[0]}"
    report_title = f"DATA: Processed - Type#1: Resulting score"
    make_dataprep_report(df, report_name, report_title)
    make_sweetviz_analyze(df, report_name, target_feat="y")

    print(f"Save the cleaned data ... ----------------------------------------")
    file_to_save = f"{report_name}.csv"
    df.to_csv(data_processed_dir / file_to_save, header=True, index=False)

    pd_reset_options()
    print("\n!!! DONE !!!")


# %% Run ------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# %%
