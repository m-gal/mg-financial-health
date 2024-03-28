"""
    LOAD raw data
    MAKE some data cleaning and transformations
    SAVE data for futher processing [./project/data/precleaned].

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

from fhealth.config import data_raw_dir
from fhealth.config import data_precleaned_dir
from fhealth.utils import pd_set_options
from fhealth.utils import pd_reset_options
from fhealth.utils import df_get_glimpse

from fhealth.data.acquire_data import get_zip_names
from fhealth.data.explore_data import load_zip
from fhealth.data.explore_data import make_dataprep_report


# %% Custom funcs ---------------------------------------------------------------
def get_tag_from_num(zip_loaded):
    """Get variable from dataset and save it to xls"""
    zip_name = next(iter(zip_loaded))
    df = zip_loaded[zip_name]["num"]
    pd.set_option("display.float_format", lambda x: "%.5f" % x)
    xls_to_save = Path("../../data/analyze/2021q1_num_tag.xlsx")
    agg_func = {
        "tag": ["count"],
        "value": ["mean", "median", "min", "max", "std"],
    }
    pd.DataFrame(df.groupby("tag").agg(agg_func)).to_excel(xls_to_save)
    print(f"xlsx file saved...")


def df_get_from_zip(zip_loaded, cols, tags):
    """Returns a one precleaned DF from one loaded zip

    Args:
        zip_loaded (dict): oreturn of custom 'load_zip' func
    """
    # # Temporary set of cik for debbuging
    # stmt = ["BS", "IS", "CF", "EQ", "CI"]
    # ciks = [
    #     931584,
    #     1001039,
    #     1399529,
    #     1361538,
    #     1385849,
    #     200406,  # "JOHNSON & JOHNSON"
    #     789019,  # "MICROSOFT CORP"
    # ]

    zip_name = next(iter(zip_loaded))

    # get separated data sets w/ needed columns
    df_num = zip_loaded[zip_name]["num"][cols["cols_num"]]
    df_pre = zip_loaded[zip_name]["pre"][cols["cols_pre"]]
    df_sub = zip_loaded[zip_name]["sub"][cols["cols_sub"]]
    df_tag = zip_loaded[zip_name]["tag"][cols["cols_tag"]]

    # # SUB data set: apply 'form' filter:
    # # form: The submission type of the registrant’s filing.
    form_filter_k = df_sub["form"].str.contains("10-K")
    form_filter_q = df_sub["form"].str.contains("10-Q")
    df_sub = df_sub[(form_filter_k) | (form_filter_q)]
    df_sub = df_sub.dropna(subset=["cik", "sic"], how="any")

    # # NUM data set: apply 'tag' filter:
    # # tag: The unique identifier (name) for a tag in a specific taxonomy release.
    df_num = df_num[(df_num["tag"].isin(tags)) & (df_num["qtrs"] < 2)]

    # # PRE data set: apply 'stmt' filter: Non needed in real
    # # tag: The unique identifier (name) for a tag in a specific taxonomy release.
    # df_pre = df_pre[df_pre["stmt"].isin(stmt)]

    # # TAG data set: apply 'custom' filter: Non needed in real
    # # custom: 1 if tag is custom (version=adsh), 0 if it is standard.
    # # Note: This flag is technically redundant with the  version and adsh columns.
    # df_tag = df_tag[(df_tag["custom"] == 0)].drop(columns=["custom"])

    #! Some filtering for debugging. Non needed in real
    # name_filter = df_sub["name"].isin(["MICROSOFT CORP", "JOHNSON & JOHNSON"])
    # name_filter = df_sub["name"] == "JOHNSON & JOHNSON"
    # name_filter = df_sub["name"] == "MICROSOFT CORP"
    # df_sub = df_sub[name_filter]
    # df_sub = df_sub[df_sub["cik"].isin(ciks)]
    # df_num = df_num[df_num["qtrs"] < 2]

    # # Merge & convert
    df = pd.merge(df_sub, df_num, on=["adsh"])
    df = pd.merge(df, df_pre, on=["adsh", "tag", "version"], how="left")
    df = pd.merge(df, df_tag, on=["tag", "version"], how="left")

    df["zip"] = zip_name
    df["sic"] = df["sic"].astype(np.int64)
    df["ddate"] = pd.to_datetime(df["ddate"], format="%Y%m%d", errors="coerce")
    df["ddate"] = df["ddate"].dt.tz_localize(None)
    df.dropna(subset=["value"])

    del df_sub, df_num, df_pre, df_tag
    return df


def cols_rename(df):
    """Rename cols at pivoted DF to be shoter and convinient"""
    df.rename(
        columns={
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments": "IncomeLossFromOperations",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "StockholdersEquityIncludingPortion",
        },
        inplace=True,
    )
    return df


def cols_calculate_tags(df):
    # --- Revenues ---
    df[["SalesRevenueGoodsNet", "SalesRevenueServicesNet"]] = df[
        ["SalesRevenueGoodsNet", "SalesRevenueServicesNet"]
    ].fillna(0)
    df["SalesRevenueNet"].fillna(
        df["SalesRevenueGoodsNet"] + df["SalesRevenueServicesNet"],
        inplace=True,
    )

    def revenue(row):
        if pd.isna(row["RevenueFromContractWithCustomerExcludingAssessedTax"]):
            if row["SalesRevenueNet"] > 0:
                val = row["SalesRevenueNet"]
            else:
                val = row["Revenues"]
        else:
            val = max(
                row["RevenueFromContractWithCustomerExcludingAssessedTax"],
                row["SalesRevenueNet"],
            )
        return val

    df["Revenue"] = df.apply(revenue, axis=1)

    # --- CostOfGoodsAndServicesSold ---
    df["CostOfGoodsAndServicesSold"].fillna(
        df["Revenue"] - df["GrossProfit"],
        inplace=True,
    )
    df["CostOfGoodsAndServicesSold"].fillna(
        df["CostOfRevenue"],
        inplace=True,
    )

    # --- StockholdersEquity ---
    df["StockholdersEquity"].fillna(
        df["LiabilitiesAndStockholdersEquity"] - df["Liabilities"],
        inplace=True,
    )
    df["StockholdersEquity"].fillna(
        df["StockholdersEquityIncludingPortion"],
        inplace=True,
    )

    # --- ShortTermInvestments ---
    df["ShortTermInvestments"].fillna(
        df["CashCashEquivalentsAndShortTermInvestments"]
        - df["CashAndCashEquivalentsAtCarryingValue"],
        inplace=True,
    )
    df["ShortTermInvestments"].fillna(0, inplace=True)

    df = df.drop(
        [
            "CostOfRevenue",
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
            "SalesRevenueServicesNet",
            "LiabilitiesAndStockholdersEquity",
            "StockholdersEquityIncludingPortion",
            "CashCashEquivalentsAndShortTermInvestments",
        ],
        axis=1,
    )

    return df


def get_begining_values(df):
    df[
        [
            "Assets_b",
            "StockholdersEquity_b",
            "InventoryNet_b",
            "AccountsReceivableNetCurrent_b",
        ]
    ] = df.groupby("cik")[
        "Assets",
        "StockholdersEquity",
        "InventoryNet",
        "AccountsReceivableNetCurrent",
    ].shift(
        periods=1
    )
    # Sort columns
    cols = list(df.columns[:5]) + sorted(df.columns[5:])
    df = df[cols]
    return df


def calculate_coeffs(df):
    """Calculates the Key Financial Ratios"""
    # Solvency Ratios
    df["solvency_debt_ratio"] = df["Liabilities"] / df["Assets"]  #!
    df["solvency_debt_to_equity"] = df["Liabilities"] / df["StockholdersEquity"]
    # Interest coverage ratio
    df["solvency_irc"] = (df["IncomeLossFromOperations"] + df["InterestExpense"]) / df[
        "InterestExpense"
    ]

    # Liquidity Ratios
    df["liquid_current_ratio"] = df["AssetsCurrent"] / df["LiabilitiesCurrent"]  #!
    df["liquid_quick_ratio"] = (
        df["CashAndCashEquivalentsAtCarryingValue"]
        + df["ShortTermInvestments"]
        + df["AccountsReceivableNetCurrent"]
    ) / df[
        "LiabilitiesCurrent"
    ]  #!
    df["liquid_cash_ratio"] = (
        df["CashAndCashEquivalentsAtCarryingValue"] + df["ShortTermInvestments"]
    ) / df[
        "LiabilitiesCurrent"
    ]  # ?

    # Profitability Ratios
    df["profit_gross_margin"] = df["GrossProfit"] / df["Revenue"]
    df["profit_net_margin"] = df["NetIncomeLoss"] / df["Revenue"]  #!
    df["profit_operating_margin"] = (df["IncomeLossFromOperations"] + df["InterestExpense"]) / df[
        "Revenue"
    ]
    # ROE (Return on equity)
    df["profit_roe"] = (
        df["NetIncomeLoss"]
        * (365 / 90)
        / df[["StockholdersEquity", "StockholdersEquity_b"]].mean(axis=1)
    )
    # ROA (Return on assets)
    # df["profit_roa"] = df["NetIncomeLoss"] * (365 / 90) / df[["Assets", "Assets_b"]].mean(axis=1)
    df["profit_roa"] = df["NetIncomeLoss"] * (365 / 90) / df["Assets"]  #!

    # ---- Activity Ratios
    # Days inventory outstanding (Inventory turnover, days)
    # df["active_dio"] = (
    #     df[["InventoryNet", "InventoryNet_b"]].mean(axis=1) / df["CostOfGoodsAndServicesSold"] * 90
    # )
    df["active_dio"] = df["InventoryNet"] / df["CostOfGoodsAndServicesSold"] * 90  #!

    # Average collection period (Receivables turnover, days)
    # df["active_acp"] = (
    #     df[["AccountsReceivableNetCurrent", "AccountsReceivableNetCurrent_b"]].mean(axis=1)
    #     / df["Revenue"] * 90
    # )
    df["active_acp"] = df["AccountsReceivableNetCurrent"] / df["Revenue"] * 90  #!

    return df


# %% Main -----------------------------------------------------------------------
def main(
    start,
    finish,
    cut_ddate,
    create_eda,
):
    warnings.filterwarnings("ignore")
    pd_set_options()

    datasets = ["num", "pre", "sub", "tag"]
    tags = [
        "AccountsReceivableNetCurrent",  # BS
        "Assets",  # BS
        "AssetsCurrent",  # BS
        "CashAndCashEquivalentsAtCarryingValue",  # BS
        "CashCashEquivalentsAndShortTermInvestments",
        "ShortTermInvestments",
        "CostOfGoodsAndServicesSold",  # IS\ = Revenues - Gross profit
        "GrossProfit",  # IS
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",  # IS
        "InterestExpense",  # IS
        "InventoryNet",  # BS
        "Liabilities",  # BS
        "LiabilitiesCurrent",  # BS
        "NetIncomeLoss",  # IS
        "RevenueFromContractWithCustomerExcludingAssessedTax",  # IS
        "StockholdersEquity",  # BS\EQ
        #! additional tags for calculating some tags w/o data ------------------
        "LiabilitiesAndStockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "CostOfRevenue",
    ]
    cols_num = [
        "adsh",  #!
        "tag",  #!
        "version",  #!
        "ddate",
        "qtrs",  # < 2
        "value",
    ]
    cols_pre = [
        "adsh",  #!
        "tag",  #!
        "version",  #!
        "stmt",
    ]
    cols_sub = [
        "adsh",
        "sic",
        "cik",
        "countryba",
        "name",
        "form",
    ]
    cols_tag = [
        "tag",  #!
        "version",  #!
    ]
    cols = {
        "cols_num": cols_num,
        "cols_pre": cols_pre,
        "cols_sub": cols_sub,
        "cols_tag": cols_tag,
    }

    tic = time.time()
    # Get list for zip will be downloaded
    zip_names = get_zip_names(start, finish)
    df = None
    # Load & Get data sets from zips_loaded
    for zip_name in zip_names:
        # load one zip archine
        zip_loaded = load_zip(data_raw_dir, zip_name, datasets)
        # derive one filtered df
        df_one = df_get_from_zip(zip_loaded, cols, tags)
        # Stack the dfs gotten
        df = pd.concat([df, df_one])
        print(f"-- one DF shape: {df.shape} + {df_one.shape[0]}")

    # Keep only last "name" from a bunch related to one "cik"
    df_cik = df[["cik", "name"]].groupby("cik", as_index=False).last()
    df = pd.merge(df.drop(columns=["name"]), df_cik, on=["cik"], how="left")
    del df_cik

    # Delete duplicates w/ keeping last data
    df.drop_duplicates(
        subset=["cik", "tag", "ddate"],
        keep="last",
        inplace=True,
        ignore_index=True,
    )

    # Cut off by ddate
    if cut_ddate:
        df = df[df["ddate"] >= cut_ddate]
    df.sort_values(["cik", "ddate", "tag"])
    df.reset_index(drop=True, inplace=True)

    print(f"------------------------ Final DF --------------------------------")
    display(df.info())
    ## Evaluate time spent
    toc = time.time()
    min, sec = divmod(toc - tic, 60)
    print(f"Time spent for loading: {int(min)}min {int(sec)}sec")

    print(f"------------------------------------------------------------------")
    print(f"Pivot the tags as columns ...")
    df = df.pivot(
        index=("sic", "cik", "countryba", "name", "ddate"),
        columns=["tag"],
        values="value",
    ).reset_index(drop=False)
    df.columns.name = None
    df = cols_rename(df)

    print(f"Fill NA w/ mean of next and previous values...")
    # Should exclude no-data columns from interpolating
    cols_interpolate = list(set(df.columns.to_list()) - set(["sic", "countryba", "name", "ddate"]))
    df[cols_interpolate] = (
        df[cols_interpolate]
        .groupby("cik")
        .apply(
            lambda group: group.interpolate(
                method="linear",
                axis=0,
                limit=1,  # Maximum number of consecutive NaNs to fill
                inplace=False,  # Update the data in place if possible. default False
                limit_direction=None,
                limit_area="inside",
            )
        )
    )

    print(f"Calculate some additional tags...")
    df = cols_calculate_tags(df)

    print(f"Calculate the coeffitients...")
    df = get_begining_values(df)
    df = calculate_coeffs(df)

    df_get_glimpse(df)

    print(f"Save the cleaned data...")
    file_to_save = f"{start}_{finish}_precleaned.csv"
    df.to_csv(data_precleaned_dir / file_to_save, header=True, index=False)

    # Create DataPrep reports
    if create_eda:
        report_title = f"PERIOD: {start}-{finish} - DATA: Roughly Precleaned"
        report_name = f"{start}_{finish}_precleaned"
        make_dataprep_report(df, report_name, report_title)

    pd_reset_options()
    print("\n!!! DONE !!!")

    return df


# %% Run ------------------------------------------------------------------------
if __name__ == "__main__":
    df = main(
        start="2012q1",
        finish="2022q1",
        cut_ddate=None,
        create_eda=False,
    )


# %% Chek for development periods
def check_for_periods(df):
    from fhealth.utils import outliers_get_quantiles
    from fhealth.utils import outliers_get_zscores
    from fhealth.utils import outliers_ridoff

    dev_periods = [
        # "2018-12-31",  # 7329 : w/o 'active_dio': 12948
        "2019-03-31",  # 6543 : w/o 'active_dio': 11564 (11603)
        "2019-06-30",  # 5730 : w/o 'active_dio': 10190
        "2019-09-30",  # 5001 : w/o 'active_dio': 8899
        "2019-12-31",  # 4282 : w/o 'active_dio': 7590
        "2020-03-31",  # 3518 : w/o 'active_dio': 6219
        "2020-06-30",  # 2757 : w/o 'active_dio': 4891
        "2020-09-30",  # 2065 : w/o 'active_dio': 3684
        "2020-12-31",  # 1370 : w/o 'active_dio': 2455
        "2021-03-31",  # 645 : w/o 'active_dio': 1185
        "2021-04-30",
        "2021-05-31",
        "2021-05-30",
        "2021-07-31",
        "2021-08-31",
        "2021-09-30",
        "2021-10-31",
        "2021-10-30",
        "2012-12-31",
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
    df_1 = df.loc[df["ddate"].isin(dev_periods)]  # 236748 -> 55871
    df_1["countryba"] = df_1["countryba"].fillna("US")
    df_1 = df_1.replace([np.inf, -np.inf], np.nan)
    df_1 = df_1.dropna(axis=0, how="any", subset=cols_coeff)  # 65163 -> 9335
    # drop nonsensical
    df_1 = df_1.loc[
        (
            df_1[
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
    ]  # 64485 -> 9220
    df_1 = df_1.loc[(df_1[["profit_net_margin"]] < 2).all(axis=1)]  # 63506 -> 9220
    df_1 = df_1.loc[(df_1[["profit_net_margin"]] > -2).all(axis=1)]  # 52923 -> 8054

    # One-side outliers by positive quantile
    df_q = outliers_get_quantiles(
        df_1,
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
    df_1 = outliers_ridoff(df_1, df_q)  # ... -> 7790

    # Two-sides outliers by z-score
    df_z = outliers_get_zscores(
        df_1,
        cols_to_check=["profit_net_margin", "profit_roa"],
        sigma=2.5,
    )
    df_1 = outliers_ridoff(df_1, df_z)  # ... -> 7402

    # Sort df
    df_1 = df_1.sort_values(["cik", "ddate"], ignore_index=True)
    df_1 = df_1[["sic", "cik", "countryba", "name", "ddate"] + cols_coeff]
    # df_get_glimpse(df_1)
    print(df_1.shape)


check_for_periods(df)

# %%
