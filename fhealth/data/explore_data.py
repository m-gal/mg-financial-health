"""
    LOAD raw data
    and GET a first glimpse
    and MAKE Exploratory Data Analysis w\o any changing original data
    and SAVE results of EDA to the [./project/reports].

    @author: mikhail.galkin
"""

# %% Load libraries -------------------------------------------------------------
import sys
import warnings
import zipfile
import pandas as pd
import sweetviz as sv
from dataprep.eda import create_report
from IPython.core.display import display


# %% Load project's stuff -------------------------------------------------------
sys.path.extend(["../.."])
from fhealth.config import data_raw_dir
from fhealth.config import reports_dir
from fhealth.utils import pd_set_options
from fhealth.utils import pd_reset_options
from fhealth.utils import df_get_glimpse


# %% Custom funcs ---------------------------------------------------------------
def load_zip(zip_dir, zip_name, datasets):
    """Load raw txt dataset from zip archive

    Args:
        zip_dir (str): folder with zips
        zip_name (str): zip archive's name
        datasets (list of str): set's name from ["num", "pre", "sub", "tag"]

    Returns:
        zip_loaded: dict for one zipped period w/ Pandas DataFrames
    """
    datafiles = [dataset + ".txt" for dataset in datasets]

    zip_file = zip_dir / str(zip_name + ".zip")
    zfile = zipfile.ZipFile(zip_file)

    # Dict for loaded datasets
    dfs = {}
    print(f"-------------- {zip_name} --------------")
    for i, datafile in enumerate(datafiles):
        # ## num of records in dataset (excludes header)
        # n_len = sum(1 for line in zfile.open(datafile)) - 1
        # print(f"\nOriginal: ./{zip_name}/{datafile} contain {n_len} rows...")

        df = pd.read_csv(
            zfile.open(datafile),
            sep="\t",
            header=0,
            # nrows=4000,
        )
        print(f"./{zip_name}/{datafile}: Loaded {len(df)} rows X {len(df.columns)} cols...")
        # display(df.sample(n=5, random_state=42).T)

        # Add dataset-df pair to dict
        dfs[datasets[i]] = df
        # Dict for whole loaded zip w/ datasets
        zip_loaded = {}
        zip_loaded[zip_name] = dfs

    return zip_loaded


# def coerse_to_date_in_num(zip_loaded, zip_name):
#     # At NUM dataset
#     zip_loaded[zip_name]["num"]["ddate"] = pd.to_datetime(
#         zip_loaded[zip_name]["num"]["ddate"],
#         format="%Y%m%d",
#         errors="raise",
#     ).dt.date
#     return True


def coerse_to_date_in_num_set(zip_loaded):
    # At NUM dataset
    # Getting first key in dictionary using next() + iter()
    zip_name = next(iter(zip_loaded))
    zip_loaded[zip_name]["num"]["ddate"] = pd.to_datetime(
        zip_loaded[zip_name]["num"]["ddate"],
        format="%Y%m%d",
        errors="raise",
    ).dt.date
    return True


def coerse_to_date_in_sub_set(zip_loaded):
    # At SUB dataset
    zip_name = next(iter(zip_loaded))
    zip_loaded[zip_name]["sub"]["changed"] = pd.to_datetime(
        zip_loaded[zip_name]["sub"]["changed"],
        format="%Y%m%d",
        errors="raise",
    ).dt.date

    zip_loaded[zip_name]["sub"]["accepted"] = pd.to_datetime(
        zip_loaded[zip_name]["sub"]["accepted"],
        format="%Y%m%d %H:%M:%S",
        errors="raise",
    ).dt.date

    zip_loaded[zip_name]["sub"]["period"] = pd.to_datetime(
        zip_loaded[zip_name]["sub"]["period"],
        format="%Y%m%d",
        errors="raise",
    ).dt.date

    zip_loaded[zip_name]["sub"]["filed"] = pd.to_datetime(
        zip_loaded[zip_name]["sub"]["filed"],
        format="%Y%m%d",
        errors="raise",
    ).dt.date
    return True


def make_dataprep_report(df, report_name=None, report_title=None):
    print(f"\nDataPrep report start...")
    report_name = f"{report_name}-dataprep"
    report = create_report(df, report_title)
    report.save(filename=report_name, to=reports_dir)
    report.show_browser()


# ------------------------------------------------------------------------------
# Sweetviz Exploratary Data Analysis
# ------------------------------------------------------------------------------
def config_sweetviz():
    """feat_cfg:
    A FeatureConfig object representing features to be skipped,
    or to be forced a certain type in the analysis.
    The arguments can either be a single string or list of strings.
    Parameters are skip, force_cat, force_num and force_text.
    The "force_" arguments override the built-in type detection.
    They can be constructed as follows:
        feature_config = sv.FeatureConfig(skip="PassengerId", force_text=["Age"])
    """
    print(f"\nConfig SweetViz...")
    sv.config_parser.set(section="Layout", option="show_logo", value="0")
    feature_config = sv.FeatureConfig(skip=None, force_text=None)
    return feature_config


def make_sweetviz_analyze(df, report_name=None, target_feat=None):
    """The analyze() function can take multiple other arguments:
    analyze(source: Union[pd.DataFrame, Tuple[pd.DataFrame, str]],
        target_feat: str = None,
        feat_cfg: FeatureConfig = None,
        pairwise_analysis: str = 'auto')
    """
    print(f"\nSweetViz analysis report start...")
    report_name = f"{report_name}-analyze.html"
    feature_config = config_sweetviz()
    report = sv.analyze(
        source=df,
        target_feat=target_feat,
        feat_cfg=feature_config,
        pairwise_analysis="on",
    )
    report.show_html(reports_dir / report_name)
    return report


# %% Main -----------------------------------------------------------------------
def main(
    zip_name="2021q1",
    datasets=["num", "pre", "sub", "tag"],
):
    warnings.filterwarnings("ignore")
    pd_set_options()

    zip_loaded = load_zip(data_raw_dir, zip_name, datasets)
    coerse_to_date_in_num_set(zip_loaded)
    coerse_to_date_in_sub_set(zip_loaded)

    for dataset in zip_loaded[zip_name].keys():
        df = zip_loaded[zip_name][dataset]
        report_name = f"{zip_name}_{dataset}"

        df_get_glimpse(df)

        # Create DataPrep reports
        report_title = f"PERIOD: {zip_name} - DATA: {dataset}"
        make_dataprep_report(df, report_name, report_title)

        # Create SweetViz analyze report
        make_sweetviz_analyze(df, report_name, target_feat=None)

    pd_reset_options()
    print("\n!!! DONE !!!")


# %% Run ------------------------------------------------------------------------
if __name__ == "__main__":
    main()


# %% Debugging stuff ------------------------------------------------------------
# zip_name = "2021q1"
# datasets = ["num", "pre", "sub", "tag"]

# zips_loaded = {}
# for zip_name in ["2020q4", "2021q1"]:
#     zip_loaded = load_zip(data_raw_dir, zip_name, datasets)
#     coerse_to_date_in_num_set(zip_loaded)
#     coerse_to_date_in_sub_set(zip_loaded)
#     zips_loaded.update(zip_loaded)

# print(zips_loaded.keys())

# display(zips_loaded["2020q4"]["num"])
# display(zips_loaded["2021q1"]["num"])

# display(zips_loaded["2020q4"]["num"]["ddate"])
# display(zips_loaded["2021q1"]["num"]["ddate"])
