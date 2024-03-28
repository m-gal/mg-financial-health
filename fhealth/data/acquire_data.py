"""
    LOAD raw data sets from https://www.sec.gov/dera/data/financial-statement-data-sets.html
    and SAVE them to the [../data/raw].

    @author: mikhail.galkin
"""

# %% Load libraries -------------------------------------------------------------
import sys
import requests


# %% Load project's stuff -------------------------------------------------------
sys.path.extend(["../.."])
from fhealth.config import data_raw_dir


# %% Custom funcs ---------------------------------------------------------------
def download_local(site, file):

    url = site + file
    file_to_save = data_raw_dir / file

    r = requests.get(url, stream=True)
    if r.ok:
        print(f"Downloading...: {file_to_save}")
        with open(file_to_save, "wb") as f:
            f.write(r.content)
    else:
        print(f"Download failed: status code: {r.status_code}\n{r.text}")


def get_zip_names(start, finish):
    qs = ["q1", "q2", "q3", "q4"]
    ys = [
        "2010",
        "2011",
        "2012",
        "2013",
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",
        "2021",
        "2022",
    ]

    zip_names = []
    for y in ys:
        for q in qs:
            file = y + q
            # print(f"{file}")
            zip_names.append(file)

    zip_names = zip_names[zip_names.index(start) : zip_names.index(finish) + 1]

    return zip_names


# %% Main -----------------------------------------------------------------------
def main(
    start="2012q1",
    finish="2021q2",
):
    site = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/"
    zips = get_zip_names(start=start, finish=finish)

    for zip in zips:
        file = zip + ".zip"
        download_local(site=site, file=file)

    print("\n!!! DONE !!!")


# %% Run ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
