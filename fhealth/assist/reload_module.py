"""
    Helps to reload project's module and get its inspections
    w\o reloading working space

    @author: mikhail.galkin
"""

# %% Import libs
import sys
import inspect
import importlib

sys.path.extend(["../.."])
import fhealth

# %% ------------------------------ CONFIG --------------------------------------
import fhealth.config

importlib.reload(fhealth.config)
from fhealth.config import model1_path

print(model1_path)

# %% ------------------------------ UTILS ---------------------------------------
import fhealth.utils

importlib.reload(fhealth.utils)
print(inspect.getsource(fhealth.utils.pd_set_options))


# %% ------------------------------ PLOTS ---------------------------------------
import fhealth.plots

importlib.reload(fhealth.plots)
print(inspect.getsource(fhealth.plots.plot_cm))


# %% --------------------------- MODEL: XGB -------------------------------------
import fhealth.model1._xgb_class

importlib.reload(fhealth.model1._xgb_class)
print(inspect.getsource(fhealth.model1._xgb_class.model))
print(inspect.getsource(fhealth.model1._xgb_class.param_search))


# %% --------------------------- MODEL: NN -------------------------------------
import fhealth.model2._nn

importlib.reload(fhealth.model2._nn)
print(inspect.getsource(fhealth.model2._nn.vanilla_multiin))


# %% ------------------------------ TEST ----------------------------------------
import fhealth.tests.test_pytest

importlib.reload(fhealth.tests.test_pytest)
print(inspect.getsource(fhealth.tests.test_pytest.test_model1_get_rating))

# %%
