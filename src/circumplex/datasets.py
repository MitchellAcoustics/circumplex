# %%
import pandas as pd
from circumplex import instrument
from importlib.resources import files

_jz2017_path = str(files("circumplex.data").joinpath("jz2017.csv"))
JZ2017 = instrument.load_instrument("CSIP").attach_data(pd.read_csv(_jz2017_path))

_satp_path = str(files("circumplex.data").joinpath("SATP Dataset v1.4.xlsx"))
satp_data = pd.read_excel(_satp_path)
SATP_ENG = instrument.load_instrument("SATP-eng").attach_data(
    satp_data.query("Language == 'eng'")
)
