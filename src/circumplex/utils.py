import numpy as np
import pandas as pd
from circumplex.instrument import Scales, Instrument


def standardize(
    data: pd.DataFrame,
    scales: Scales,
    angles: np.array,
    instrument: Instrument,
    sample: int = 1,
    prefix: str = "",
    suffix: str = "_z",
):
    scale_names = scales.abbrev
    assert len(scale_names) == len(angles)

    key = instrument.norms.get_sample(sample)
    assert len(scale_names) == len(key)

    for i in range(len(angles)):
        scale_i = scale_names[i]
        new_var = f"{prefix}{scale_i}{suffix}"
        index_i = key["angle"][i]
        m_i = key["m"][i]
        s_i = key["sd"][i]
        data[new_var] = (data[scale_i] - m_i) / s_i

    return data
