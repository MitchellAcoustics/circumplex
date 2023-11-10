import numpy as np
import pandas as pd
import pytest

import circumplex

SCALES = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8")


def fixed_data(angles=circumplex.OCTANTS, ampl=0.5, disp=180, elev=0):
    return circumplex.cosine_form(np.array(angles), ampl, disp, elev)


@pytest.fixture
def ssm_params():
    scores = fixed_data()
    ssm_params = circumplex.SSMParams(
        scores, SCALES, circumplex.OCTANTS, group="group", measure="measure"
    )
    return ssm_params


def test_ssm_parameters():
    # elev, xval, yval, ampl, disp, r2
    fix_data = fixed_data()
    fixed_ssm = circumplex.ssm_parameters(fix_data, circumplex.OCTANTS)
    np.testing.assert_allclose(fixed_ssm, (0.0, -0.5, 0, 0.5, 180, 1.0), atol=1e-4)

    fix_data = fixed_data(ampl=0.5, disp=45, elev=0)
    fixed_ssm = circumplex.ssm_parameters(fix_data, circumplex.OCTANTS)
    np.testing.assert_allclose(
        fixed_ssm, (0.0, 0.35355, 0.35355, 0.5, 45, 1.0), atol=1e-4
    )

    fix_data = fixed_data(ampl=0.3, disp=90, elev=0.1)
    fixed_ssm = circumplex.ssm_parameters(fix_data, circumplex.OCTANTS)
    np.testing.assert_allclose(fixed_ssm, (0.1, 0.0, 0.3, 0.3, 90, 1.0), atol=1e-4)


def test_label(ssm_params):
    assert ssm_params.label == "group_measure"


def test_table(ssm_params):
    assert isinstance(ssm_params.table, pd.DataFrame)


def test_params(ssm_params):
    params = ssm_params.params
    assert isinstance(params, dict)
    assert len(params) == 9
