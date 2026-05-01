from unittest.mock import patch

import pandas as pd

from circumplex.ssm import SSM, SSMDetails


def test_ssm_details_from_dict():
    data = {
        "boots": 2000,
        "interval": 0.95,
        "listwise": True,
        "angles": [0, 90, 180, 270],
        "contrast": False,
        "score_type": "mean",
    }
    details = SSMDetails.from_dict(data)
    assert details.boots == 2000
    assert details.interval == 0.95
    assert details.listwise is True
    assert details.angles == [0, 90, 180, 270]
    assert details.contrast is False
    assert details.score_type == "mean"


def test_ssm_details_to_dict():
    details = SSMDetails(
        2000,
        0.95,
        listwise=True,
        angles=[0, 90, 180, 270],
        contrast=False,
        score_type="mean",
    )
    data = details.to_dict()
    assert data["boots"] == 2000
    assert data["interval"] == 0.95
    assert data["listwise"] is True
    assert data["angles"] == [0, 90, 180, 270]
    assert data["contrast"] is False
    assert data["score_type"] == "mean"


def test_ssm_from_dict():
    data = {
        "results": pd.DataFrame(
            {
                "Label": ["A"],
                "e_est": [1.0],
                "x_est": [0.5],
                "y_est": [0.5],
                "fit_est": [0.9],
            }
        ),
        "scores": pd.DataFrame({"Score": [1, 2, 3]}),
        "details": {
            "boots": 2000,
            "interval": 0.95,
            "listwise": True,
            "angles": [0, 90, 180, 270],
            "contrast": False,
            "score_type": "mean",
        },
        "type": "profile",
    }
    ssm = SSM.from_dict(data)
    assert ssm.type == "profile"
    assert ssm.results.shape[0] == 1
    assert ssm.details.boots == 2000


def test_ssm_to_dict():
    details = SSMDetails(
        2000,
        0.95,
        listwise=True,
        angles=[0, 90, 180, 270],
        contrast=False,
        score_type="mean",
    )
    results = pd.DataFrame(
        {
            "Label": ["A"],
            "e_est": [1.0],
            "x_est": [0.5],
            "y_est": [0.5],
            "fit_est": [0.9],
        }
    )
    scores = pd.DataFrame({"Score": [1, 2, 3]})
    ssm = SSM(results, scores, details, "profile")
    data = ssm.to_dict()
    assert data["type"] == "profile"
    assert data["results"].shape[0] == 1
    assert data["details"]["boots"] == 2000


def test_ssm_summary():
    results = pd.DataFrame(
        {
            "Label": ["A"],
            # Estimates + CIs expected by summary
            "e_est": [1.0],
            "e_lci": [0.8],
            "e_uci": [1.2],
            "x_est": [0.5],
            "x_lci": [0.3],
            "x_uci": [0.7],
            "y_est": [0.5],
            "y_lci": [0.3],
            "y_uci": [0.7],
            "a_est": [0.71],
            "a_lci": [0.6],
            "a_uci": [0.8],
            "d_est": [0.0],
            "d_lci": [-0.1],
            "d_uci": [0.1],
            "fit_est": [0.9],
        }
    )
    scores = pd.DataFrame({"Score": [1, 2, 3]})
    details = SSMDetails(
        2000,
        0.95,
        listwise=True,
        angles=[0, 90, 180, 270],
        contrast=False,
        score_type="mean",
    )
    ssm = SSM(results, scores, details, "profile")

    with patch("builtins.print") as mocked_print:
        ssm.summary(rich_print=False)
        mocked_print.assert_called()  # Ensure print was called
