import pandas as pd

from agentloop_trader.display import dataframe_for_streamlit


def test_dataframe_for_streamlit_coerces_mixed_value_columns_without_mutating_source():
    source = pd.DataFrame(
        [
            {"Field": "Ticker", "Value": "AAPL"},
            {"Field": "Shares", "Value": 7},
            {"Field": "Ready", "Value": True},
        ]
    )

    display = dataframe_for_streamlit(source)

    assert list(display["Value"]) == ["AAPL", "7", "True"]
    assert source.loc[1, "Value"] == 7


def test_dataframe_for_streamlit_preserves_bool_columns_outside_display_text_columns():
    source = pd.DataFrame([{"Check": "Ready", "Passed": True}])

    display = dataframe_for_streamlit(source)

    assert display["Passed"].dtype == bool
