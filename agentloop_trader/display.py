from __future__ import annotations

from typing import Any

import pandas as pd


DISPLAY_TEXT_COLUMNS = {
    "Value",
    "Status / Value",
    "Read",
    "Plain English",
    "Detail",
    "Reason",
    "Reasons",
    "Message",
    "Next Action",
    "Current Read",
}


def _display_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def dataframe_for_streamlit(data: pd.DataFrame) -> pd.DataFrame:
    """Return a display-safe dataframe without changing the source records."""
    df = data.copy()
    for column in df.columns:
        if column in DISPLAY_TEXT_COLUMNS:
            df[column] = df[column].map(_display_text)
            continue
        if pd.api.types.is_object_dtype(df[column]):
            inferred = pd.api.types.infer_dtype(df[column], skipna=True)
            if inferred.startswith("mixed") or inferred in {"bytes", "string"}:
                df[column] = df[column].map(_display_text)
    return df
