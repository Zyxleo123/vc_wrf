"""A markdown table from a DataFrame, without the optional `tabulate` dependency."""
from __future__ import annotations

import pandas as pd


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    cell = lambda v: floatfmt.format(v) if isinstance(v, float) else ("" if v is None else str(v))
    head = [str(c) for c in df.columns]
    rows = [[cell(v) for v in row] for row in df.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows])
