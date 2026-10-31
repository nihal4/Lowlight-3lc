"""Submission building and validation.

Validation mirrors the official 3LC plugin's pre-flight checks (which in turn
mirror the competition metric parser), so a malformed CSV fails here instead of
burning one of your 3 daily Kaggle submissions.
"""

from __future__ import annotations

from .common import EXPECTED_TEST_ROWS, NUM_CLASSES, SUBMISSION_COLUMNS


def format_boxes(cls, confs, xywhn) -> str:
    """Model outputs -> 'class conf xc yc w h ...' or 'no box'."""
    clamp = lambda v: min(1.0, max(0.0, float(v)))  # noqa: E731
    parts = []
    for c, cf, (xc, yc, w, h) in zip(cls, confs, xywhn):
        c = int(c)
        if 0 <= c < NUM_CLASSES:
            parts.append(f"{c} {clamp(cf):.5f} {clamp(xc):.5f} {clamp(yc):.5f} {clamp(w):.5f} {clamp(h):.5f}")
    return " ".join(parts) if parts else "no box"


def build_df(pred_map: dict[str, str]):
    import pandas as pd

    ids = sorted(pred_map)
    return pd.DataFrame(
        {"id": range(len(ids)), "image_id": ids, "prediction_string": [pred_map[i] for i in ids]}
    )


def _check_pred_string(s: str, image_id: str) -> int:
    s = str(s).strip()
    if s == "" or s.lower() == "no box" or s.lower() == "nan":
        return 0
    where = f" (image_id {image_id!r})"
    try:
        v = list(map(float, s.split()))
    except ValueError:
        raise ValueError(f"non-numeric values in prediction_string{where}")
    if len(v) % 6:
        raise ValueError(f"prediction_string must be groups of 6 values, got {len(v)}{where}")
    for i in range(0, len(v), 6):
        c, cf, xc, yc, w, h = v[i : i + 6]
        if not (float(c).is_integer() and 0 <= c <= NUM_CLASSES - 1):
            raise ValueError(f"bad class id {c}{where}")
        if not (0 <= cf <= 1 and all(0 <= x <= 1 for x in (xc, yc, w, h))):
            raise ValueError(f"confidence/coords outside [0,1]: {v[i:i+6]}{where}")
    return len(v) // 6


def validate_df(df, expected_rows: int | None = EXPECTED_TEST_ROWS, sample_ids=None) -> dict:
    """Raise ValueError on the first violation; return a summary dict."""
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"columns must be {SUBMISSION_COLUMNS}, got {list(df.columns)}")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(df)}")
    ids = df["image_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"{int(ids.duplicated().sum())} duplicated image_id rows")
    if list(df["id"]) != list(range(len(df))) or list(ids) != sorted(ids):
        raise ValueError("id must be 0..N-1 aligned to image_id sorted ascending")
    if sample_ids is not None:
        missing = set(map(str, sample_ids)) - set(ids)
        extra = set(ids) - set(map(str, sample_ids))
        if missing or extra:
            raise ValueError(f"image_id set differs from sample_submission "
                             f"(missing {len(missing)}, extra {len(extra)})")
    n_boxes = 0
    per_class = [0] * NUM_CLASSES
    empty = 0
    for image_id, s in zip(ids, df["prediction_string"]):
        n = _check_pred_string(s, image_id)
        n_boxes += n
        if n == 0:
            empty += 1
        else:
            vals = str(s).split()
            for i in range(0, len(vals), 6):
                per_class[int(float(vals[i]))] += 1
    return {"rows": len(df), "boxes": n_boxes, "empty_images": empty,
            "boxes_per_image": round(n_boxes / max(len(df), 1), 2), "per_class": per_class}
