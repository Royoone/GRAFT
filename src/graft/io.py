"""Read named response matrices and frozen, non-pickle model bundles."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_strings(values, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=str)
    if array.ndim != 1 or any(not value or value != value.strip() for value in array):
        raise ValueError(f"{label} must be a one-dimensional list of nonempty, unpadded strings")
    if len(set(array)) != len(array):
        raise ValueError(f"Duplicate {label} are not allowed")
    return array


def align_response(frame: pd.DataFrame, id_column: str, genes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate identifiers and finite values, then align gene columns by name."""
    columns = list(frame.columns)
    if not columns or columns[0] != id_column:
        raise ValueError(f"The first column must be {id_column!r}")
    if len(set(columns)) != len(columns):
        raise ValueError("Duplicate column names are not allowed")
    expected = set(genes.tolist())
    supplied = set(columns[1:])
    missing, extra = sorted(expected - supplied), sorted(supplied - expected)
    if missing or extra:
        raise ValueError(f"Gene panel mismatch: {len(missing)} missing, {len(extra)} extra columns. "
                         f"Missing examples: {missing[:5]}; extra examples: {extra[:5]}")
    if frame[id_column].isna().any():
        raise ValueError(f"Missing {id_column} values are not allowed")
    ids = unique_strings(frame[id_column].to_numpy(), id_column)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            values = frame.loc[:, genes.tolist()].to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Response columns must contain numeric values") from error
    if not np.isfinite(values).all():
        raise ValueError("Responses must be finite; NaN, infinity and float32 overflow are not allowed")
    return ids, values


def read_response_csv(path: str | Path, id_column: str, genes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Check the original header before pandas can rename duplicate columns.
    with open(path, encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names are not allowed")
    frame = pd.read_csv(path, dtype={id_column: str}, encoding="utf-8-sig")
    return align_response(frame, id_column, genes)


def read_drugs(path: str | Path) -> np.ndarray:
    with open(path, encoding="utf-8-sig") as handle:
        drugs = unique_strings([line.strip() for line in handle if line.strip()], "query drug IDs")
    if not len(drugs):
        raise ValueError("The query drug list is empty")
    return drugs


def load_bundle(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    path = Path(path)
    with open(path / "manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("model") != "GRAFT" or manifest.get("n_genes") != 976:
        raise ValueError("This release requires a GRAFT bundle with the 976-gene panel")
    models = manifest.get("models", [])
    if not models or len({m["seed"] for m in models}) != len(models):
        raise ValueError("The bundle must have distinct, declared model seeds")
    entries = [manifest["reference"], *models, *manifest.get("tables", [])]
    if manifest.get("adapter"):
        entries.append(manifest["adapter"])
    for entry in entries:
        name = entry["file"]
        if Path(name).name != name:
            raise ValueError("Bundle file names must be local file names")
        if sha256(path / name) != entry["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    with np.load(path / manifest["reference"]["file"], allow_pickle=False) as archive:
        reference = {key: archive[key] for key in archive.files}
    reference["genes"] = unique_strings(reference["genes"], "model gene names")
    if len(reference["genes"]) != 976:
        raise ValueError("The reference gene panel must contain 976 unique names")
    for kind, count in [("genetic", manifest["n_genetic"]), ("drug", manifest["n_drugs"])]:
        ids = unique_strings(reference[kind + "_ids"], kind + " IDs")
        if len(ids) != count:
            raise ValueError(f"The {kind} catalog size does not match the manifest")
        reference[kind + "_ids"] = ids
        for field, width in [("embeddings", 256), ("reference", 976)]:
            array = reference[kind + "_" + field]
            if array.shape != (count, width) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Invalid {kind} {field} shape, dtype or values")
        counts = reference[kind + "_source_counts"]
        if counts.shape != (count,) or np.any(counts < 1) or np.any(counts > len(manifest["sources"])):
            raise ValueError(f"Invalid source counts for {kind}")
    for model in models:
        if not np.isfinite(model["scale"]) or model["scale"] <= 0 or model.get("gain", 1) != 1:
            raise ValueError("Invalid trained response scale or gain")
    return manifest, reference


def load_weights(module: torch.nn.Module, path: str | Path) -> None:
    """Strictly load NPZ tensors without executing Python pickle payloads."""
    with np.load(path, allow_pickle=False) as archive:
        state = {}
        for key in archive.files:
            array = archive[key]
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Invalid parameter tensor: {key}")
            state[key] = torch.from_numpy(array.copy())
    module.load_state_dict(state, strict=True)


def lookup(ids: np.ndarray, catalog: np.ndarray, kind: str) -> np.ndarray:
    positions = {name: index for index, name in enumerate(catalog)}
    unknown = [name for name in ids if name not in positions]
    if unknown:
        raise ValueError(f"Unknown {kind}: {unknown[:8]}. Use IDs supported by the bundle.")
    return np.asarray([positions[name] for name in ids], dtype=np.int64)
