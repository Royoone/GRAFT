"""A small prediction API over the frozen GRAFT model bundle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch

from .io import align_response, load_bundle, load_weights, lookup, read_drugs, read_response_csv, sha256, unique_strings
from .model import GRAFT, GRAFTAdapter


@dataclass
class PredictionResult:
    """An ensemble response matrix with named axes and run provenance."""

    predictions: np.ndarray
    drug_ids: np.ndarray
    genes: np.ndarray
    source_mean: np.ndarray
    metadata: dict

    def save(self, output: str | Path, *, csv: bool = False) -> None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "predictions.npz", predictions=self.predictions,
                            drug_ids=self.drug_ids, genes=self.genes, source_mean=self.source_mean,
                            context=np.asarray(self.metadata["context"]), mode=np.asarray(self.metadata["mode"]))
        rms = np.sqrt(np.mean(self.predictions.astype(np.float64)**2, axis=1))
        summary = pd.DataFrame({"drug_id": self.drug_ids, "response_rms": rms})
        summary.to_csv(output / "summary.tsv", sep="\t", index=False)
        (output / "run.json").write_text(json.dumps(self.metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if csv:
            table = pd.DataFrame(self.predictions, columns=self.genes)
            table.insert(0, "drug_id", self.drug_ids)
            table.to_csv(output / "predictions.csv", index=False)


class GRAFTPredictor:
    """Load one matching bundle; predict one target context at a time.

    CPU is the default. ``predict`` accepts named response DataFrames;
    ``predict_files`` is the corresponding CSV/text-file entry point.
    """

    def __init__(self, bundle: str | Path, *, device: str = "cpu", chunk_size: int = 256) -> None:
        self.bundle = Path(bundle)
        self.manifest, self.reference = load_bundle(self.bundle)
        self.genes = self.reference["genes"]
        self.device = torch.device(device)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.models = []
        for entry in self.manifest["models"]:
            model = GRAFT(genes=len(self.genes), seed=entry["seed"], device=self.device)
            load_weights(model, self.bundle / entry["file"])
            model.eval().requires_grad_(False)
            self.models.append(model)
        self.adapter = None
        entry = self.manifest.get("adapter")
        if entry is not None:
            if entry["kind"] != "ridge_remix":
                raise ValueError("This usage release expects the frozen ridge_remix adapter")
            self.adapter = GRAFTAdapter(n_genes=len(self.genes), rank=entry["rank"]).to(self.device)
            load_weights(self.adapter, self.bundle / entry["file"])
            self.adapter.eval().requires_grad_(False)

    def predict(self, genetic: pd.DataFrame, drug_ids, *, anchors: pd.DataFrame | None = None,
                context: str = "target_context") -> PredictionResult:
        genetic_ids, genetic_values = align_response(genetic, "perturbation_id", self.genes)
        if anchors is None:
            anchor_ids, anchor_values = np.array([], dtype=str), np.empty((0, len(self.genes)), np.float32)
        else:
            anchor_ids, anchor_values = align_response(anchors, "drug_id", self.genes)
        return self._predict(genetic_ids, genetic_values, drug_ids, anchor_ids, anchor_values, context)

    def predict_files(self, genetic: str | Path, drugs: str | Path, *, anchors: str | Path | None = None,
                      context: str = "target_context") -> PredictionResult:
        genetic_ids, genetic_values = read_response_csv(genetic, "perturbation_id", self.genes)
        drug_ids = read_drugs(drugs)
        if anchors is None:
            anchor_ids, anchor_values = np.array([], dtype=str), np.empty((0, len(self.genes)), np.float32)
        else:
            anchor_ids, anchor_values = read_response_csv(anchors, "drug_id", self.genes)
        result = self._predict(genetic_ids, genetic_values, drug_ids, anchor_ids, anchor_values, context)
        result.metadata["input_sha256"] = {"genetic": sha256(genetic), "drugs": sha256(drugs)}
        if anchors is not None:
            result.metadata["input_sha256"]["anchors"] = sha256(anchors)
        return result

    @torch.no_grad()
    def _predict(self, genetic_ids, genetic_values, drug_ids, anchor_ids, anchor_values, context) -> PredictionResult:
        drug_ids = unique_strings(drug_ids, "query drug IDs")
        if not len(drug_ids):
            raise ValueError("At least one query drug is required")
        if not isinstance(context, str) or not context.strip():
            raise ValueError("context must be a nonempty label")
        overlap = sorted(set(drug_ids) & set(anchor_ids))
        if overlap:
            raise ValueError(f"Anchor and query drug IDs must be disjoint: {overlap[:5]}")
        if len(anchor_ids) and self.adapter is None:
            raise ValueError("The bundle has no compatible adapter for drug anchors")
        if len(genetic_ids) < 2:
            warnings.warn("Fewer than two genetic support conditions: the genetic state is zero; "
                          "D0 equals the source same-drug mean.", UserWarning, stacklevel=2)
        gi = lookup(genetic_ids, self.reference["genetic_ids"], "genetic perturbation IDs")
        all_drugs = np.concatenate([drug_ids, anchor_ids])
        di = lookup(all_drugs, self.reference["drug_ids"], "drug IDs")
        genetic_ref = self.reference["genetic_reference"][gi]
        drug_ref = self.reference["drug_reference"][di]
        n_query = len(drug_ids)
        query_idx = torch.arange(n_query, device=self.device)
        anchor_idx = torch.arange(n_query, len(all_drugs), device=self.device)
        predictions = []
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        for entry, model in zip(self.manifest["models"], self.models):
            scale = entry["scale"]
            support = {"genetic": {"x": tensor(self.reference["genetic_embeddings"][gi]),
                                   "mean": tensor(genetic_ref / scale),
                                   "delta": tensor((genetic_values - genetic_ref) / scale)}}
            q = {"x": tensor(self.reference["drug_embeddings"][di]), "mean": tensor(drug_ref / scale)}
            base, features = model.inference(q, support, chunk=self.chunk_size, return_features=True)
            base = base * scale
            if len(anchor_ids):
                observed = torch.full_like(base, float("nan"))
                observed[anchor_idx] = tensor(anchor_values)
                prediction = self.adapter(base, observed, features, anchor_idx, query_idx)
            else:
                prediction = base[:n_query]
            predictions.append(prediction.cpu().numpy().astype(np.float64))
        ensemble = np.mean(predictions, axis=0).astype(np.float32)
        if not np.isfinite(ensemble).all():
            raise FloatingPointError("Prediction produced non-finite values; check input scales")
        metadata = {
            "software": "GRAFT", "software_version": "0.1.0",
            "bundle_version": self.manifest["bundle_version"],
            "bundle_manifest_sha256": sha256(self.bundle / "manifest.json"),
            "context": context, "mode": "GRAFT-Adapt" if len(anchor_ids) else "D0",
            "seeds": [entry["seed"] for entry in self.manifest["models"]],
            "sources": self.manifest["sources"], "n_genetic": len(genetic_ids),
            "n_anchors": len(anchor_ids), "n_queries": n_query, "n_genes": len(self.genes),
            "genetic_ids": genetic_ids.tolist(), "anchor_drug_ids": anchor_ids.tolist(),
            "query_drug_ids": drug_ids.tolist(), "query_unit": "drug record ID",
            "genetic_state_zero_by_support_count": len(genetic_ids) < 2,
            "gene_alignment": "complete panel matched by gene name",
            "response_scale": "original prepared response scale", "device": str(self.device),
            "ensemble": "arithmetic mean of signed gene-level predictions before RMS",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {"numpy": np.__version__, "torch": torch.__version__},
        }
        return PredictionResult(ensemble, drug_ids, self.genes.copy(), drug_ref[:n_query].copy(), metadata)
