"""Behavioral checks on the supplied frozen bundle and real small example."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from graft import GRAFTPredictor
from graft.io import read_response_csv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def predictor():
    torch.set_num_threads(2)
    return GRAFTPredictor(ROOT / "resources")


@pytest.fixture(scope="module")
def inputs():
    genetic = pd.read_csv(ROOT / "examples/genetic_responses.csv")
    anchors = pd.read_csv(ROOT / "examples/anchors.csv")
    drugs = (ROOT / "examples/query_drugs.txt").read_text().splitlines()
    return genetic, anchors, drugs


def test_real_example_and_named_axis_invariance(predictor, inputs):
    genetic, _, drugs = inputs
    result = predictor.predict(genetic, drugs)
    assert result.predictions.shape == (5, 976)
    assert np.isfinite(result.predictions).all()
    # Both support rows and gene columns may arrive in a different order.
    reordered = genetic.iloc[::-1, [0, *range(genetic.shape[1]-1, 0, -1)]]
    other = predictor.predict(reordered, list(reversed(drugs)))
    np.testing.assert_allclose(other.predictions[::-1], result.predictions, rtol=2e-5, atol=2e-5)
    assert result.metadata["seeds"] == [52, 229, 564]


@pytest.mark.parametrize("n_support", [0, 1])
def test_insufficient_support_returns_source_mean(predictor, inputs, n_support):
    genetic, _, drugs = inputs
    with pytest.warns(UserWarning, match="state is zero"):
        result = predictor.predict(genetic.iloc[:n_support], drugs)
    np.testing.assert_allclose(result.predictions, result.source_mean, rtol=2e-6, atol=2e-6)


def test_zero_anchor_matches_d0_and_adapter_does_not_retrain(predictor, inputs):
    genetic, anchors, drugs = inputs
    before = {key: value.clone() for key, value in predictor.adapter.state_dict().items()}
    d0 = predictor.predict(genetic, drugs)
    empty = predictor.predict(genetic, drugs, anchors=anchors.iloc[:0])
    np.testing.assert_array_equal(empty.predictions, d0.predictions)
    adapted = predictor.predict(genetic, drugs, anchors=anchors)
    assert adapted.metadata["mode"] == "GRAFT-Adapt"
    assert np.max(np.abs(adapted.predictions-d0.predictions)) > 1e-5
    for key, value in predictor.adapter.state_dict().items():
        assert torch.equal(value, before[key])
    expected = pd.read_csv(ROOT / "examples/expected_summary.tsv", sep="\t")
    for result, column in [(d0, "D0_response_rms"), (adapted, "Adapt_response_rms")]:
        rms = np.sqrt((result.predictions.astype(float)**2).mean(1))
        np.testing.assert_allclose(rms, expected[column], rtol=2e-5, atol=2e-5)


def test_query_batch_does_not_change_prediction(predictor, inputs):
    genetic, anchors, drugs = inputs
    full = predictor.predict(genetic, drugs, anchors=anchors)
    single = predictor.predict(genetic, drugs[:1], anchors=anchors)
    np.testing.assert_allclose(single.predictions, full.predictions[:1], rtol=2e-5, atol=2e-5)


def test_reject_unknown_identity_and_overlap(predictor, inputs):
    genetic, anchors, drugs = inputs
    with pytest.raises(ValueError, match="Unknown drug"):
        predictor.predict(genetic, ["unsupported_drug"])
    bad = genetic.copy()
    bad.iloc[0, 0] = "unsupported_genetic_identity"
    with pytest.raises(ValueError, match="Unknown genetic"):
        predictor.predict(bad, drugs)
    with pytest.raises(ValueError, match="disjoint"):
        predictor.predict(genetic, [anchors.iloc[0, 0]], anchors=anchors)


def test_reject_missing_nonfinite_and_duplicate_inputs(predictor, inputs):
    genetic, _, drugs = inputs
    with pytest.raises(ValueError, match="Gene panel mismatch"):
        predictor.predict(genetic.iloc[:, :-1], drugs)
    with pytest.raises(ValueError, match="Duplicate"):
        predictor.predict(pd.concat([genetic, genetic.iloc[:1]]), drugs)
    bad = genetic.copy()
    bad.iloc[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        predictor.predict(bad, drugs)
    with pytest.raises(ValueError, match="Duplicate"):
        predictor.predict(genetic, [drugs[0], drugs[0]])


def test_reject_duplicate_csv_header_before_pandas_renaming(predictor, tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text("perturbation_id,GNPDA1,GNPDA1\nABC,1,2\n")
    with pytest.raises(ValueError, match="Duplicate CSV"):
        read_response_csv(path, "perturbation_id", predictor.genes)


def test_results_roundtrip_and_rms(predictor, inputs, tmp_path):
    genetic, _, drugs = inputs
    result = predictor.predict(genetic, drugs, context="test_context")
    result.save(tmp_path, csv=True)
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["predictions"], result.predictions)
        np.testing.assert_array_equal(z["genes"], predictor.genes)
        assert str(z["context"]) == "test_context"
    summary = pd.read_csv(tmp_path / "summary.tsv", sep="\t")
    np.testing.assert_allclose(summary.response_rms, np.sqrt((result.predictions.astype(float)**2).mean(1)))
    assert pd.read_csv(tmp_path / "predictions.csv").shape == (5, 977)
