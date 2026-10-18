"""Run the supplied small HepG2 example on CPU after `pip install -e .`."""

from pathlib import Path

from graft import GRAFTPredictor

ROOT = Path(__file__).resolve().parents[1]
predictor = GRAFTPredictor(ROOT / "resources")
result = predictor.predict_files(ROOT / "examples/genetic_responses.csv",
                                 ROOT / "examples/query_drugs.txt", context="HepG2_demo")
result.save(ROOT / "results/demo")
print(f"Saved {result.predictions.shape} D0 matrix to results/demo")

adapted = predictor.predict_files(ROOT / "examples/genetic_responses.csv",
                                  ROOT / "examples/query_drugs.txt",
                                  anchors=ROOT / "examples/anchors.csv", context="HepG2_demo")
adapted.save(ROOT / "results/demo_adapt")
print(f"Saved {adapted.predictions.shape} GRAFT-Adapt matrix to results/demo_adapt")
