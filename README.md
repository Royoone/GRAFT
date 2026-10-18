# GRAFT

GRAFT predicts mean drug-induced transcriptional responses in a new cellular context from a set of measured genetic perturbation responses in that context. Predictions cover a fixed panel of 976 genes and drugs represented in the supplied source catalog.

The standard **D0** workflow uses no drug-response measurements from the target context. If you have measured responses to a separate set of drugs in the same context, **GRAFT-Adapt** uses these drug anchors to calibrate predictions while keeping the neural-network parameters fixed.

## Install

Use Python 3.10 or later. Clone the repository and enter its directory:

```bash
git clone https://github.com/Royoone/GRAFT.git
cd GRAFT
```

Install the package:

```bash
python -m pip install -e .
```

Keep the supplied model bundle in `resources/`. The bundle contains the trained models, source-response references, fixed gene panel, frozen UniPert representations and drug catalog. The supplied bundle includes three trained model seeds, 4,994 genetic perturbation IDs and 7,856 drug IDs. These resources must be used together. Prediction runs on CPU by default.

## Run the example

```bash
graft predict \
  --bundle resources \
  --genetic examples/genetic_responses.csv \
  --drugs examples/query_drugs.txt \
  --context HepG2_demo \
  --output results/demo
```

The HepG2 example contains 64 genetic support conditions and five query drugs. To also use the five measured drug anchors:

```bash
graft predict \
  --bundle resources \
  --genetic examples/genetic_responses.csv \
  --drugs examples/query_drugs.txt \
  --anchors examples/anchors.csv \
  --context HepG2_demo \
  --output results/demo_adapt
```

Both commands produce five 976-gene profiles. This is a usage demonstration, not a reproduction of the paper’s benchmark metrics. See the [example notes](examples/README.md) for its data selection. You can also run both workflows through the Python example:

```bash
python examples/quickstart.py
```

Results are written to the output directory:

| File | Contents |
| --- | --- |
| `predictions.npz` | Predicted signed responses and their drug and gene identifiers; one response profile per query drug |
| `summary.tsv` | One row per drug, with `drug_id` and response root mean square (`response_rms`) |
| `run.json` | Input, model-bundle and run metadata |

Add `--csv` to also write the full prediction matrix as `predictions.csv`. RMS summarizes the magnitude of a predicted transcriptional response; it is not a probability or a measure of clinical benefit.

## Use your own data

Prepare a CSV of **mean genetic perturbation responses**, not raw single-cell counts. The first column must be `perturbation_id`; the remaining columns are the 976 gene names used by the model. Each row is one genetic perturbation in a single target context. Gene names, perturbation IDs and response definitions must match the supplied resources.

Create a text file with one query drug ID per line. Use IDs listed in `resources/catalog.tsv`. Replace the example input paths and give your context a descriptive name:

```bash
graft predict \
  --bundle resources \
  --genetic my_genetic_responses.csv \
  --drugs my_query_drugs.txt \
  --context my_context \
  --output results/my_context
```

The context name labels the output; it is not an input expression profile or a learned cell-type code. See [Input format](docs/input_format.md) for alignment, preprocessing and output details.

## Optional drug anchors

An anchor is a measured drug-response profile from the same target context. Prepare an anchor CSV with `drug_id` as the first column and the same 976 gene columns, then add `--anchors`:

```bash
graft predict \
  --bundle resources \
  --genetic my_genetic_responses.csv \
  --drugs my_query_drugs.txt \
  --anchors my_anchors.csv \
  --context my_context \
  --output results/my_context_adapt
```

Anchor drugs and query drugs must be disjoint. Supply responses on their original prepared response scale: GRAFT handles its own internal scaling. Without anchors, the command runs D0.

## What the model returns

GRAFT returns context-level mean responses for drugs measured in its source catalog. It does not directly accept a new SMILES string, an arbitrary protein embedding or untreated expression alone. With fewer than two genetic support conditions, the cross-moment state is zero and D0 returns the source same-drug mean.

The project contains the usage workflow. The paper benchmarking and selected figure-reproduction pipeline are available in [GRAFT-reproducibility](https://github.com/Royoone/GRAFT-reproducibility). Model-resource provenance and release terms are described in [THIRD_PARTY.md](THIRD_PARTY.md) and the bundle manifest.

The supplied models were trained on A375, A549, HT29, MCF7 and PC3 for use in other target contexts. They are not the separate held-out folds used to benchmark those five source contexts in the paper.

## Versioned release

The manuscript-associated application code and supplied inference resources are included in [v0.1.0](https://github.com/Royoone/GRAFT/releases/tag/v0.1.0).
