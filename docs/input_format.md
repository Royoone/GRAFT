# Input format

GRAFT operates on prepared, signed transcriptional responses. It predicts one mean 976-gene response profile per query drug in one target context.

## Genetic support CSV

Use a UTF-8 CSV with this structure:

| Column | Meaning |
| --- | --- |
| `perturbation_id` | Genetic perturbation identifier supported by the bundle |
| Each of the 976 model gene names | Mean response of that gene to the perturbation in the target context |

The first row contains column names. Each subsequent row describes one perturbation. Use `examples/genetic_responses.csv` as the format template, retaining its gene header when replacing the response rows with your data.

Requirements:

- All rows must come from the same target context.
- Use one row per genetic perturbation ID; combine replicate measurements before supplying the CSV.
- Provide exactly the 976 model genes with unique column names and finite numeric values. Gene columns are matched by name and reordered to the model panel. Missing and extra gene columns are rejected, so remove unrelated metadata columns before prediction.
- Use perturbation IDs present in the bundle's genetic identity table. IDs must be nonempty, unique and free of surrounding whitespace. A gene with no supported identity cannot be replaced by a zero or random embedding.
- Keep responses on the prepared response scale. Do not divide by the model scale, normalize each row to unit length or standardize each target sample before prediction.

Supported genetic IDs are stored in `reference.npz` as `genetic_ids`; the ordered model panel is stored as `genes`. You can inspect them with:

```python
import numpy as np

with np.load("resources/reference.npz", allow_pickle=False) as reference:
    print(reference["genetic_ids"])
    print(reference["genes"])
```

The number of support rows is not fixed. Their order does not determine the prediction. A single genetic support condition produces a zero cross-moment state after centering, so its D0 prediction equals the source same-drug mean. More support conditions do not by themselves guarantee better predictions.

### Preparing responses from expression data

The CSV must contain mean perturbation responses, rather than cell-level raw counts or untreated expression. Response preparation depends on the measurement platform and experiment. Record your preprocessing, control group, perturbation identities and target context before constructing the input.

For example, the manuscript's iPSC analysis normalized full-transcriptome counts to 10,000 UMIs per cell, applied `log1p`, and subtracted the same-line NonTarget control mean from each genetic perturbation mean. Counts were merged by gene symbol before normalization. The full-transcriptome total was used for normalization before selecting the 976-gene model panel. That dataset's model-input filters required at least 20 targeted cells, two guides and 50 same-line controls. These are the manuscript's iPSC preparation choices, not a claim that one preprocessing recipe has been validated for every platform.

If your experiment does not measure some model genes, do not silently treat them as measured zeros. The default input contract requires the complete panel. GRAFT cannot reconstruct unmeasured genes from a CSV header.

## Query drug list

Pass a UTF-8 text file to `--drugs`, with one supported drug ID per line:

```text
<drug_id_from_catalog>
<another_drug_id_from_catalog>
```

Use the `drug_id` values in `resources/catalog.tsv`, rather than free-text drug names, SMILES strings or new identifiers. The packaged source reference and frozen UniPert representation must both be available for each query. Blank lines are ignored; duplicate IDs and an empty query list are rejected. Unknown IDs are reported as errors.

Queries can be unmeasured in the target context. They must still belong to the source drug catalog: this workflow does not establish predictions for drugs absent from all source contexts.

The catalog is organized at the drug-identifier level. Distinct records are not automatically merged into a single chemical entity merely because they share an InChIKey. Retain the returned IDs when joining predictions to annotations.

## Optional anchor CSV

Pass an anchor CSV to `--anchors` to use GRAFT-Adapt:

| Column | Meaning |
| --- | --- |
| `drug_id` | Supported drug ID for a measured anchor |
| Each of the 976 model gene names | Measured mean drug response in the same target context |

Use one row per anchor drug, complete unique gene columns and finite values. Anchor and query drug IDs must not overlap. Anchors must represent measured transcriptional responses, rather than mechanism-of-action labels, binding activities or source-context responses.

Prepare anchors in the response definition and scale appropriate to the model output. Do not divide them by an internal model scale or normalize each anchor profile independently. Genetic support responses and drug anchors can originate from different assays, but their response definitions and scales must be documented; arbitrary rescaling is not corrected automatically.

GRAFT-Adapt uses the supplied anchors to solve a ridge-regression correction using frozen features and learned adapter parameters. It does not retrain the neural networks on your context. With no anchors, predictions follow D0. The number of anchor rows is not fixed to ten or twenty.

## Model bundle

`--bundle` points to a directory containing:

```text
resources/
├── manifest.json
├── reference.npz
├── graft_seed*.npz
├── adapter.npz
└── catalog.tsv
```

The resources contain the fixed gene order, identity lookups, source references, source-derived scales, trained model weights and compatible adapter. Use a complete matching bundle. A different gene panel, arbitrary 256-dimensional embeddings or a reference from another source split cannot be substituted for these resources.

The default source contexts are A375, A549, HT29, MCF7 and PC3. Inference uses the supplied model bundle; users do not need the original full LINCS dataset. The exact model files and provenance are recorded in `manifest.json`.

This bundle is the five-source model for prediction in other target contexts. Evaluating a source context as held out requires its corresponding source-excluded model and references from the paper-reproduction workflow. Changing `--context` only changes an output label; it does not remove training data or rebuild a held-out fold.

## Command options

```bash
graft predict \
  --bundle resources \
  --genetic my_genetic_responses.csv \
  --drugs my_query_drugs.txt \
  --context my_context \
  --output results/my_context
```

| Option | Purpose |
| --- | --- |
| `--bundle` | Model-resource directory |
| `--genetic` | Genetic support CSV |
| `--drugs` | Query drug ID text file |
| `--anchors` | Optional measured drug-anchor CSV |
| `--context` | Label written to the results |
| `--output` | Output directory |
| `--csv` | Also save the full prediction matrix as `predictions.csv` |
| `--device` | Torch device; defaults to `cpu` |

## Outputs

`predictions.npz` contains these arrays:

| Array | Shape / meaning |
| --- | --- |
| `predictions` | `N_query × 976` signed predicted responses |
| `drug_ids` | `N_query` drug IDs, in query-file order |
| `genes` | `976` gene names, in the model's output order |
| `source_mean` | `N_query × 976` source same-drug mean responses |
| `context` | Scalar target-context label |
| `mode` | Scalar `D0` or `GRAFT-Adapt` label |

Gene profiles are returned on the original prepared response scale, after the model reverses its internal scaling. The context-dependent correction can be obtained as `predictions - source_mean`.

`summary.tsv` has two columns: `drug_id` and `response_rms`. `run.json` records the mode, support/anchor/query counts and IDs, model seeds and source contexts, bundle version and manifest checksum, input-file checksums, device and runtime versions. With `--csv`, `predictions.csv` contains `drug_id` followed by the 976 gene columns.

The supplied three-seed bundle averages predictions at the gene-profile level. The reported RMS is then calculated from the averaged profile, as the square root of the mean squared response across the 976 genes. It describes response magnitude and does not establish drug efficacy, toxicity or a calibrated uncertainty interval.
