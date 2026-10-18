# HepG2 example

These files demonstrate prediction for one target context using prepared HepG2 LINCS L1000 response data:

| File | Contents |
| --- | --- |
| `genetic_responses.csv` | 64 genetic perturbation mean-response profiles, with `perturbation_id` and 976 gene columns |
| `query_drugs.txt` | Five catalog drug IDs, one per line |
| `anchors.csv` | Five measured drug-response profiles, with `drug_id` and the same 976 gene columns |
| `quickstart.py` | Runs D0 and GRAFT-Adapt on CPU using the supplied bundle |
| `expected_summary.tsv` | Reference response RMS values from the supplied example, for checking a local run |
| `provenance.json` | Example selection rule and source-file checksums |

The example uses a small deterministic subset of publicly available HepG2 LINCS L1000 responses, aligned to the model panel and supported catalog IDs. It selects the first 64 finite, supported genetic IDs in lexical order, then the first ten finite, supported drug IDs in lexical order, assigning five to anchors and five to queries. Anchor and query drugs are disjoint. Selection details and source checksums are recorded in [provenance.json](provenance.json). The selection is for a short usage demonstration and does not recreate the paper's query panels, anchor lists or reported benchmark results.

From the project root, after installation:

```bash
python examples/quickstart.py
```

This writes D0 results to `results/demo` and GRAFT-Adapt results to `results/demo_adapt`. Each contains five predicted drug profiles across 976 genes, `summary.tsv` and `run.json`. Equivalent command-line examples are in the main [README](../README.md).

The measured responses in `anchors.csv` are used only for the adaptation example. The query list is used by both workflows. Replacing these files with your own data requires the response definitions and complete gene panel described in [Input format](../docs/input_format.md).

`expected_summary.tsv` was generated on CPU with Python 3.12, NumPy 2.3.5 and PyTorch 2.5.1. Small floating-point differences across platforms are expected. Its RMS columns summarize predicted transcriptional responses and are not experimental accuracy scores.
