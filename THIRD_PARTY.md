# Resource provenance and release terms

GRAFT uses trained GRAFT models together with frozen identity representations and source-response references. The terms for code, model weights and data-derived files must be recorded separately. No separate licence grant is made for GRAFT code, model weights or prepared resources in this release; third-party terms remain applicable.

## GRAFT models and adapter

`resources/graft_seed*.npz` contains GRAFT backbone parameters. `resources/adapter.npz` contains the compatible learned adapter. Their training sources, architecture and provenance belong in `resources/manifest.json`.

**Distribution:** GRAFT weights and the compatible adapter are included in the repository. Their provenance and SHA-256 checksums are recorded in the bundle manifest.

## UniPert representations

Frozen 256-dimensional UniPert representations provide the genetic and drug identities used by GRAFT. They are supplied as model resources for the inference workflow; users do not need to run the upstream sequence or molecular encoders for supported IDs.

Author sources:

- [UniPert usage repository](https://github.com/TencentAILabHealthcare/UniPert)
- [UniPert source license](https://github.com/TencentAILabHealthcare/UniPert/blob/main/LICENCE)
- [UniPert–G2CP reproduction repository](https://github.com/lynn-1998/UniPert-G2CP_reproduce)

The independent UniPert encoder repository identifies its source-code license as GPL-3.0. This must not be treated as a complete license statement for every weight, embedding or dataset distributed by the authors, nor inferred from a different reproduction repository's license.

**Attribution:** the bundled identity tables are derived from UniPert. Refer to the linked author sources for the original project and terms; no GRAFT code licence is applied to these tables.

## LINCS source responses and annotations

GRAFT's source-response references were derived from LINCS L1000 signatures for the five source cell contexts. The prepared references, gene panel, catalog annotations and any example responses should be traceable to the source data and processing records in the bundle manifest.

Authoritative data source: [CLUE / LINCS2020](https://clue.io/data/CMap2020#LINCS2020).

**Attribution:** the bundled response references and example inputs are prepared derivatives of LINCS L1000 data. Refer to CLUE/LINCS for the source collection and its terms.

## Recording a release

The bundle manifest records resource filenames, source checkpoints, source collections and SHA-256 checksums. Preserve this provenance and the linked upstream notices when using the supplied resources.

The repository is intended for GRAFT prediction with supplied resources. It does not redistribute the reference papers or their full analysis repositories.
