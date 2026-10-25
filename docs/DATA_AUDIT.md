# Data audit

All checks below use local files only; no model output or external resource was used.

## Outcome

**PASS.** The prepared English-Turkish data, fixed partitions, and manifest are consistent. No tested form of cross-partition leakage was found.

## Fixed settings

- Split seed: 42
- Target ratios: 70% train, 20% development, 10% test
- Fixed label order: anger, anticipation, disgust, fear, joy, sadness, surprise, trust
- Label source: projected reference labels reconstructed from English source annotation metadata
- Zero-division policy for later metrics: 0.0

## Dataset and partition counts

| Partition | Rows | Observed ratio | Leakage groups | Single-label | Multi-label |
|---|---:|---:|---:|---:|---:|
| train | 4,852 | 70.0043% | 3,154 | 3,244 | 1,608 |
| dev | 1,386 | 19.9971% | 502 | 893 | 493 |
| test | 693 | 9.9986% | 172 | 466 | 227 |
| Total | 6,931 | 100.0000% | 3,828 | 4,603 | 2,328 |

Multi-label targets are present in every partition and remain JSON lists rather than being collapsed to one class.

## Emotion counts

| Emotion | Prepared/aligned | Train | Development | Test |
|---|---:|---:|---:|---:|
| anger | 1,775 | 1,243 | 355 | 177 |
| anticipation | 1,654 | 1,157 | 331 | 166 |
| disgust | 852 | 596 | 170 | 86 |
| fear | 894 | 626 | 179 | 89 |
| joy | 1,278 | 894 | 256 | 128 |
| sadness | 1,001 | 700 | 200 | 101 |
| surprise | 1,095 | 767 | 219 | 109 |
| trust | 1,238 | 867 | 248 | 123 |

## Leakage and consistency checks

| Check | Observed | Status |
|---|---:|---|
| Cross-partition `alignment_id` values | 0 | PASS |
| Cross-partition `leakage_group_id` values | 0 | PASS |
| Cross-partition `english_text` values | 0 | PASS |
| Cross-partition `turkish_text` values | 0 | PASS |
| Cross-partition `bilingual_text_pair` values | 0 | PASS |
| Cross-partition `source_segment` values | 0 | PASS |
| Manifest records | 6,931 | PASS |
| Manifest exactly matches split assignment | yes | PASS |
| Split union exactly covers all prepared rows | yes | PASS |
| Raw English-Turkish pair rows matched | 6,931 | PASS |
| Source-metadata labels reconstructed | 6,931 | PASS |
| Missing English texts | 0 | PASS |
| Missing Turkish texts | 0 | PASS |
| Missing labels | 0 | PASS |
| Labels outside permitted inventory | 0 | PASS |
| Malformed metadata rows ignored by the existing preparation rule | 24 | PASS |

The pairing check rebuilds every alignment ID from the English and Turkish document and segment fields. It checks all six raw pair fields against `pairs-tr.txt`, then checks each row's label names and numeric IDs against the source annotation metadata.

## File hashes

| File | SHA-256 |
|---|---|
| `data/processed/xed_en_tr_aligned.tsv` | `4555338e8b81981491a9e6c28f6a9cd9a914db411c43dc337a132c866cd6abf1` |
| `data/splits/train.tsv` | `0cb65716e900124d5ab384438eb80a39c5aa782f9674e723a9f35a1868d2ad85` |
| `data/splits/dev.tsv` | `6bbc4d92b62b20c7508de474b38fd3cae5fabd83e2f5d7240a34324efad00318` |
| `data/splits/test.tsv` | `2373682101c9b712f98d047dde88eb58db81260d02f3c33754807b5a44b53326` |
| `data/splits/split_manifest.tsv` | `91fb0b7844850f80b44773220f8ac77d89b7c20c8b9a91f1a18604e07233729f` |
| `configs/data_split.json` | `f80c6f8216ab6b02b589adb343ec0eb6eb14343f03a2f60eb49cc7c2a8a1eb4e` |

## Interpretation limitation

The Turkish targets are aligned subtitle counterparts carrying projected reference labels reconstructed through their English source annotations. They are not independently human-annotated Turkish emotion labels. The split prevents exact/source-identity leakage, but it does not establish that every subtitle alignment or projected label is semantically perfect.
