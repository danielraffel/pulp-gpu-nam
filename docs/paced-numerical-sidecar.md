# Paced numerical sidecar

`gpu-nam-stamped-validation --paced` reports `sidecar_schema=pulp.gpu_nam.paced.v2`. Existing CSV columns remain in order; new columns are appended. The sidecar now attributes numerical failures to each callback's selected output, including priming and CPU fallback.

Each row records mismatch count, nonfinite mismatch count, maximum absolute error among finite comparisons, and the first mismatch's zero-based channel/frame, actual value, expected value, and finite/nonfinite classification. First-mismatch fields are empty when no mismatch occurred. NaN and infinity remain explicit; a zero finite maximum does not mean a row containing nonfinite values passed. The existing 1e-4 absolute tolerance is unchanged.

The independent CPU oracle fills these fields after the measured callback loop and worker retirement. No comparison, serialization, or additional allocation occurs in the callback. Row storage is allocated before timing. Aggregate mismatch count and finite maximum still determine the existing diagnostic verdict.

`delivered_input_sequence` remains the expected input index derived from callback index minus pipeline lead. It is not an observed worker source sequence. The log records `input_sequence_provenance=callback_minus_lead_not_worker`. These fields can locate a bad worker-selected or fallback-selected callback, but cannot alone prove stale worker acceptance or state-history corruption. A separate worker provenance experiment would be needed for that attribution.

Old campaign CSVs are preserved unchanged and do not contain these fields. New diagnostics require a rebuilt, distinctly identified executable; no old result should be retroactively filled in.
