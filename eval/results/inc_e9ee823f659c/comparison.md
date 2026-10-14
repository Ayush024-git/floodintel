# Flood polygon agreement

Incident: inc_e9ee823f659c

Reference acquisition date: not available
Reference metadata date stamp (not acquisition): 2026-08-29
Our post-image acquisition date: not available
Absolute date gap in days: not available

Grid: 3032 rows × 3048 columns; EPSG:32645.
Grid transform: [10.0, 0.0, 302500.0, 0.0, -10.0, 3105240.0]
Pixel area: 100.0 m²; reference polygons used: 2.
Valid pixels: 9241536; ignored pixels: 0.
TP: 124; FP: 2839; FN: 58765; TN: 9179808.

IoU: 0.002008812856402281
Precision: 0.041849476881538984
Recall: 0.0021056564044218784
F1: 0.0040095712345599175

Our area: 0.2963 km²; reference area: 5.8889 km²; overlap: 0.0124 km².

The numbers above retain the legacy full-valid-grid flood-only domain.
Analysed polygon domain: 18.483406755710487 km²; valid analysed grid: 18.4842 km².
Share of our valid flood area outside the analysed domain: 0.958150523118461.
Declared pixel-centre tolerance: 50.0 m (not fitted to this reference).

All variants below use the same analysed, valid domain. Debris is an unconfirmed candidate, not a water classification.

| Variant | TP | FP | FN | TN | IoU | Precision | Recall | F1 | Ours km² | Reference km² | Overlap km² | Precision tol | Recall tol |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flood_mask | 124 | 0 | 58765 | 125953 | 0.0021056564044218784 | 1.0 | 0.0021056564044218784 | 0.004202463863894396 | 0.0124 | 5.8889 | 0.0124 | 1.0 | 0.007964135916724685 |
| debris_candidate | 24234 | 1534 | 34655 | 124419 | 0.4010724393029145 | 0.9404687985097796 | 0.41151997826419195 | 0.5725220596052305 | 2.5768 | 5.8889 | 2.4234 | 0.9739211425023284 | 0.8790096622459203 |
| union | 24358 | 1534 | 34531 | 124419 | 0.40312463796898534 | 0.9407539008187857 | 0.41362563466861385 | 0.5746098772130548 | 2.5892 | 5.8889 | 2.4358 | 0.9740460373860652 | 0.8803851313488088 |

Areas use valid grid cells only. Zero metric denominators return zero.
Overlay: ours blue, reference orange, overlap white; neither dark, ignored pixels transparent.
Pixel agreement with this reference is not a claim of model accuracy. Variant comparisons exclude pixels outside the supplied analysed domain. Legacy top-level numbers retain the full valid grid; survey completeness is not established.

Warnings:
- Reference image acquisition date is absent or ambiguous in the XML; its metadata creation/date stamp is not used as acquisition.
- Post-image acquisition date is not recorded in flood_meta.json; the acquisition-date gap cannot be checked.
