# Evaluation Report

Test set: reconciled `test.csv`; excluded raw test rows were not evaluated because their labels were outside the train-derived production class map.

| Metric | Value |
|---|---:|
| top_1_accuracy | 0.216700 |
| top_3_accuracy | 0.359841 |
| top_5_accuracy | 0.430417 |
| macro_precision | 0.067411 |
| macro_recall | 0.151364 |
| macro_f1 | 0.069680 |
| weighted_f1 | 0.184110 |
| balanced_accuracy | 0.151364 |
| expected_calibration_error_10_bins | 0.057924 |

Auxiliary mainclass/subclass metrics are reported only when the selected checkpoint contains hierarchical heads.
The calibration value is expected calibration error over 10 confidence bins.
