# Reduced-Budget Ablation

Subset rows: 1248 (36.652% of reconciled train)
Two epochs per configuration; model selection used validation macro-F1 only.

| Model | Val macro-F1 | Val accuracy | Final val loss | Seconds |
|---|---:|---:|---:|---:|
| C_cross_attention | 0.023281 | 0.287375 | 1.508289 | 1018.34 |
| D_cross_attention_hierarchical | 0.014874 | 0.269103 | 2.309438 | 998.01 |
| B_concat_metadata | 0.014439 | 0.252492 | 1.605831 | 815.69 |
| E_concat_hierarchical | 0.009882 | 0.234219 | 2.371571 | 719.92 |
| A_image_only | 0.004881 | 0.121262 | 1.740138 | 823.64 |

Winner: **C_cross_attention** with validation macro-F1 0.023281.
