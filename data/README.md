# Derived data

Aggregate tables behind every figure and table of the article. Each row summarises seeds, label
draws or comparison cells; no row describes a patient or a scan. `columns.csv` describes every
column of every file.

| File | Article item | Written by |
|---|---|---|
| `table1_cohorts.csv` | Table 1 (cohort characteristics) | nb9 |
| `table1_conditioning.csv` | Table 2, Figs 1-2 | nb7 (from nb6 E0) |
| `table2_placement_and_arms.csv` | Fig 3b, Section 4.3 | nb7 |
| `stats_B_placement.csv` | Table 3 (Family B) | nb7 |
| `stats_A_quantum_vs_classical.csv` | Table 4 (Family A), Fig 3a | nb7 |
| `stats_A_equivalence.csv` | Online Resource 2 (TOST) | nb7 |
| `table5_transfer_matrix.csv` | Table 5 | nb5 Part C |
| `stats_C_india_label_budget.csv` | Table 6 (Family C), Figs 4-5 | nb7 (from nb6 E6) |
| `stats_C_sign_test.csv` | Section 4.7, Online Resource 2 | nb7 |
| `table3_label_efficiency.csv` | Figs 4-5 | nb7 (from nb6 E6) |
| `table4_transfer.csv` | Fig 6 | nb7 (from nb6 E7) |
| `sweep_lr_wd.csv` | Table 7 | nb8 |
| `sweep_qubits.csv` | Table 8 | nb8 |
| `nisq_ablation.csv` | Online Resource 1 | nb2 |

In `stats_C_india_label_budget.csv` the `cell` labels are truncated by nb7: `abnormal_` is the
any-finding endpoint and `hemorrhag` is hemorrhage. In the sweep files, `key` is a run
configuration string and contains no patient information.

