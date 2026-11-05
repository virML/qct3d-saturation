# qct3d-saturation

Code and derived data for

> **Quantum or just quantised? Angle-encoder saturation confounds ansatz and placement comparisons in hybrid models for 3D CT.**
> Virinchi Puvvada, Maneesh Meka, Dhanalaxmi Dhanpal. *Quantum Machine Intelligence* (submitted, 2026).

Hybrid quantum–classical adapters on frozen 3D CT features are usually compared by ansatz and by where the circuit is attached. This repository shows that both comparisons can be dominated by a classical component: the learned projection that maps features to rotation angles through a bounded activation. It contains everything needed to rerun the study, plus the aggregate tables behind every figure and table in the article.

**No patient-level data are included.** The development cohort is the public RSNA 2019 Intracranial Hemorrhage Detection dataset. The external cohort is protected health information. It is available only under the institutional data-sharing agreement (see *Data availability*).

---

## Contents

```
notebooks/        the analysis, as Google Colab notebooks (outputs cleared; see "Privacy")
src/qct3d/        code shared by several notebooks, moved out of them verbatim
  config.py         reads every path from config.yaml
  preprocessing.py  resample -> crop/pad -> HU windowing (identical for both cohorts)
  models.py         circuit families Q1-Q4, adapters, parameter-matched controls, training, metrics
  protocol.py       patient-grouped splits, the three arms, paired comparisons, noise model
data/derived/     aggregate result tables behind every figure and table (no identifiers)
data/columns.csv  description of every column in data/derived/
scripts/          make_figures.py (Figs 1-6) and build_online_resources.py (Online Resources 1-3)
tests/            unit tests, checks of the released tables against the article, release checks
tools/            check_release.py: fails if anything identifying or private is present
config.example.yaml   template for config.yaml (paths)
```

## Notebooks and what they produce

The notebooks keep their original numbers because their comments cross-reference one another by those numbers (for example "nb5 cell 7"). Run them in the order below. Each one resumes after an interruption: rerun it and it continues from the log it writes.

| Run | Notebook | What it does | Writes | Used for |
|---|---|---|---|---|
| 1 | `nb1_rsna_feature_extraction` | Reads the RSNA archive, assembles 3D volumes, preprocesses them and extracts frozen features with backbones B1–B3 (GPU) | `features_B*.parquet`, spatial maps, preprocessing signature | inputs to everything else |
| 2 | `nb2_rsna_adapter_grid` | The original, unconditioned adapter grid on RSNA, plus the depolarising-noise ablation | `phaseb_results.jsonl`, `nisq_ablation.csv` | the run in which the saturation was first found (§3.4), Online Resource 1 |
| 3 | `nb5_external_cohort_extraction` | Builds the external-cohort manifest and labels from the radiology reports, then extracts features with the same preprocessing and backbones (GPU). Its Part C computes the cross-cohort transfer matrix | `matched_table.parquet`, external `features_B*.parquet`, Part C tables | Table 5, inputs to nb6 and nb9 |
| 4 | `nb6_main_experiments` | E0: saturation audit and encoder conditioning, placement, quantum vs classical (5 seeds). E6: external-cohort label efficiency (50 draws per budget). E7: transfer from RSNA | `overnight_results.jsonl` | input to nb7 |
| 5 | `nb8_optimiser_and_width_sweeps` | Learning-rate × weight-decay sweep; qubit-width sweep with the sign-code diagnostic | `sweep_lr_wd.csv`, `sweep_qubits.csv` | Tables 7 and 8 |
| 6 | `nb7_figures_and_statistics` | Reads nb6's log, writes the result tables and runs the formal statistics (paired tests, Benjamini–Hochberg, TOST equivalence, sign test) (CPU) | the CSVs in `data/derived/` | Tables 2–4 and 6, Figs 1–6 data, Online Resource 2 |
| 7 | `nb9_table1_cohorts` | Cohort characteristics; prints aggregates only and checks itself against the article's counts (CPU) | `table1_cohorts.csv` | Table 1 |

After that, `scripts/make_figures.py` draws the publication figures and `scripts/build_online_resources.py` builds the Online Resources. Both read only `data/derived/`.

| Article item | Source table in `data/derived/` | Produced by |
|---|---|---|
| Table 1 | `table1_cohorts.csv` | nb9 |
| Table 2, Figs 1–2 | `table1_conditioning.csv` | nb6 E0 → nb7 |
| Table 3, Fig 3b | `stats_B_placement.csv`, `table2_placement_and_arms.csv` | nb6 E0 → nb7 |
| Table 4, Fig 3a | `stats_A_quantum_vs_classical.csv` | nb6 E0 → nb7 |
| Table 5 | `table5_transfer_matrix.csv` | nb5 Part C (cells C2–C3) |
| Table 6, Figs 4–5 | `stats_C_india_label_budget.csv`, `table3_label_efficiency.csv` | nb6 E6 → nb7 |
| Fig 6 | `table4_transfer.csv` | nb6 E7 → nb7 |
| Table 7 | `sweep_lr_wd.csv` | nb8 |
| Table 8 | `sweep_qubits.csv` | nb8 |
| Online Resource 1 | `nisq_ablation.csv` | nb2 |
| Online Resource 2 | `stats_A_equivalence.csv`, `stats_C_sign_test.csv` | nb7 |
| Online Resource 3 | all of the above | `scripts/build_online_resources.py` |

The CSV file names are the ones nb7 writes. They predate the article's final table numbering, which is why `table1_conditioning.csv` feeds Table 2.

Not included: two exploratory notebooks from the same project (a 2D-versus-3D comparison and an attribution analysis) that the article does not report. `nb5` also contains cells from an earlier pass with the Pillar-0 head-CT model (classifier sweep and leaderboard). The article does not report them; they are left in place because that pass also wrote the cohort manifest that nb6 and nb9 read.

## Running on Colab

1. Clone the repository into the Colab runtime, using the URL shown under the green **Code** button on this repository's GitHub page:
   `!git clone <repository URL> /content/qct3d-saturation`
2. Copy `config.example.yaml` to `config.yaml` in the repository root. Edit the paths there; every notebook reads them from this one file. `config.yaml` is git-ignored.
3. Open a notebook from `/content/qct3d-saturation/notebooks/` and run all. Cell 0 finds the repository and loads `config.yaml`; the next cell mounts Google Drive.
4. nb1 and nb5 need a GPU (a T4 is enough). nb5 also needs a Hugging Face token in Colab Secrets (`HF_TOKEN`), because its first pass downloads a gated checkpoint. The other notebooks run on CPU, more slowly.

To run outside Colab, install `requirements.txt`, set `QCT3D_CONFIG` to your config file (or place `config.yaml` in the repository root), and point `drive_root` at the folder that holds the data.

## Reproducing the figures and Online Resources from the released tables

```bash
pip install -r requirements-figures.txt
python scripts/make_figures.py data/derived figures/
python scripts/build_online_resources.py data/derived online_resources/
```

With the versions in `requirements-figures.txt`, the six 600-dpi PNG figures come out byte-identical to the submitted artwork; the EPS and PDF files differ only in their embedded creation dates. The resource builder checks every number it prints against the article and stops if one disagrees.

## Tests and checks

```bash
pip install -r requirements.txt
pytest                          # unit tests + checks of data/derived against the article
python tools/check_release.py   # must pass before every commit and release
```

What the tests cover:

- **Circuits:** every circuit family trains and gradients reach the rotation angles.
- **Fair comparison:** the classical control is parameter-matched, and encoder conditioning adds no parameters.
- **Splits and determinism:** splits are patient-grouped, and runs are deterministic.
- **Preprocessing:** its self-test passes.
- **Released tables:** they reproduce the article's counts. For example: 27 of 36 Family C comparisons survive correction; 17 of 18 any-finding and 10 of 18 hemorrhage comparisons survive; 33 of 36 favour the classical arm; 9 of 12 Family A cells are equivalent at ±0.02. The Benjamini–Hochberg q-values also recompute exactly from the p-values.

**Refactoring did not change the computation.** Shared notebook cells were moved into `src/qct3d/` unchanged, and the notebooks import them. Before release, the original notebook cells and the released notebooks were run side by side on synthetic data. They produced bitwise-identical outputs, covering:

- model and protocol code: the parameter-budget table, the training self-test, and the predictions of all three arms for three circuit families;
- encoder conditioning;
- patient-grouped splits;
- preprocessing: a full volume, run through both nb1's and nb5's copies.

## Privacy

The external cohort is protected health information. This repository is built so that none of it can be committed.

- **Notebook outputs and metadata are removed.** Outputs can show identifiers, and Colab stores the user's name and account id in cell metadata.
- **Do not commit an executed notebook.** `tools/check_release.py` rejects any notebook with outputs.
- **Patient-level files are never released.** This covers feature tables, embeddings, manifests, result logs with per-series rows, DICOM, and anything carrying a name, PatientID, UID, accession number or date. `.gitignore` excludes the common formats, and the checker rejects them.
- **Identifying passages were rewritten.** A few comments in the original notebooks quoted individual patients to explain a design decision, and nb5 embedded a study-to-report map. The comments now describe the same reasoning without identifiers. The map is read from a private path (`external_part1_map_csv`).
- **The released tables are aggregates only.** `data/derived/` holds summary statistics over seeds, draws or cells.

## Data availability

- **Development cohort:** RSNA 2019 Intracranial Hemorrhage Detection, available from the challenge organisers under their terms.
- **External cohort:** routine head CT studies from A C Subba Reddy Government Medical College & Hospital, Nellore, India, used under the approval of the Institutional Ethics Committee of that institution (EC reference ECR/961/Inst/AP/2017/RR-25/99-13/01-10-26). It contains protected health information and cannot be released. Derived image features without identifiers and the per-cell result log are available from the corresponding author on reasonable request, subject to approval by the Institutional Ethics Committee and a data-sharing agreement.

## Citation

See `CITATION.cff`. Please cite the article and the archived release of this repository on Zenodo.

## Licence

MIT (see `LICENSE`). The pretrained backbones (SuPreM, MedicalNet) and datasets keep their own licences.
