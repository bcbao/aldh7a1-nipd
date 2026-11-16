# ALDH7A1 NIPD by Relative Haplotype Dosage (RHDO)

Analysis and figure-generation code accompanying the manuscript:

> **Non-invasive prenatal diagnosis in recurrence-risk pregnancies of families with a previously affected child: a proof-of-concept study of genotype-guided perinatal care in *ALDH7A1*-associated pyridoxine-dependent epilepsy.**
> Truong N-S\*, Tran K-T\*, Vu QB, Bui TTT, Nguyen QA, Nguyen TM, Bui C-B. (\*equal contribution)

## Important note on data

**No patient data are included in this repository.** Raw sequencing data, parental genotypes and pregnancy-level results are confidential (IRB-approved, consent-restricted) and are available from the corresponding author on request (see the manuscript's Data availability statement).

By default, the script generates a **simulated cohort** whose summary statistics match the ranges reported in the manuscript (fetal fraction, informative SNP counts, dosage values, genotype distribution). Figures produced from simulated data are for demonstration and reproducibility of the pipeline only, **not** the published results.

## What the code does

| Component | Description |
|---|---|
| Fetal fraction | `FF = 2ΣBᵢ / Σ(Aᵢ + Bᵢ)` at SNPs where mother and father are homozygous for different alleles |
| Imputation filter | Retain imputed parental genotypes with Rsq ≥ 0.82 and genotype posterior ≥ 0.90 |
| Maternal RHDO | Binomial likelihood over informative SNPs (type α / β), equal priors, posterior probability of risk-haplotype inheritance |
| Recombination scan | No-switch vs. single-switch models, bidirectional, maximum log-likelihood ratio |
| Paternal transmission | Paternal-specific allele fraction (≈ FF/2) vs. background error rate |
| Genotype integration | Autosomal recessive model → affected / carrier / unaffected / inconclusive |
| Reportability gate | FF ≥ 4%, ≥ 25 informative SNPs, posterior confidence ≥ 0.98, phasing ≥ 0.95, depth ≥ 30× |
| Statistics | Sensitivity, specificity, PPV, concordance, first-pass reportability with exact (Clopper–Pearson) 95% CIs |
| Dilution experiment | Simulated FF dilution series (0.5–15%) illustrating the 4% threshold logic |
| Figures | Figure 2A–E, Figure 3A–D, Figure 4A–B (Sankey diagrams drawn with matplotlib) |

SHAPEIT4 / Minimac4 imputation (VN1K panel) is **not** run by this script; only the downstream quality filter is implemented.

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/<your-username>/aldh7a1-nipd-rhdo.git
cd aldh7a1-nipd-rhdo
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run with the simulated demo cohort:

```bash
python aldh7a1_nipd_analysis.py --outdir nipd_output
```

Run with your own pregnancy-level table (CSV):

```bash
python aldh7a1_nipd_analysis.py --data my_cohort.csv --outdir nipd_output
```

### Expected CSV columns

| Column | Meaning |
|---|---|
| `family` | Family ID (1–17) |
| `genotype` | Final fetal genotype: `Affected`, `Carrier`, `Unaffected` |
| `ff_8w`, `ff_10w`, `ff_12w` | Fetal fraction (%) at 8, 10, 12 weeks |
| `snps_8w`, `snps_final` | Informative SNPs at 8 weeks and at the final reportable sample |
| `dos_8w`, `dos_10w`, `dos_12w` | Integrated disease-allele dosage |
| `first_reportable_w` | First gestational week at which the sample was reportable (8, 10 or 12) |
| `nocall_8w` | `True` if no-call at 8 weeks |
| `nipd_call` | NIPD classification |
| `reference` | Invasive / postnatal reference genotype |

## Output

Written to `--outdir`:

- `Figure_2A_schematic.png`, `Figure_2_BE.png`
- `Figure_3.png`
- `Figure_4.png`
- `cohort_table.csv`

Diagnostic performance and demo model outputs are printed to the console.

## Thresholds

The reportability thresholds are study-specific quality criteria defined in the manuscript, not universal cut-offs. They are set as constants at the top of the script (`FF_MIN`, `SNP_MIN`, `POST_MIN`, `PHASE_MIN`, `DEPTH_MIN`).

## Citation

If you use this code, please cite the manuscript above. A `CITATION.cff` file or Zenodo DOI can be added after publication.

## License

Released under the MIT License (see `LICENSE`).

## Contact

Corresponding author: Chi-Bao Bui — bcbao@uhsvnu.edu.vn
Department of Biochemistry and Immunology, University of Health Sciences, Vietnam National University Ho Chi Minh City.
