# Deciphering Unspoken Tone in Ghanaian Social-Media Text

**Research project:** *Deciphering Unspoken Tone in text data*  
**Author:** Arnold B. Kyei  
**Course:** DCIT 316 — Computational Modelling in Social Media Mining  
**Department:** Computer Science, University of Ghana

## Overview

This project explores whether observable written features—such as repeated characters, emojis, punctuation, capitalization, and candidate pragmatic markers—provide useful information for sentiment classification in Ghanaian social-media text.

The project uses these features as **textual proxies**. It does not recover acoustic pitch, measure spoken prosody, or establish a writer's actual emotional state. A proxy may have different meanings depending on context.

## Research question

Do selected orthographic and pragmatic proxy features improve sentiment classification compared with a lexical TF-IDF baseline?

## Current experimental snapshot

The final corpus used in the reported experiment contains 9,111 records labelled positive, neutral, or negative. The reported split is 6,377 training, 1,367 development, and 1,367 test records.

| Experiment | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| TF-IDF baseline | 69.71% | 66.59% | 69.73% |
| TF-IDF + all proxy features (standalone enhanced run) | 70.81% | 68.05% | 70.89% |
| Ablation full model | 70.52% | 67.66% | 70.55% |



## Repository structure

```text
data/
  raw/asante_twi_senti/     Source dataset files
  processed/                Prepared corpus and extracted features
  annotated/                 Small annotation/example file
docs/
  annotation_guidelines.md  Notes on proxy annotation
  data_card.md               Dataset provenance and limitations
  ethics_statement.md        Data and interpretation safeguards
  model_card.md              Model purpose, evaluation, limitations
notebooks/                   Exploratory notebook placeholders
reports/                     Metrics, ablation table, figures
scripts/                     Preparation, feature extraction, training scripts
src/                         Preprocessing and feature-extraction modules
tests/                       Test placeholders
```



## Environment setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Workflow

Run commands from the repository root. The intended workflow is:

1. Prepare and filter the corpus:
   ```bash
   python scripts/prepare_data.py
   ```
2. Extract proxy features:
   ```bash
   python scripts/extract_features.py
   ```
3. Train the lexical baseline:
   ```bash
   python scripts/train_baseline.py
   ```
4. Train the proxy-enhanced model:
   ```bash
   python scripts/train_proxy_model.py
   ```
5. Run ablation experiments:
   ```bash
   python scripts/run_ablation.py
   ```


## Data and responsible use

The project is based on AsanteTwiSenti-derived Ghanaian social-media data.

See [`docs/data_card.md`](docs/data_card.md), [`docs/model_card.md`](docs/model_card.md), and [`docs/ethics_statement.md`](docs/ethics_statement.md).

## Results interpretation

The reported full ablation model improves over the baseline by about 0.81 percentage points in accuracy and 1.07 points in Macro-F1. Emoji and elongation features were the strongest individual groups in the reported ablation. Punctuation/capitalization did not improve the baseline in that experiment. These are modest improvements and should be interpreted with the limitations above.

## Citation



Gyimah, M. S., Hayfron-Acquah, J. B., Mensah Gyening, R.-M., Asante, M., Umar, F. I. A., & Kotei, E. (2025). *AsanteTwiSenti: A sentiment dataset of Ghanaian Asante Twi Tweets in a multilingual context*. Data in Brief, 60, 111460. https://doi.org/10.1016/j.dib.2025.111460

Dataset release: https://doi.org/10.5281/zenodo.14506733
