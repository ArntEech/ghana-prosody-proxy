# Model Card

## Model
TF-IDF text representation with Logistic Regression, compared with a model that also uses 21 hand-engineered textual proxy features.

## Intended task
Predict positive, neutral, or negative labels in the prepared dataset. The proxy-enhanced model tests whether features such as elongation, emoji use, punctuation, capitalization, and candidate pragmatic markers add information beyond lexical TF-IDF features.

## Reported evaluation
The reported baseline achieves 69.71% accuracy, 66.59% Macro-F1, and 69.73% weighted-F1. The reported ablation full model achieves 70.52% accuracy, 67.66% Macro-F1, and 70.55% weighted-F1. A separate enhanced-model run reported 70.81% accuracy and 68.05% Macro-F1; this small discrepancy indicates the saved scripts/configurations need reconciliation before claiming exact reproducibility.

## Not intended for
- Recovering acoustic pitch or spoken prosody from text.
- Determining a person's true emotions, intent, or mental state.
- Making high-stakes decisions about individuals.
- Treating one spelling, emoji, or particle as having a fixed meaning in every context.

## Limitations
The dataset's sentiment labels do not directly measure prosody or pragmatic function. Performance can be affected by class imbalance, language mixing, rare expressions, ambiguity, annotation conventions, and dataset-specific vocabulary. Reported improvements are modest and have not been established as statistically significant.
