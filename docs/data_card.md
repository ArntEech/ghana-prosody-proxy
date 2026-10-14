# Data Card

## Dataset used
The experimental corpus is derived from the AsanteTwiSenti dataset, which contains Ghanaian Asante Twi and multilingual social-media text, including Ghanaian Pidgin material.

**Dataset paper:** Gyimah et al. (2025), *Data in Brief*, 60, 111460. DOI: https://doi.org/10.1016/j.dib.2025.111460  
**Dataset release:** https://doi.org/10.5281/zenodo.14506733

## Experimental subset
The reported processed dataset contains 9,111 records after retaining positive, negative, and neutral sentiment labels. Class counts reported by the preparation pipeline are:
- Positive: 4,398
- Neutral: 2,575
- Negative: 2,138

Reported split: 6,377 train, 1,367 development, 1,367 test.

## Intended use
The dataset is used to evaluate sentiment classification and explore whether observable text features add predictive value. Existing sentiment labels are not labels for prosodic tone, expressive intensity, or pragmatic function.

## Known limitations
- Social-media spelling and language use vary by user and context.
- A sentiment label cannot fully represent sarcasm, stance, pragmatic meaning, or vocal delivery.
- Counts in dataset publications/releases may differ by version and filtering procedure; this project documents the actual retained subset used in the experiment.
