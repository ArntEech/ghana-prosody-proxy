# Annotation and Feature Guidelines

## Scope
The current classifier uses existing positive, neutral, and negative sentiment labels. It does **not** use a completed human annotation set for expressive intensity or pragmatic function. The rules below describe observable feature detection, not definitive interpretation.

## Observable feature families
- **Elongation:** repeated characters, e.g. `chaleeee`.
- **Emoji:** count and presence of emoji symbols.
- **Punctuation:** counts of `!`, `?`, ellipses, and stacked punctuation.
- **Capitalization:** uppercase-token or uppercase-ratio measures.
- **Candidate pragmatic markers:** selected forms such as `o`, `oo`, `paa`, `ei`, `eish`, and `hmm`; these are candidate markers and require context-sensitive interpretation.
- **Text structure:** token and character counts.

