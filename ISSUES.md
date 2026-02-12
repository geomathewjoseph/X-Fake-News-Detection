# Analysis Notes and Issues Found

## Fixed in this update

1. **Dependency-heavy runtime (`joblib`, `nltk`, `scikit-learn`) blocked app startup**
   - Replaced runtime dependency on pickle/scikit artifacts with a JSON Naive Bayes model loaded via pure Python.

2. **No reproducible tuning path for the classifier**
   - Added `train_model.py` to parse `aippp.xlsx`, split train/validation deterministically, evaluate hyperparameter configs, and save best model.

3. **Model artifact location fragility**
   - Kept artifact resolution logic to support `model/` and project-root paths.

4. **Missing validation for empty form input**
   - `/predict` now returns a friendly message when text input is blank.

5. **No visibility into model quality in UI**
   - The home page now surfaces validation metrics from the tuned model metadata.

## Remaining recommendations

- Add automated tests for `train_model.py` parsing logic and Flask route behavior.
- Add calibration/probability output if confidence display is needed in UI.
- Add language-specific tokenization if multilingual (non-English) performance is a priority.
