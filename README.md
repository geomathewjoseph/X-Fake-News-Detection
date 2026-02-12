# X-Fake-News-Detection

A Flask web app to detect whether Twitter-like text/claims are likely **True** or **False**.

## What is improved

- Uses a tuned **Multinomial Naive Bayes** model serialized to JSON (`model/news_nb_model.json`).
- Training uses only free/open tooling (Python standard library + local Excel files), no paid API and no proprietary model service.
- Training now merges both available datasets (`aippp.xlsx` + `bharatfakenewskosh.xlsx`) and combines statement/body fields to improve robustness.
- Inference supports:
  - direct text input,
  - Twitter/X status URLs (tweet text extracted via free oEmbed endpoint),
  - general article/news URLs (title + meta description + paragraph extraction).
- URL validation blocks private/local hosts for safer fetching.

## Run the app

```bash
python -m pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

## Train / re-train model

```bash
python train_model.py
```

Output model artifact:

- `model/news_nb_model.json`

## Test it yourself with a virtual environment

```bash
bash scripts/test_in_venv.sh
```

This creates `.venv`, tries to install dependencies, and runs tests.
If dependency installation fails (e.g., restricted network/proxy), it automatically falls back to offline checks.

## Offline test (no Flask install required in CI)

```bash
python tests/test_offline.py
```

This runs syntax checks, retrains the model, validates the generated JSON artifact,
and unit-tests core app logic using a tiny Flask stub.

## Free-only project policy

This project is implemented with free options only:

- Flask + Python stdlib
- local datasets in repository
- no paid inference APIs
- no paid data extraction service

## Full run guide

See `HOW_TO_RUN.md` for a complete step-by-step setup, run, retrain, and troubleshooting guide.
