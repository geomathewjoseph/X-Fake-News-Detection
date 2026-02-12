# How to Run X-Fake-News-Detection

## 1) Clone and enter project

```bash
git clone <your-repo-url>
cd X-Fake-News-Detection
```

## 2) Recommended: run with virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your environment blocks package downloads (proxy/network restriction), use the offline test flow:

```bash
python tests/test_offline.py
```

## 3) Start the web app

```bash
python app.py
```

Open in browser:

- `http://127.0.0.1:5000`

## 4) Use the app

You can submit:

- Plain text (tweet/claim/news text), or
- A Twitter/X status URL, or
- A general news/article URL.

The app extracts text (for URLs) and predicts **True** or **False** with confidence.

## 5) Re-train the model

```bash
python train_model.py
```

This regenerates:

- `model/news_nb_model.json`

## 6) Run tests

### One-command venv + tests

```bash
bash scripts/test_in_venv.sh
```

### Direct offline test

```bash
python tests/test_offline.py
```

---

## Troubleshooting

- If `pip install` fails with proxy/network errors, run only offline checks:
  - `python tests/test_offline.py`
- If model file is missing, retrain:
  - `python train_model.py`
- If Flask import fails, ensure venv is active and requirements installed.
