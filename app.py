import html
import ipaddress
import json
import math
import re
import socket
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from flask import Flask, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent


def _resolve_artifact_path(filename: str) -> Path:
    candidate_paths = [BASE_DIR / "model" / filename, BASE_DIR / filename]
    for path in candidate_paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Unable to locate '{filename}'. Expected one of: "
        + ", ".join(str(path) for path in candidate_paths)
    )


class NaiveBayesModel:
    def __init__(self, payload: dict) -> None:
        self.stopwords = set(payload["stopwords"])
        self.token_re = re.compile(payload["token_pattern"])
        self.log_prior = {int(k): float(v) for k, v in payload["log_prior"].items()}
        self.unk_log_prob = {int(k): float(v) for k, v in payload["unk_log_prob"].items()}
        self.token_log_prob = {
            token: {0: float(probs["0"]), 1: float(probs["1"])}
            for token, probs in payload["token_log_prob"].items()
        }
        self.validation = payload.get("validation", {})
        self.tuning = payload.get("tuning", {})

    def preprocess(self, text: str) -> list[str]:
        text = re.sub(r"https?://\S+|www\.\S+", " ", text.lower())
        return [
            token
            for token in self.token_re.findall(text)
            if token not in self.stopwords and len(token) > 2
        ]

    def predict_with_confidence(self, text: str) -> tuple[int, float]:
        tokens = self.preprocess(text)
        score0 = self.log_prior[0]
        score1 = self.log_prior[1]

        for token in tokens:
            probs = self.token_log_prob.get(token)
            if probs is None:
                score0 += self.unk_log_prob[0]
                score1 += self.unk_log_prob[1]
            else:
                score0 += probs[0]
                score1 += probs[1]

        # Convert log-scores to probabilities stably.
        if score1 > score0:
            p1 = 1.0 / (1.0 + math.exp(score0 - score1))
        else:
            p1 = math.exp(score1 - score0) / (1.0 + math.exp(score1 - score0))

        label = 1 if score1 >= score0 else 0
        confidence = p1 if label == 1 else 1.0 - p1
        return label, confidence


TWEET_URL_RE = re.compile(r"(?:twitter\.com|x\.com)/[^/]+/status/(\d+)", re.IGNORECASE)


def _is_private_host(hostname: str) -> bool:
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True

    for item in addresses:
        raw_ip = item[4][0]
        ip = ipaddress.ip_address(raw_ip)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
    return False


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL must start with http:// or https://")
    if not parsed.hostname:
        raise ValueError("Invalid URL: host is missing")
    if _is_private_host(parsed.hostname):
        raise ValueError("Private/local URLs are not allowed")


def _fetch_tweet_text(tweet_url: str, timeout_s: float = 10.0) -> str:
    if not TWEET_URL_RE.search(tweet_url):
        raise ValueError("Please provide a valid Twitter/X status URL")

    query = urlencode({"url": tweet_url, "omit_script": "1", "dnt": "1"})
    endpoint = f"https://publish.twitter.com/oembed?{query}"
    req = Request(endpoint, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))

    tweet_html = payload.get("html", "")
    raw_text = re.sub(r"<[^>]+>", " ", tweet_html)
    clean_text = html.unescape(re.sub(r"\s+", " ", raw_text)).strip()
    if not clean_text:
        raise ValueError("Could not extract tweet text from this URL")
    return clean_text


def _extract_from_html_page(url: str, timeout_s: float = 12.0) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout_s) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            raise ValueError("URL did not return an HTML page")
        body = response.read().decode("utf-8", errors="ignore")

    body = re.sub(r"<script[\s\S]*?</script>", " ", body, flags=re.IGNORECASE)
    body = re.sub(r"<style[\s\S]*?</style>", " ", body, flags=re.IGNORECASE)

    title = " ".join(re.findall(r"<title[^>]*>(.*?)</title>", body, flags=re.IGNORECASE | re.DOTALL))
    meta_desc = " ".join(
        re.findall(
            r'<meta[^>]+(?:name=["\']description["\']|property=["\']og:description["\'])[^>]*content=["\'](.*?)["\']',
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    paragraphs = " ".join(re.findall(r"<p[^>]*>(.*?)</p>", body, flags=re.IGNORECASE | re.DOTALL))

    combined = " ".join([title, meta_desc, paragraphs])
    combined = re.sub(r"<[^>]+>", " ", combined)
    combined = html.unescape(re.sub(r"\s+", " ", combined)).strip()
    if len(combined) < 40:
        raise ValueError("Could not extract enough readable text from this URL")
    return combined


def _resolve_input_text(news_text: str, news_url: str) -> tuple[str, str]:
    if news_text.strip():
        return news_text.strip(), ""
    if not news_url.strip():
        raise ValueError("Please provide text or a URL to analyze.")

    _validate_public_url(news_url)
    if TWEET_URL_RE.search(news_url):
        extracted = _fetch_tweet_text(news_url)
    else:
        extracted = _extract_from_html_page(news_url)
    return extracted, extracted


MODEL_PATH = _resolve_artifact_path("news_nb_model.json")
CLASSIFIER = NaiveBayesModel(json.loads(MODEL_PATH.read_text(encoding="utf-8")))


@app.route("/")
def home():
    return render_template(
        "index.html",
        model_version="Naive Bayes (tuned)",
        model_metrics=CLASSIFIER.validation,
        model_tuning=CLASSIFIER.tuning,
        news_text="",
        news_url="",
    )


@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text", "")
    news_url = request.form.get("news_url", "")

    common_ctx = {
        "news_url": news_url,
        "model_version": "Naive Bayes (tuned)",
        "model_metrics": CLASSIFIER.validation,
        "model_tuning": CLASSIFIER.tuning,
    }

    try:
        text_for_prediction, extracted_text = _resolve_input_text(news_text, news_url)
    except Exception as exc:
        return render_template(
            "index.html",
            prediction=f"Input error: {exc}",
            extracted_text="",
            **common_ctx,
        )

    prediction, confidence = CLASSIFIER.predict_with_confidence(text_for_prediction)
    label = "True" if prediction == 1 else "False"

    return render_template(
        "index.html",
        prediction=label,
        confidence=f"{confidence * 100:.2f}%",
        extracted_text=extracted_text,
        news_text=text_for_prediction,
        **common_ctx,
    )


if __name__ == "__main__":
    app.run(debug=True)
