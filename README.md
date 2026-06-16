# Message Classifier

Intelligent Messaging Safety & Priority Classifier — a Python micro‑service
that categorises text messages into one of four classes:

| Label      | Meaning                                                              |
| ---------- | -------------------------------------------------------------------- |
| `normal`   | Regular conversational message — safe to display in the main chat.   |
| `spam`     | Promotional, fraudulent, or otherwise unwanted message.              |
| `abusive`  | Aggressive, insulting, or harassing language directed at others.     |
| `hateful`  | Hateful or discriminatory language targeting a person or group.      |

The service is designed to be called by a MERN backend (Node.js + Express)
over HTTP. The MERN side stores messages in MongoDB and uses the returned
label to decide whether a message goes straight into the chat or is hidden
behind a "view this flagged message?" warning, as described in the project
report (Chapters 1–3).

Stack: **FastAPI**, **scikit-learn** (TF‑IDF + Multinomial Naive Bayes),
**NLTK** (preprocessing), **uv** (package management), **Docker**.

## Quickstart

### 1. Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (`brew install uv`)
- Python 3.10–3.12 (uv will auto-provision the version pinned in
  [`.python-version`](.python-version) if needed)
- Docker (Docker Desktop / OrbStack / colima) for the container workflow

### 2. Install dependencies

```bash
make sync          # uv sync --all-groups
```

### 3. Build the dataset

Downloads five public datasets, normalises and balances them into
`data/processed/messages.csv` (12,000 rows, 3,000 per class):

| Source | Rows | Used for |
| ------ | ---: | -------- |
| **SMS Spam Collection** (UCI) | ~5,500 | `spam` (ham + spam) |
| **Davidson Hate-Speech & Offensive Language** | ~25,000 | `hateful` / `abusive` / `normal` |
| **HateXplain** (Mathew et al.) | ~20,000 | `hateful` / `abusive` / `normal` (majority vote of 3 annotators) |
| **DailyDialog** (Li et al., 2017 -- `pixelsandpointers/better_daily_dialog` mirror) | ~87,000 utterances | `normal` |
| **Enron-Spam** (Metsis et al. -- `SetFit/enron_spam` mirror) | ~33,000 emails | `spam` + `normal` (Subject + Message concatenated; rows outside 10-800 chars dropped to stay close to chat-messaging length) |

```bash
make data           # downloads + builds messages.csv
make data EXTRA="--exclude hatexplain dailydialog enron"   # ablation against the original 2-source baseline
```

Reading DailyDialog requires `pyarrow`, declared in the build-time
`data` dependency group of [pyproject.toml](pyproject.toml). It is *not*
installed in the runtime Docker image.

Per-class raw pools after all sources (before balancing): normal ~97k,
abusive ~25k, spam ~9k, hateful ~7.6k. The balance step samples 3,000
rows per class with `random_state=42`.

### 4. Train the classifier

```bash
make train          # baseline (~3 s on the full 12k dataset)
make train-tuned    # GridSearchCV over alpha + word/char ngram ranges (~20 s)
```

Outputs:

- `models/classifier.joblib` — the fitted sklearn `Pipeline`
  (preprocessing + word/char TF-IDF FeatureUnion + MultinomialNB)
- `models/metrics.json` — accuracy, per-class precision/recall/F1,
  confusion matrix, train/val/test split sizes, timestamp, and (when
  `--grid-search` is used) the tuned hyperparameters under a
  `grid_search` key.

Optional: `make evaluate` re-runs the test split and saves
`models/confusion_matrix.png`.

Latest test-split metrics on the 12,000-row dataset (`random_state=42`,
n_test=1,200):

| Variant                                   | macro-F1 | normal F1 | spam F1 | abusive F1 | hateful F1 |
| ----------------------------------------- | -------: | --------: | ------: | ---------: | ---------: |
| Round 3 (word TF-IDF only)                |    0.782 |     0.743 |   0.921 |      0.733 |      0.731 |
| Round 4 baseline (preproc + char n-grams) |    0.807 |     0.772 |   0.935 |      0.755 |      0.766 |
| Round 4 tuned (`make train-tuned`)        |    0.814 |     0.799 |   0.940 |      0.763 |      0.755 |

Best params chosen by 5-fold CV (`f1_macro`):
`clf__alpha=0.1`, `word__ngram_range=(1,3)`, `char__ngram_range=(3,4)`.

### 5. Run the service

```bash
make run                       # uvicorn on http://127.0.0.1:8000
# or
make docker-up                 # docker compose up -d
make docker-logs               # tail container logs
make docker-down
```

OpenAPI docs: <http://localhost:8000/docs>

## Make targets

| Target          | Description                                       |
| --------------- | ------------------------------------------------- |
| `sync`          | Create venv and install runtime + dev deps (uv)   |
| `data`          | Download raw datasets and build `messages.csv`    |
| `train`         | Train the classifier and save artifacts           |
| `train-tuned`   | Train via GridSearchCV over alpha + ngram ranges  |
| `evaluate`      | Re-evaluate the trained model on the test split   |
| `run`           | Run FastAPI locally with `--reload`               |
| `test`          | Run pytest                                        |
| `lint`          | Run ruff lint                                     |
| `format`        | Run ruff format                                   |
| `docker-build`  | Build the Docker image                            |
| `docker-up`     | Start the classifier via `docker compose`         |
| `docker-down`   | Stop the docker-compose stack                     |
| `docker-logs`   | Tail container logs                               |
| `clean`         | Remove caches and build artifacts                 |

## Configuration

Settings are loaded from environment variables (or a local `.env` —
see [`.env.example`](.env.example)).

| Variable          | Default                       | Notes                                                    |
| ----------------- | ----------------------------- | -------------------------------------------------------- |
| `MODEL_PATH`      | `models/classifier.joblib`    | Path to the trained pipeline.                            |
| `METRICS_PATH`    | `models/metrics.json`         | Optional metrics file surfaced by `/api/v1/info`.        |
| `API_KEY`         | *(empty)*                     | When set, all `/predict*` and `/info` calls require `X-API-Key`. |
| `LOG_LEVEL`       | `INFO`                        | Standard Python logging level.                           |
| `MAX_BATCH_SIZE`  | `100`                         | Maximum number of texts in `/predict/batch`.             |
| `MAX_TEXT_LENGTH` | `4000`                        | Hard cap on individual text length (chars).              |
| `NLTK_DATA`       | `<repo>/.nltk_data`           | Where NLTK corpora live.                                 |
| `CLASSIFIER_PORT` | `8000`                        | Host port mapping in `docker-compose.yml`.               |
| `CLASSIFIER_API_KEY` | *(empty)*                  | Forwarded as `API_KEY` to the container.                 |

## API contract

All endpoints live under `/api/v1`. Endpoints marked **(auth)** require
the `X-API-Key` header when `API_KEY` is configured.

### `GET /api/v1/health`

Liveness probe. Always returns 200 if the process is running.

```json
{ "status": "ok", "app": "message-classifier", "version": "0.1.0" }
```

### `GET /api/v1/ready`

Readiness probe. Returns 200 once the model is loaded; 503 otherwise
(e.g. before training or if the artifact is missing).

### `GET /api/v1/info` *(auth)*

Returns model metadata and metrics:

```json
{
  "app_version": "0.1.0",
  "model_path": "/app/models/classifier.joblib",
  "model_version": "0.1.0",
  "trained_at": "2026-06-16T10:00:00+00:00",
  "classes": ["abusive", "hateful", "normal", "spam"],
  "metrics": { "test": { "accuracy": 0.93, "f1_macro": 0.92, "...": "..." } }
}
```

### `POST /api/v1/predict` *(auth)*

Request:

```json
{ "text": "congrats you won a free prize click here now" }
```

Response (200):

```json
{
  "label": "spam",
  "confidence": 0.93,
  "probabilities": {
    "normal":  0.02,
    "spam":    0.93,
    "abusive": 0.03,
    "hateful": 0.02
  }
}
```

Validation errors return 422; oversized text returns 413.

### `POST /api/v1/predict/batch` *(auth)*

Request:

```json
{ "texts": ["see you tomorrow", "you idiot shut up"] }
```

Response (200):

```json
{
  "results": [
    { "label": "normal", "confidence": 0.81, "probabilities": { "..." : 0.0 } },
    { "label": "abusive", "confidence": 0.74, "probabilities": { "...": 0.0 } }
  ]
}
```

Batches larger than `MAX_BATCH_SIZE` (default 100) return 400.

## Calling from the MERN backend

```env
# backend/.env
CLASSIFIER_URL=http://localhost:8000
CLASSIFIER_API_KEY=change-me
```

```js
// backend/src/services/classifier.js
const CLASSIFIER_URL = process.env.CLASSIFIER_URL;
const CLASSIFIER_API_KEY = process.env.CLASSIFIER_API_KEY;

export async function classifyMessage(text) {
  const res = await fetch(`${CLASSIFIER_URL}/api/v1/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(CLASSIFIER_API_KEY ? { "X-API-Key": CLASSIFIER_API_KEY } : {}),
    },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    throw new Error(`classifier error: ${res.status} ${await res.text()}`);
  }
  return res.json(); // { label, confidence, probabilities }
}
```

```js
// backend/src/routes/messages.js
router.post("/messages", auth, async (req, res, next) => {
  try {
    const { text, recipientId } = req.body;
    const { label, confidence } = await classifyMessage(text);

    const message = await Message.create({
      sender: req.user.id,
      recipient: recipientId,
      text,
      label,
      confidence,
    });

    res.status(201).json(message);
  } catch (err) {
    next(err);
  }
});
```

Equivalent curl:

```bash
curl -s -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{"text":"congrats you won a free prize click here now"}'
```

## Project layout

```
message-classifier/
  app/
    api/routes.py          # /health, /ready, /info, /predict, /predict/batch
    core/config.py         # pydantic-settings: MODEL_PATH, API_KEY, ...
    core/security.py       # X-API-Key header dependency
    core/logging.py        # central logging config
    ml/labels.py           # canonical 4-class label set
    ml/preprocessing.py    # lowercase + neg-aware tokenise + punctuation sentinels + lemmatise
    ml/pipeline.py         # build_pipeline() -> Pipeline w/ word + char_wb FeatureUnion
    ml/predictor.py        # Predictor.load(...).predict(...)
    schemas.py             # Pydantic request / response models
    main.py                # FastAPI app factory + lifespan
  scripts/
    download_data.py       # fetch SMS Spam + Davidson + HateXplain + DailyDialog + Enron-Spam
    build_dataset.py       # combine + balance -> data/processed/messages.csv
    train.py               # train and persist models/classifier.joblib
    evaluate.py            # re-evaluate + confusion matrix PNG
  tests/                   # pytest suite (preprocessing, predictor, api)
  data/                    # raw/, processed/  (gitignored)
  models/                  # artifacts            (gitignored)
  Dockerfile               # multi-stage (uv builder -> python:3.12-slim runtime)
  docker-compose.yml       # single classifier service
  pyproject.toml           # project deps + dev group + ruff/pytest config
  uv.lock                  # uv lockfile (committed)
  Makefile                 # convenience targets
```

## ML pipeline

A single sklearn `Pipeline` so preprocessing is bundled with the model and
identical at train time and serve time:

```
TextCleaner                                # lowercase, strip URLs/emails,
   |                                       # keep negations (`not`, `n't`, ...),
   |                                       # emit _excl_ / _qst_ punctuation
   |                                       # sentinels, tokenise via NLTK,
   |                                       # drop other stopwords, lemmatise
   |
FeatureUnion
   ├── TfidfVectorizer (analyzer="word",    ngram_range=(1,2|3))
   └── TfidfVectorizer (analyzer="char_wb", ngram_range=(3,4|5))
   |
MultinomialNB                              # alpha=0.3 (or tuned by GridSearchCV)
```

The `char_wb` branch lets the classifier generalise across simple
obfuscations (`idi0t`, `st*pid`) that pure word n-grams miss; the
preserved negation tokens prevent `"i do not hate you"` and
`"i hate you"` from collapsing into the same bag-of-words; and the
`_excl_` / `_qst_` sentinels surface punctuation as a feature without
touching the vectorizer config.

- Train/val/test split: 70/20/10 stratified, `random_state=42`
  (matches report §3.9).
- `make train` fits with the default hyperparameters above.
- `make train-tuned` runs `GridSearchCV` (5-fold, scoring `f1_macro`)
  over `alpha`, `word_ngram_range`, and `char_ngram_range`
  (3 × 3 × 2 = 18 combos), and saves `grid_search.best_params` into
  `metrics.json`.
- Persisted with `joblib.dump` → a single `models/classifier.joblib` file.
- `models/metrics.json` records accuracy, macro / weighted F1, per-class
  precision / recall / F1, and the test-split confusion matrix.

## Tests

```bash
make test
```

Covers:

- Preprocessing (lower-casing, URL/email stripping, stopwords,
  lemmatisation, edge cases, **negation retention**, and
  **`_excl_` / `_qst_` punctuation sentinels**).
- Pipeline structure (word + `char_wb` `FeatureUnion`) and a smoke
  test asserting that char n-grams generalise from `idiot` to `id1ot`.
- `Predictor` roundtrip (train → save → load → predict).
- `--grid-search` smoke test (1-combo grid on a 200-row fixture)
  validating that `metrics.json` gets a `grid_search.best_params` block.
- FastAPI surface: `/health`, `/ready` (with and without a model),
  `/info`, `/predict`, `/predict/batch`, validation failures, batch
  limit, and `X-API-Key` enforcement.

## License

MIT. See [LICENSE](LICENSE) (add your own if redistributing).
