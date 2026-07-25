# Medical Chatbot System

An intent-classification chatbot for health-related questions. A Keras LSTM classifier maps a user's question to one of ~1,900 topic tags, and the matching answer is returned over a small Flask API.

> ## ⚠️ Not medical advice
>
> This project is an academic natural-language-processing exercise. It is **not** a diagnostic tool, and it must not be used to make decisions about anyone's health. Its answers come from a scraped Q&A corpus of unverified provenance, it has no safety filtering, no escalation path for emergencies, and no measured accuracy on held-out data. Anyone with a medical concern should contact a qualified clinician.

---

## How it works

```
question ──▶ preprocess ──▶ tokenize ──▶ LSTM classifier ──▶ tag ──▶ answer lookup ──▶ JSON
             (regex, lowercase,           (Embedding → LSTM ×2
              stopwords, lemmatize)        → Dense → softmax)
```

1. **Preprocessing** strips non-alphabetic characters, tokenizes with NLTK, lowercases, removes English stopwords, and lemmatizes with WordNet.
2. **Tokenizer** maps text to a 2,000-word vocabulary with an `<OOV>` token, padded/truncated to 200 tokens.
3. **Model** is an Embedding layer (dim 16) → two stacked LSTM layers (110 units each) → Dense(208, ReLU) → softmax over the tag classes.
4. **Serving** looks up the predicted tag in `intents.json` and returns the corresponding answer with the model's confidence score.

## Repository contents

| File | Purpose |
| --- | --- |
| `train.py` | Trains the classifier and writes `chat_model/`, `tokenizer.pickle`, `label_encoder.pickle` |
| `main.py` | Flask API serving the trained model |
| `intents.json` | Corpus of 6,318 question/answer records across 1,887 tags (~4.9 MB) |
| `tokenizer.pickle` | Fitted Keras tokenizer |
| `label_encoder.pickle` | Fitted scikit-learn `LabelEncoder` for tags |
| `requirements.txt` | Pinned dependencies |

Each record in `intents.json` has `question`, `answer`, `url`, and `tags` fields; most also carry `question_text` and `answer_author`.

---

## Setup

The dependencies pin TensorFlow 2.12, which supports **Python 3.8–3.11 only**. It will not install on 3.12 or newer.

```bash
git clone https://github.com/Mushu555/Medical-Chatbot-System.git
cd Medical-Chatbot-System

python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

NLTK corpora (`punkt`, `stopwords`, `wordnet`) download automatically on first run.

## Training

**You must train before serving.** The `chat_model/` directory is not committed to this repository, and `main.py` fails at import without it.

```bash
python train.py
```

This trains for 10 epochs and writes `chat_model/`, overwriting the two pickle files.

## Running the API

```bash
python main.py
```

Serves on `http://127.0.0.1:8090`.

**`GET /`** — health check, returns a plain-text status string.

**`POST /chat`**

```bash
curl -X POST http://127.0.0.1:8090/chat \
  -H "Content-Type: application/json" \
  -d '{"input": "what causes migraines?"}'
```

```json
{
  "response": "...",
  "score": "0.87"
}
```

---

## Known issues

### 1. The API returns a placeholder for every request

`main.py` reads the answer from `intent['responses']`:

```python
if 'responses' in intent:
    response_text = random.choice(intent['responses'])
else:
    response_text = "Sorry, I don't have a response for that yet."
```

No record in `intents.json` has a `responses` key — the field is called `answer`, which is also what `train.py` reads. The `else` branch therefore fires on every request and the real answer is never returned. Fix:

```python
response_text = intent['answer']
```

### 2. The match loop never breaks and can crash

The loop scans all 6,318 intents on every request instead of stopping at the first tag match, and `response_text` is only bound inside the loop body. If no intent matches the predicted tag, the function raises `UnboundLocalError` and the request 500s — the `# type: ignore` comment suppresses the warning rather than fixing it. Restore a `break` and a fallback default:

```python
response_text = "Sorry, I didn't understand that."
for intent in intents['intents']:
    if intent.get('tags') and intent['tags'][0] == tag:
        response_text = intent['answer']
        break
```

### 3. No held-out evaluation

The `train_test_split` call and the evaluation block in `train.py` are commented out, so the model trains on 100% of the data and reports only training accuracy. There is no measurement of how it performs on unseen questions.

### 4. Very sparse classes

6,318 examples spread over 1,887 tags is roughly 3 examples per class, with many classes having only one. A softmax over 1,887 classes at that density will generalise poorly, and the confidence score returned by the API should not be read as reliable. Consolidating tags into broader categories, or switching to sentence-embedding similarity rather than classification, would suit the data better.

### 5. Other

- `requirements.txt` begins with a UTF-8 BOM, which some tooling mis-parses on the first line.
- No `.gitignore`; `venv/`, `__pycache__/`, and `chat_model/` should be excluded.
- Adam at `learning_rate=0.01` is high for an LSTM; the default `0.001` is a better starting point.
- The Flask dev server binds `127.0.0.1` and is not suitable for production. Use a WSGI server such as Gunicorn if deploying.
- The repository has no GitHub description, topics, or licence.

---

## Data provenance

`intents.json` contains `url` and `answer_author` fields, indicating the corpus was scraped from an online medical Q&A service. The original source and its licensing terms are not documented in this repository. Confirm you have the right to redistribute this data before publishing the project further, and treat the answers themselves as unverified user-generated content rather than clinical guidance.
