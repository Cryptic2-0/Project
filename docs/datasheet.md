# Datasheet — IMDb 50K reviews

Following the [Gebru et al. 2018 Datasheets for Datasets](https://arxiv.org/abs/1803.09010) template.

## Motivation

### For what purpose was the dataset created?

Maas et al. 2011 released the IMDb 50K dataset to benchmark binary sentiment classification on movie reviews. It is balanced (25K positive + 25K negative) and split into 50/50 train/test by the original authors.

For MovieSentiment, the dataset is the training and evaluation corpus for the production DistilBERT INT8 classifier.

### Who created the dataset?

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts (Stanford). Original publication: ["Learning Word Vectors for Sentiment Analysis"](https://ai.stanford.edu/~amaas/data/sentiment/), ACL 2011.

### Who funded the creation?

Stanford NLP Group.

---

## Composition

### What do the instances represent?

User-submitted reviews on IMDb (`imdb.com`) for movies. Each instance is:
- `text`: the review body (English prose)
- `label`: 1 (positive, rating ≥ 7/10) or 0 (negative, rating ≤ 4/10)
- Movies in the [4, 6] middle range were **excluded** by the original authors to ensure clean polarity.

### How many instances total?

50,000 reviews. 25K training + 25K test in the original split. Class-balanced 50/50.

### Does the dataset contain all possible instances, or is it a sample?

A sample. The original authors selected at most 30 reviews per movie to prevent any single film from dominating the dataset, and excluded reviews from movies with fewer than 5 reviews. The full IMDb corpus is much larger.

### What data does each instance consist of?

Raw HTML-bearing user-submitted text. Cleaning is part of this project's pipeline (`src/moviesentiment/data/clean.py`):
- Strip `<br />` and other HTML tags
- Remove URLs
- Lowercase
- Collapse whitespace
- Drop duplicates
- Drop rows where cleaned text < 10 chars

### Is there a label?

Yes — binary {0, 1}. No "neutral" class.

### Is any information missing from individual instances?

No — every instance has both `text` and `label`. After cleaning, any row with null `text` is dropped (verified by the data-quality gate, `src/moviesentiment/data/validate.py`).

### Are there relationships between individual instances?

Some reviews are written by the same user; the original release does not preserve user identity. Some movies appear in multiple reviews; we don't deduplicate at the movie level (intentional — different reviewers, different sentiments).

### Are there recommended splits?

Yes — the original 25K/25K train/test. This project further splits train into train/val (80/20 by default, controlled by `params.yaml::split.val_size`). Stratified by label to preserve balance.

### Are there any errors, sources of noise, or redundancies?

- **Subjective labels**: rating-to-polarity is a heuristic. A reviewer who gave 7/10 to a movie they personally hated would be labeled positive.
- **Self-selection bias**: people who write IMDb reviews are more opinionated than the median moviegoer. Distribution skews polarized.
- **Era bias**: most reviews are from 2000–2011 (the collection window). Vocabulary, named entities, and genre conventions are weighted to that era.
- **English only**: explicitly filtered to English. Non-English input at inference time is out-of-distribution.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources?

Self-contained — text is included verbatim. The HuggingFace `datasets` mirror (`imdb`) ships the same content as the original Stanford tarball.

---

## Collection process

### How was the data acquired?

Scraped from `imdb.com` between 2008–2011 by the original authors using the IMDb-published review pages.

### What mechanisms or procedures were used?

Custom web scraper (not released). For this project we re-fetch via the HuggingFace `datasets` library:

```python
from datasets import load_dataset
ds = load_dataset("imdb")
```

### Who was involved in the data collection process?

The original Stanford NLP team. For this project, no further collection beyond HF download.

### Over what timeframe was the data collected?

2008–2011 (per Maas et al. 2011).

### Were any ethical review processes conducted?

No formal IRB. IMDb reviews are publicly posted by users who have agreed to IMDb's terms of service permitting redistribution of their reviews.

---

## Preprocessing / cleaning / labeling

Implemented in `src/moviesentiment/data/clean.py`. Verified by `src/moviesentiment/data/validate.py` (the data-quality gate in `dvc.yaml`).

Rules:
- Strip HTML tags via regex `<[^>]+>` → space
- Strip URLs via regex `https?://\S+|www\.\S+` → space
- Lowercase everything
- Collapse runs of whitespace
- Drop reviews where cleaned text length ≤ 10 characters
- Deduplicate on `text`
- Validate: rows non-null, label ∈ {0, 1}, P(label=1) ∈ [0.40, 0.60]

---

## Uses

### Has the dataset been used for any tasks already?

Yes — it's one of the most widely benchmarked sentiment classification datasets. Common reference numbers (test F1):
- TF-IDF + LR baseline: 0.88–0.90
- BiLSTM: 0.92
- ULMFiT: 0.95
- DistilBERT (FP32, full fine-tune): 0.93
- DistilBERT INT8 (this project): 0.939
- BERT-large fine-tuned: 0.96+

### Are there tasks for which the dataset should NOT be used?

- Multi-class sentiment (no neutral / mixed labels)
- Aspect-based sentiment (no aspect annotations)
- Non-English sentiment
- Out-of-domain sentiment (product reviews, news, social media)
- Anything that needs reviewer demographic information

### Is there a repository for tracking subsequent uses?

This project tracks downstream artifacts via DVC (`dvc.yaml`) and MLflow (`mlflow.db`). External uses outside this repo are not tracked.

---

## Distribution

### How is the dataset distributed?

Original: Stanford tarball at `ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz`. Mirror: HuggingFace `datasets/imdb`. License: per the original release, freely available for research use.

### When was it first distributed?

2011.

### Are there any restrictions on use?

The dataset is free for research and non-commercial use. Commercial use should review IMDb's terms of service.

---

## Maintenance

### Who is supporting / hosting / maintaining the dataset?

- Original: Stanford NLP group — no active maintenance.
- HuggingFace mirror: HF maintains the `datasets/imdb` namespace.
- For this project: snapshotted into `data/raw/reviews.parquet` and version-controlled via DVC (S3 remote). The hash is pinned in `dvc.lock`.

### Is there an erratum?

No formal errata. Known issues are documented above in "Composition → Are there any errors..."

### Will the dataset be updated?

The IMDb 50K is a frozen 2011 release and is not updated. New movies / reviews are not added. For drift detection, this project compares production traffic against this fixed reference set.
