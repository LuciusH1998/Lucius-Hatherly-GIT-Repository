# Quora Question Pair Duplicate Detection — BERT Models

A comparative study of modern transformer-based approaches for detecting duplicate questions on the Quora platform. This repository contains two notebooks implementing and comparing **4 models** across **3 fundamentally different NLP paradigms**.

---

## Notebooks

| Notebook | Models Inside |
|---|---|
| `baseline_and_embeddings.ipynb` | Logistic Regression · LightGBM · XGBoost |
| `baseline_and_embeddings.ipynb` | BGE · E5 · GTE |
| `baseline_and_embeddings.ipynb` | Siamese GRU |
| `qwen_model.ipynb` | QWEN LLM with LoRA |
| `models/WECNN/wecnn_main.ipynb` | WECNN |
| `deberta_crossencoder.ipynb` | DeBERTa-v3 Cross-Encoder |
| `sbert_setfit_stacking.ipynb` | SBERT Bi-Encoder · RoBERTa Cross-Encoder · Stacking Ensemble |

---

## Dataset

**Quora Question Pairs** — [Kaggle](https://www.kaggle.com/c/quora-question-pairs)

| Property | Value |
|---|---|
| Full dataset | 400,000+ question pairs |
| Sample used | 30,000 pairs |
| Train / Val / Test | 72% / 8% / 20% |
| Class balance | 63% Not Duplicate / 37% Duplicate |
| Split seed | `random_state=42` (identical across all notebooks for fair comparison) |

---

## Models

### 1. Logistic Regression Baseline
**Notebook:** `baseline_and_embeddings.ipynb`

**Architecture:** Classical linear classifier trained on engineered lexical, structural, and vectorized text features.

**Why useful:** Logistic Regression provides a simple but important linear reference point. If advanced neural models do not outperform it, their added complexity is not justified.

**Results:**

| Metric | Score |
|---|---:|
| Accuracy | 0.7600 |
| F1 | 0.6600 |
| AUC-ROC | 0.8360 |
| Log Loss | 0.4870 |

---

### 2. LightGBM Baseline
**Notebook:** `baseline_and_embeddings.ipynb`

**Architecture:** Gradient boosting decision-tree model trained on the same engineered and vectorized features as the baseline experiments.

**Why useful:** LightGBM captures nonlinear interactions between lexical and structural similarity features.

**Results:**

| Metric | Score |
|---|---:|
| Accuracy | 0.7840 |
| F1 | 0.7800 |
| AUC-ROC | 0.8740 |
| Log Loss | 0.4170 |

---

### 3. XGBoost Baseline
**Notebook:** `baseline_and_embeddings.ipynb`

**Architecture:** Gradient boosting decision-tree model trained on engineered lexical, structural features.

**Why useful:** XGBoost provides a second nonlinear classical baseline and tests whether boosted tree methods can exploit handcrafted similarity features.

**Results:**

| Metric | Score |
|---|---:|
| Accuracy | 0.7780 |
| F1 | 0.7800 |
| AUC-ROC | 0.6910 |
| Log Loss | 0.4320 |

---

### 4. BGE Embedding Model
**Notebook:** `baseline_and_embeddings.ipynb`

**Architecture:** Each question is encoded independently using a pretrained BGE sentence embedding model. Classification is then performed using similarity-based interaction features.

**Features:** cosine similarity, dot product, Euclidean distance, and Manhattan distance between question embeddings.

**Why unique:** BGE gives strong semantic representations while remaining scalable because question embeddings can be precomputed and reused.

**Results:**

| Metric | Score |
|---|---:|
| Accuracy | 0.7890 |
| F1 | 0.7200 |
| AUC-ROC | 0.8730 |
| Log Loss | 0.4220 |

---

### 5. E5 Embedding Model
**Notebook:** `baseline_and_embeddings.ipynb`

**Architecture:** Each question is encoded independently using a pretrained E5 embedding model, followed by similarity-feature classification.

**Features:** cosine similarity, dot product, Euclidean distance, and Manhattan distance between embeddings.

**Why useful:** E5 tests another modern embedding family under the same duplicate-question setup, allowing comparison between pretrained embedding spaces.

**Results:**

| Metric | Score |
|---|---:|
| Accuracy | 0.7510 |
| F1 | 0.6900 |
| AUC-ROC | 0.8440 |
| Log Loss | 0.4510 |

---

### 6. GTE Embedding Model
**Notebook:** `baseline_and_embeddings.ipynb`

**Architecture:** Each question is encoded independently using a pretrained GTE embedding model, then classified using embedding similarity features.

**Features:** cosine similarity, dot product, Euclidean distance, and Manhattan distance between embeddings.

**Why useful:** GTE provides another pretrained embedding baseline for comparing representation quality across embedding models.

**Results:**

| Metric | Score |
|---|---:|
| Accuracy | 0.7630 |
| F1 | 0.6900 |
| AUC-ROC | 0.8530 |
| Log Loss | 0.4510 |

---

### 7. Siamese GRU
**Notebook:** `siamese_gru.ipynb`

**Architecture:** Q1 and Q2 are encoded independently using a shared embedding layer followed by a GRU encoder. The resulting hidden states are concatenated and passed to a feedforward classifier.

**Why useful:** The Siamese GRU tests a non-transformer neural sequence model. Its weak results show the limitation of training sequence representations without strong pretrained contextual embeddings.

**Results:**

| Metric | Score |
|---|---:|
| Accuracy | 0.6290 |
| F1 | 0.0100 |
| AUC-ROC | 0.5580 |
| Log Loss | 0.6630 |

---

### 8. WECNN
**Notebook:** `wecnn.ipynb`

**Architecture:** Weighted Embedding Convolutional Neural Network that uses word embeddings and multi-scale 1D convolutions to capture local semantic and structural patterns between question pairs.

**Input views:**

| View | Description |
|---|---|
| Full sequence | Original token sequence for each question |
| Unique-token sequence | Duplicate tokens removed to emphasize lexical coverage |
| Keyword sequence | Keyword-focused representation with a frozen embedding layer |

**Feature extraction:** Multiple 1D convolutional filters are applied over embeddings. Global max pooling, global min pooling, and chunk-wise max pooling summarize the convolutional feature maps. Paired question features are compared using cosine similarity with a learnable bias term.

**Similarity signal:**

```text
cos(x, y) + λ = (xᵀy / ||x||||y||) + λ
```

**Why useful:** WECNN tests whether convolutional n-gram features and similarity-based reasoning can compete with pretrained embeddings and transformer cross-encoders. It performs better than the Siamese GRU, but does not meaningfully improve over the strongest classical and embedding baselines.

**Results:**

| Metric | Score |
|---|---:|
| Accuracy | 0.7835 |
| F1 | 0.6967 |
| AUC-ROC | 0.8469 |
| Log Loss | 0.4932 |

---

### 9. QWEN LLM with LoRA
**Notebook:** `qwen_lora.ipynb`

**Architecture:** QWEN is fine-tuned as a prompt-based generative classifier. The two questions are concatenated into one input sequence, and the model generates a label token indicating whether the questions are duplicates.

**Model:** Qwen3.5-0.8B / 0.8B-parameter QWEN configuration

**Training:**

| Parameter | Value |
|---|---|
| Fine-tuning method | LoRA |
| Framework | HuggingFace |
| Task format | Prompt-based binary classification |
| Input format | `Q1 [SEP] Q2 [SEP]` |
| Output format | Label token: `Yes` / `No` |
| Epochs | 3 |

**Why useful:** QWEN tests whether a modern LLM can solve duplicate detection as a generative semantic reasoning task. It obtains strong AUC, but its accuracy and F1 are lower than the transformer cross-encoders, likely because it is optimized as a generative label-token model rather than a discriminative classifier.

**Results:**

| Metric | Score |
|---|---:|
| Accuracy | 0.7500 |
| F1 | 0.7410 |
| AUC-ROC | 0.9050 |
| Log Loss | 0.5320 |

---

### 10. DeBERTa-v3 Cross-Encoder
**Notebook:** `deberta_crossencoder.ipynb`

**Architecture:** Both questions fed together as one sequence — `[CLS] Q1 [SEP] Q2 [SEP]`. Every token in Q1 directly attends to every token in Q2 at every transformer layer via **disentangled attention** (position and content encoded separately).

**Model:** `microsoft/deberta-v3-small` (44M parameters)

**Why unique over BERT:** DeBERTa separates position and content into independent vectors. Standard BERT sums them together, losing the ability to reason about positional and semantic similarity independently. DeBERTa achieves better accuracy with 3× fewer parameters than RoBERTa.

**Results:**

| Metric | Score |
|---|---|
| Accuracy | 0.8700 |
| F1 (weighted) | 0.8707 |
| AUC-ROC | 0.9387 |

**Outputs generated:**
- `deberta_results.png` — confusion matrix, loss curve, ROC curve

---

### 11. SBERT Bi-Encoder
**Notebook:** `sbert_setfit_stacking.ipynb`

**Architecture:** Q1 and Q2 encoded **separately** into 768-dim vectors using `sentence-transformers/all-mpnet-base-v2`. Classification uses interaction features: cosine similarity + element-wise difference + element-wise product → Logistic Regression.

**Why unique:** Bi-encoders allow pre-computing and caching all question embeddings. At inference time only one forward pass is needed per new question — 100× faster than cross-encoders at scale. This is how production search engines (Google, Bing) do semantic search.

**Results:**

| Metric | Score |
|---|---|
| Accuracy | 0.8392 |
| F1 (weighted) | 0.8400 |
| AUC-ROC | 0.9153 |

**Outputs generated:**
- `sbert_evaluation.png` — confusion matrix, ROC curve, PR curve
- `sbert_cosine_analysis.png` — cosine similarity distribution by class + threshold sweep
- `shap_sbert_bar.png` — SHAP feature importance (cosine_similarity dominates)
- `shap_sbert_beeswarm.png` — direction of each feature's effect
- `shap_sbert_waterfall.png` — individual prediction explanation
- `sbert_error_analysis.png` — false positive / false negative breakdown

---

### 12. RoBERTa Cross-Encoder
**Notebook:** `sbert_setfit_stacking.ipynb`

**Architecture:** Same cross-encoder pattern as DeBERTa — `[CLS] Q1 [SEP] Q2 [SEP]` fed as one sequence. Uses `roberta-base` (125M parameters) fine-tuned end-to-end for binary classification.

**Why unique over BERT:** RoBERTa removes BERT's flawed Next Sentence Prediction objective and uses dynamic masking — strictly better pre-training. Every token in Q1 attends to every token in Q2 directly, enabling richer cross-question interaction than bi-encoders.

**Training:**

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Max sequence length | 128 tokens |
| Best epoch | Epoch 2 (val F1 = 0.8537) |

**Results:**

| Metric | Score |
|---|---|
| Accuracy | 0.8640 |
| F1 (weighted) | 0.8650 |
| AUC-ROC | 0.9373 |

**Outputs generated:**
- `roberta_evaluation.png` — confusion matrix, ROC curve, PR curve
- `roberta_comparison.png` — training curve + all models bar chart
- `shap_roberta_bar.png` — top 20 most influential words
- `roberta_error_analysis.png` — false positive / false negative breakdown

---

### 13. Stacking Ensemble
**Notebook:** `sbert_setfit_stacking.ipynb`

**Architecture:** Meta-learning approach. Takes probability outputs from SBERT and RoBERTa plus hand-crafted features, feeds them into a Logistic Regression meta-classifier that learns when to trust which model.

**Meta-classifier inputs:**

| Feature | Source |
|---|---|
| SBERT probability | `sbert_clf.predict_proba()[:,1]` |
| RoBERTa probability | `softmax(roberta_logits)[:,1]` |
| jaccard | word set intersection/union |
| word_share | common/total words |
| q1_len, q2_len | character counts |
| len_diff | absolute length difference |
| q1_words, q2_words | word counts |

**Why it works:** Each model makes different errors. SBERT fails on zero-vocabulary-overlap paraphrases. RoBERTa sometimes over-relies on surface patterns. Hand features catch simple cases. The meta-classifier covers each model's blind spots — SHAP analysis confirms it assigns highest weight to neural model probabilities.

**Results:**

| Metric | Score |
|---|---|
| Accuracy | 0.8707 |
| F1 (weighted) | 0.8711 |
| AUC-ROC | **0.9426** ← best of all models |

**Outputs generated:**
- `stacking_evaluation.png` — all 3 models ROC curves on one plot
- `stacking_analysis.png` — meta-classifier coefficients + prediction breakdown
- `shap_stacking_bar.png` — which model the ensemble trusts most
- `shap_stacking_beeswarm.png` — SHAP direction of effects
- `shap_stacking_waterfall.png` — individual prediction explanation

---

## Full Results Comparison

| Model | Paradigm | Accuracy | F1 | AUC | Log Loss | Params |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | Classical ML baseline | 0.7600 | 0.6600 | 0.8360 | 0.4870 | — |
| LightGBM | Gradient boosting baseline | 0.7840 | 0.7800 | 0.8740 | 0.4170 | — |
| XGBoost | Gradient boosting baseline | 0.7780 | 0.7800 | 0.6910 | 0.4320 | — |
| BGE | Embedding model | 0.7890 | 0.7200 | 0.8730 | 0.4220 | — |
| E5 | Embedding model | 0.7510 | 0.6900 | 0.8440 | 0.4510 | — |
| GTE | Embedding model | 0.7630 | 0.6900 | 0.8530 | 0.4510 | — |
| GRU Siamese | Neural Siamese model | 0.6290 | 0.0100 | 0.5580 | 0.6630 | — |
| SBERT Bi-Encoder | Bi-encoder | 0.8390 | 0.8400 | 0.9150 | 0.3600 | 109M |
| RoBERTa Cross-Encoder | Cross-encoder | 0.8670 | 0.8680 | 0.9400 | 0.3440 | 125M |
| DeBERTa-v3 Cross-Encoder | Cross-encoder | 0.8690 | 0.8690 | 0.9390 | 0.3760 | 44M |
| **Stacking Ensemble** | **Meta-learning** | **0.8720** | **0.8720** | **0.9420** | 0.3610 | — |
| QWEN LLM | Generative LLM classifier | 0.7500 | 0.7410 | 0.9050 | 0.5320 | 0.8B |
| WECNN | CNN similarity model | 0.7835 | 0.6967 | 0.8469 | 0.4932 | — |

**Key findings:**
- Transformer cross-encoders and the stacking ensemble achieve the strongest overall performance, with AUC values around 0.94.
- The stacking ensemble obtains the best overall accuracy, F1, and AUC by combining SBERT, RoBERTa, and handcrafted feature signals.
- DeBERTa remains the strongest single model by accuracy/F1 while using fewer parameters than RoBERTa.
- SBERT performs below cross-encoders but remains much more scalable because embeddings can be precomputed.
- LightGBM is a strong classical baseline and performs competitively with the embedding models by AUC.
- QWEN reaches a high AUC of 0.905, outperforming many baselines by AUC, but underperforms the specialized transformer cross-encoders.
- WECNN improves over the Siamese GRU and is comparable to some baseline methods, but it does not close the gap with pretrained transformer models.
- The Siamese GRU performs poorly, showing that architecture alone is insufficient without strong pretrained representations.

---

## Notebook Structure

### `deberta_crossencoder.ipynb`

| Cell | Purpose |
|---|---|
| Cell 1 | Install dependencies (`transformers==4.40.0`, `datasets`, `sentencepiece`) |
| Cell 2 | Imports |
| Cell 3 | Load + sample 30k rows, stratified train/val/test split |
| Cell 4 | Load DeBERTa tokenizer, verify cross-encoder input format |
| Cell 5 | `QuoraCrossEncoderDataset` class — tokenizes Q1+Q2 together |
| Cell 6 | Load `deberta-v3-small`, print parameter counts |
| Cell 7 | Training setup — AdamW, linear warmup scheduler, CrossEntropyLoss |
| Cell 8 | Training loop with best model checkpoint saving |
| Cell 9 | Load best checkpoint, evaluate on test set |
| Cell 10 | Loss curve, accuracy curve, confusion matrix plots |
| Cell 11 | Live inference function |

### `sbert_setfit_stacking.ipynb`

| Cell | Purpose |
|---|---|
| Cell 1–2 | Install + imports |
| Cell 3 | Load data (same split as DeBERTa — `random_state=42`) |
| Cell 4–5 | SBERT: encode all pairs, build interaction features, train LR classifier |
| Report Cells 1–5 | SBERT evaluation plots, cosine analysis, SHAP, error analysis |
| Cell 11 | Redefine `QuoraCrossEncoderDataset` (needed since different notebook) |
| Cell 12 | RoBERTa: load model, train 3 epochs, evaluate |
| Report Cells 1–4 | RoBERTa evaluation plots, training curve, SHAP, error analysis |
| Cell 17 | Stacking ensemble: hand features + meta-classifier training |
| Cell 18–21 | Stacking evaluation plots, SHAP cells |

---

## Environment Setup

This project should be run inside a dedicated Conda environment to avoid dependency conflicts.

```bash
conda create -n qqp-env python=3.10 -y
conda activate qqp-env
conda install pip -y
pip install -r requirements.txt
pip install ipykernel

## Requirements

```
torch
transformers==4.40.0
sentence-transformers
datasets
sentencepiece
protobuf
scikit-learn
pandas
numpy
matplotlib
seaborn
shap
```

Install all:
```bash
pip install transformers==4.40.0 sentence-transformers datasets sentencepiece protobuf shap scikit-learn
```

---

## How to Run

### On Kaggle (recommended — free GPU)

1. Create a new notebook on [kaggle.com](https://www.kaggle.com)
2. Add the **Quora Question Pairs** dataset: `+ Add Data → Search "quora question pairs"`
3. Set accelerator: `Settings → Accelerator → GPU T4 x1, ideally use stronger GPUs like A100 or H100 to avoid long-run times on the tranformer and LLM methods which can take hours to train 
4. Upload the notebook file
5. Click **Run All**

### Data path
```python
# Kaggle path used in both notebooks
df = pd.read_csv('/kaggle/input/quora-question-pairs/train.csv.zip')
```

### Run order
Run `deberta_crossencoder.ipynb` first (standalone).  
Then run `sbert_setfit_stacking.ipynb` (also standalone — no dependency on DeBERTa notebook).

---

## Explainability (SHAP)

Both notebooks include SHAP analysis:

- **SBERT:** `LinearExplainer` on Logistic Regression — shows cosine_similarity is the dominant feature
- **RoBERTa:** Text masker with word-level attribution — shows which words drive duplicate prediction
- **Stacking:** `LinearExplainer` on meta-classifier — shows which model the ensemble trusts most

All SHAP cells install `shap` automatically via `!pip install shap -q` and can be run immediately after model evaluation without retraining.

---

## Error Analysis

Both notebooks include systematic error analysis identifying two failure modes:

**False Positives** (said duplicate, actually different):
- Pattern: same topic, different intent
- Example: *"What is Bitcoin?"* vs *"How do I buy Bitcoin?"*

**False Negatives** (said not duplicate, actually duplicate):
- Pattern: heavy paraphrase with zero shared vocabulary
- Example: *"How do I lose weight?"* vs *"What are ways to shed pounds?"*

---
