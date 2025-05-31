# Abstractive-Summarization

**Sai Arun Mendu**  
Data Science for Society and Business (DSSB)  
Constructor University, Bremen  
Spring 2025

---

## Overview

This repository contains code and documentation for an end-to-end pipeline that fine-tunes long-context Transformer models on the [GovReport]([https://github.com/huggingface/datasets/tree/master/datasets/gov_report](https://drive.google.com/file/d/1ik8uUVeIU-ky63vlnvxtfN2ZN-TUeov2/view)) corpus—a collection of multi-thousand-token U.S. government reports (CRS & GAO) paired with expert-written summaries/highlights.

We evaluate two state-of-the-art long-document models:

1. **LED-Base-16384** 
2. **Long-T5-TGlobal-Base** 

Our primary objective is to demonstrate that modern long-context architectures can accurately condense full‐length CRS and GAO reports into concise, reliable summaries. We benchmark:

- **Decoding strategies** – Greedy vs. Beam-4  
- **Context-length ablations** – 512 → 16 384 tokens  
- **Evaluation metrics** – ROUGE-1/2/L (F₁), BERTScore (F₁ on a 500-sample subset), plus a small human spot-check  

---

## Key Findings

- **LED-Base (Beam-4)** achieves ROUGE-1 ≈ 0.614, ROUGE-2 ≈ 0.569, ROUGE-L ≈ 0.567, BERTScore ≈ 0.936 on the combined CRS + GAO test sets.  
- **Long-T5 (Beam-4)** scores significantly lower (ROUGE-1 ≈ 0.445, BERTScore ≈ 0.753).  
- **Beam-4** offers slight but consistent improvements vs. greedy decoding (e.g., +0.014 ROUGE-2, +0.023 ROUGE-L for LED-Base).  
- **Context-length ablation (Long-T5)**: performance drops steeply below 1 024 tokens (e.g., ROUGE-1 0.563 @ 1 024 → 0.512 @ 512), and plateaus ≥ 4 096 tokens.  

Overall, LED-Base emerges as the best production model, balancing quality (both lexical and semantic) with modest computational overhead.

---

## Dataset: GovReport

GovReport aggregates:

- **CRS** (Congressional Research Service) briefs 
- **GAO** (Government Accountability Office) audits

Each JSON contains:

- **metadata**: `id`, `title`, `date`  
- **full report** (`report` field): a deeply nested hierarchy of `section_title` + `paragraphs` + `subsections`  
- **human summary**  
  - CRS: `summary` (list of paragraphs)  
  - GAO: `highlight` (structured object)  

GovReport provides official 80/10/10 splits via `.ids` files (e.g., `crs_train.ids`, `gao_test.ids`). We use these partitions verbatim for reproducibility.

---

## Repository Structure

```text
.
├── config.py
├── dataset.py
├── datamodule.py
├── descriptive_stats.py
├── evaluate.py
├── train_t5.py
├── train_led.py
├── metrics.py
├── preprocess.py
├── predict_led.py
├── predict_t5.py
├── requirements.txt
├── README.md
└── data/
    └── gov-report/
        ├── crs/
        │   ├── RLXXXXX.json
        │   └── … 
        ├── gao/
        │   ├── GAO-XX-XXX.json
        │   └── … 
        ├── split_ids/
        │   ├── crs_train.ids
        │   ├── crs_valid.ids
        │   ├── crs_test.ids
        │   ├── gao_train.ids
        │   ├── gao_valid.ids
        │   └── gao_test.ids 
```

## Limitations

- Results are based solely on data
- No fine-grained human annotation for factual correctness
- Memory limits constrain batch size and input length on modest GPUs


