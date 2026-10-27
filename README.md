# Chunking, Word Segmentation and Retrieval for Vietnamese Legal RAG

Code and manuscript for the paper *Chunking, Word Segmentation and Retrieval
Strategies for Vietnamese Legal RAG: An Empirical Study and a Benchmark
Contamination Warning* (in preparation for the Vietnam Journal of Computer
Science).

## Contents

| Path | Description |
|------|-------------|
| `rag_bench.py` | Experiment module: data loading, dev/test splits, syllable and word (pyvi) tokenization, chunking (article, fixed128, fixed256, sent256), sparse BM25, dense retrieval, RRF fusion, cross-encoder reranking, metrics, paired bootstrap. |
| `manuscript/vjcs_manuscript.tex` | Manuscript in the official VJCS LaTeX template. |
| `manuscript/refs.bib` | References. |
| `manuscript/ws-vjcs.cls`, `ws-vjcs.bst` | World Scientific VJCS class and bibliography style. |
| `manuscript/vjcs_manuscript_preview.pdf` | Preview PDF compiled with Tectonic (XeTeX). |
| `manuscript/preview_xetex.tex` | Wrapper used only to compile the preview with XeTeX. Do not submit. |
| `paper_outline.md` | Working outline and result notes (Vietnamese). |

## Data

Zalo AI 2021 Legal Text Retrieval, via the Hugging Face dataset
[`GreenNode/zalo-ai-legal-text-retrieval-vn`](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn)
(61,425 articles, 3,196 questions). The 788 queries of the official test qrels
are used for reporting only; the 2,408 non-overlapping training queries form
the development set used for every model-selection decision.

## Running the experiments

The experiments were run on Google Colab with an A100 GPU.

```bash
pip install sentence-transformers FlagEmbedding pyvi scikit-learn huggingface_hub
```

```python
import rag_bench as rb
corpus, queries, qrels_test, qrels_train = rb.load_all()
splits, desc = rb.make_splits(queries, qrels_test, qrels_train)
```

`rag_bench.py` caches segmentations, embeddings, per-query scores
(`perq.pkl`) and the result log (`results_v2.jsonl`) under `/content/cache`.

## Main results (test set, 788 queries, nDCG@10)

| System | nDCG@10 |
|--------|--------:|
| BM25, word segmentation, sent256 | 69.00 |
| multilingual-e5-large, fixed128 | 74.33 |
| RRF(BM25 + e5 + bge-m3 + EmbeddingGemma-2) | 79.19 |
| + bge-reranker-v2-m3 | 80.61 |

Vietnamese-specific retrievers and rerankers score higher on this benchmark,
but one was trained on data from the same competition and the others do not
disclose their training data; see Section 5.1 of the manuscript.

## Building the PDF

On Overleaf, upload the `manuscript/` folder and compile with pdfLaTeX
(pdflatex → bibtex → pdflatex × 2). Locally, with
[Tectonic](https://github.com/tectonic-typesetting/tectonic):

```bash
cd manuscript
tectonic preview_xetex.tex
```

## TODO

- Add the Colab notebook used to run the experiments.
- Add `results_v2.jsonl` and `perq.pkl` after re-running (the original Colab
  runtime was reset).
