# Khung bài báo (nháp) — nộp Vietnam Journal of Computer Science

**Tiêu đề đề xuất:**
*Chunking, Word Segmentation and Retrieval Strategies for Vietnamese Legal RAG: An Empirical Study and a Contamination Warning*

> Trạng thái (27/10/2026): đã chạy xong 2 lượt trên Colab A100 80GB.
> - Lượt 1: 31 cấu hình (gồm cả mô hình tiếng Việt có dấu hiệu nhiễm dữ liệu).
> - Lượt 2: chọn cấu hình trên **tập dev riêng (2.408 câu hỏi train)**, báo cáo trên **788 câu test**, lưu điểm từng câu, **paired bootstrap B = 10.000**; thêm EmbeddingGemma-2; pipeline lai + rerank chỉ dùng mô hình sạch.
> - Chưa chạy: EmbeddingGemma-300m (repo gated, cần HF token).
> Đây là nháp có hỗ trợ của AI — giữ mục "Declaration of generative AI use" trước References.

---

## Abstract (nháp tiếng Anh)

Retrieval quality is the main bottleneck of retrieval-augmented generation (RAG) over legal documents, yet design choices for Vietnamese — a language whose word boundaries do not coincide with whitespace — have not been studied systematically. We benchmark sparse, dense, hybrid and reranking configurations on the Zalo AI 2021 Legal Text Retrieval corpus (61,425 articles), selecting configurations on a separate development set of 2,408 queries and reporting on 788 test queries with paired bootstrap significance tests. Word segmentation significantly improves BM25 (+2.2 to +2.4 nDCG@10 in three of four chunking settings), and sentence-based chunking adds a further +3.0 points. For dense retrievers, the benefit of chunking depends on the model's context limit: multilingual-E5 (512 tokens) gains +7.8 nDCG@10, whereas BGE-M3 and EmbeddingGemma-2 (1,024-token inputs) show no significant gain. Reciprocal rank fusion of BM25 with all clean dense retrievers reaches 79.2 nDCG@10 (+4.9 over the best single dense model) at negligible cost. Vietnamese-specific embedding models score 10–15 points above the best clean single model, and a Vietnamese reranker adds +10.3 points on top of fusion, but we document that one was trained on the benchmark's source data and the others do not disclose their training data; a general multilingual reranker gives no significant nDCG gain. We therefore caution against using this benchmark to rank Vietnamese retrieval models and release our code and per-query results.

**Keywords:** retrieval-augmented generation; Vietnamese; legal information retrieval; chunking; word segmentation; benchmark contamination

---

## 1. Introduction
- RAG đang phổ biến; chất lượng truy xuất quyết định chất lượng câu trả lời.
- Văn bản luật: dài (trung bình 301 âm tiết/điều, 10% dài hơn 608 âm tiết), cấu trúc phân cấp (Điều/Khoản/Điểm), thuật ngữ chuyên ngành.
- Tiếng Việt: đơn vị âm tiết ≠ đơn vị từ ("tòa án" = 2 âm tiết, 1 từ).
- Câu hỏi nghiên cứu:
  - **RQ1:** Tách từ (pyvi) so với tách âm tiết ảnh hưởng thế nào đến BM25?
  - **RQ2:** Chiến lược chunking nào tốt nhất cho sparse và dense retrieval?
  - **RQ3:** Mô hình embedding đa ngôn ngữ so với mô hình chuyên tiếng Việt?
  - **RQ4:** Truy xuất lai (RRF) và rerank cải thiện bao nhiêu?
- Đóng góp:
  1. Benchmark có hệ thống trên cùng dữ liệu, cùng phần cứng; chọn cấu hình trên dev, kiểm định thống kê trên test.
  2. Phát hiện: lợi ích của chunking phụ thuộc giới hạn ngữ cảnh của mô hình embedding.
  3. Pipeline sạch, rẻ: RRF(BM25 + dense đa ngôn ngữ) +4.9 nDCG@10 so với dense đơn lẻ.
  4. **Cảnh báo nhiễm dữ liệu**: mô hình/reranker tiếng Việt hàng đầu đã hoặc có thể đã train trên nguồn dữ liệu của benchmark.
  5. Mã nguồn + điểm từng câu hỏi công khai.

## 2. Related Work
- Truy xuất văn bản luật tiếng Việt (Zalo AI 2021, ALQAC, các hệ thống multi-stage).
- Mô hình embedding: multilingual-E5, BGE-M3, EmbeddingGemma, mô hình tiếng Việt (bkai, AITeamVN).
- Chunking trong RAG.
- Tách từ tiếng Việt (pyvi, VnCoreNLP, underthesea).
- Nhiễm dữ liệu benchmark (data contamination); kiểm định ý nghĩa thống kê trong IR (paired bootstrap).
> Cần tự tìm & kiểm tra từng trích dẫn trên Google Scholar.

## 3. Methodology
### 3.1 Dữ liệu và chia tập
- GreenNode/zalo-ai-legal-text-retrieval-vn (bản MTEB của Zalo AI 2021 Legal Text Retrieval).
- Corpus: 61.425 điều luật; độ dài (âm tiết): mean 301, median 197, p90 608, max 55.386.
- `queries.jsonl` có 3.196 câu hỏi có nội dung.
- **Test:** 788 câu hỏi, 793 cặp liên quan (~1,006 điều/câu hỏi) — chỉ dùng để báo cáo.
- **Dev:** 2.408 câu hỏi từ `qrels/train.jsonl` (không trùng test) — dùng để chọn chunking, chọn mô hình cho RRF, chọn tầng 1 cho reranker.

### 3.2 Chunking
| Chiến lược | Mô tả | Số chunk |
|---|---|---|
| article | nguyên điều luật | 61.425 |
| fixed128 | cửa sổ 128 âm tiết, chồng lấp 25% | 204.484 |
| fixed256 | cửa sổ 256 âm tiết, chồng lấp 25% | 113.751 |
| sent256 | gộp câu đến ≤256 âm tiết | 107.052 |

Điểm chunk được gộp về điều luật bằng max (MaxP) để đánh giá ở mức điều luật.

### 3.3 Tách từ cho BM25
- Âm tiết (regex `\w+`, lowercase) vs. từ (pyvi ViTokenizer).
- BM25: k1 = 1.2, b = 0.75, idf kiểu Lucene; cài đặt bằng ma trận thưa.

### 3.4 Mô hình dense
| Ký hiệu | Mô hình | Max token | Nhóm |
|---|---|---|---|
| e5-large | intfloat/multilingual-e5-large | 512 | sạch |
| bge-m3 | BAAI/bge-m3 | 1024 | sạch |
| gemma-emb-2 | google/embeddinggemma-2 (740M; prompt SearchQuery/Document, bf16) | 1024 | sạch |
| bkai-vi † | bkai-foundation-models/vietnamese-bi-encoder (input tách từ pyvi) | 256 | nhiễm |
| aiteam-vi ‡ | AITeamVN/Vietnamese_Embedding | 1024 | không rõ |

- EmbeddingGemma-300m: dự kiến nhưng **không chạy được** (repo gated, cần chấp nhận giấy phép + HF token).
- † bkai-vi: model card ghi huấn luyện trên MS MARCO-vi + SQuAD v2-vi + **80% tập train Zalo Legal 2021** → đã tiếp xúc dữ liệu cùng nguồn với tập đánh giá.
- ‡ aiteam-vi: ~300k triplet tiếng Việt, **không công bố nguồn**; model card tuyên bố không train trên Zalo Legal.

### 3.5 Lai ghép & rerank
- Reciprocal Rank Fusion (k = 60).
- **Lượt 2 (kết quả chính):** chỉ mô hình sạch. Trên dev chọn: BM25 = bm25-word|sent256; dense xếp theo dev nDCG@10 = e5-large|fixed128 (77.65) > bge-m3|fixed128 (76.69) > gemma-emb-2|fixed256 (72.43). Tầng 1 cho reranker chọn theo dev R@20 → RRF(BM25 + cả 3 dense).
- Rerank top-50 (max 1024 token) bằng BAAI/bge-reranker-v2-m3 (đa ngôn ngữ) và AITeamVN/Vietnamese_Reranker ‡ (~1,1 triệu triplet, **không công bố nguồn**).
- **Lượt 1 (phân tích phụ):** cấu hình được chọn trên test và tình cờ dựa toàn bộ trên bkai-vi † → chỉ dùng để minh họa ảnh hưởng của nhiễm dữ liệu.

### 3.6 Độ đo và kiểm định
- Recall@{1,5,10,20}, MRR@10, nDCG@10.
- Paired bootstrap trên điểm từng câu hỏi (B = 10.000, seed 0), báo cáo Δ, khoảng tin cậy 95% và p hai phía. p = 0.0000 nghĩa là p < 1/B = 0.0001.
- Phần cứng: 1× NVIDIA A100 80GB (Colab Pro), 12 vCPU. Kết quả lượt 1 và lượt 2 trên test trùng nhau (sai khác ≤ 0.05 điểm) → tái lập được.

## 4. Results
> Nguồn: `/content/cache/results.jsonl` (lượt 1), `results_v2.jsonl` + `perq.pkl` (lượt 2). Số liệu trên 788 câu test trừ khi ghi khác. In đậm = tốt nhất trong nhóm.

### Bảng 1 — BM25: tách âm tiết vs. tách từ (pyvi) × chunking

| Chunking | Tách | R@1 | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 | Thời gian (s) |
|---|---|---|---|---|---|---|---|---|
| article | âm tiết | 44.16 | 75.57 | 82.49 | 86.99 | 57.63 | 63.68 | 22.9 |
| article | từ | 46.32 | 77.92 | 85.98 | 91.12 | 59.68 | 66.03 | 65.9 |
| fixed128 | âm tiết | 48.29 | **79.44** | 84.45 | 90.04 | 61.94 | 67.47 | 34.3 |
| fixed128 | từ | 51.02 | 79.06 | 85.15 | **92.70** | 63.27 | 68.58 | 82.9 |
| fixed256 | âm tiết | 47.59 | 77.79 | 83.57 | 88.32 | 60.93 | 66.46 | 31.3 |
| fixed256 | từ | 50.25 | 79.19 | **86.42** | 91.81 | 63.04 | 68.69 | 82.8 |
| sent256 | âm tiết | 49.11 | 77.66 | 83.31 | 89.02 | 61.56 | 66.85 | 23.1 |
| sent256 | từ | **52.66** | 78.68 | 84.96 | 91.31 | **63.90** | **69.00** | 64.0 |

Dev chọn bm25-word|sent256 (dev nDCG@10 69.85) — trùng cấu hình tốt nhất trên test.

### Bảng 2 — Dense retrieval: nDCG@10 test (dev) theo mô hình × chunking

| Mô hình | Max tok | article | fixed128 | fixed256 | sent256 | Dev chọn |
|---|---|---|---|---|---|---|
| e5-large | 512 | 66.53 (71.76) | 74.33 (**77.65**) | 75.14 (77.21) | **75.39** (77.45) | fixed128 |
| bge-m3 | 1024 | 72.14 (72.70) | **73.64** (**76.69**) | 73.57 (75.94) | 72.98 (75.75) | fixed128 |
| gemma-emb-2 | 1024 | **71.80** (69.84) | 68.68 (71.33) | 71.10 (**72.43**) | 70.85 (71.17) | fixed256 |
| *bkai-vi †* | 256 | 84.08 | 87.75 | 88.55 | **88.63** | — (lượt 1) |
| *aiteam-vi ‡* | 1024 | **85.39** | 81.54 | 84.33 | 84.58 | — (lượt 1) |

Chi tiết cấu hình được dev chọn (R@1 / R@5 / R@10 / R@20 / MRR@10):
- e5-large|fixed128: 55.46 / 85.34 / 92.51 / 96.32 / 68.59
- bge-m3|fixed128: 55.52 / 84.90 / 91.37 / 95.05 / 67.97
- gemma-emb-2|fixed256: 53.05 / 81.22 / 88.01 / 92.89 / 65.72

Thời gian mã hóa trên A100: e5/bge-m3 120–200 s; gemma-emb-2 256–421 s (chậm hơn ~2 lần, đặc biệt với 204k chunk fixed128).

### Bảng 3 — Pipeline sạch: RRF và rerank (lượt 2, chọn trên dev)

| Hệ thống | R@1 | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 | Thời gian (s) |
|---|---|---|---|---|---|---|---|
| BM25 (bm25-word\|sent256) | 52.66 | 78.68 | 84.96 | 91.31 | 63.90 | 69.00 | 69 |
| Dense1 (e5-large\|fixed128) | 55.46 | 85.34 | 92.51 | 96.32 | 68.59 | 74.33 | 199 |
| RRF(e5 + bge-m3) | 56.85 | 87.37 | 93.65 | 96.32 | 70.04 | 75.73 | 0.4 |
| RRF(BM25 + e5) | 60.60 | 86.42 | 93.78 | 96.57 | 71.97 | 77.21 | 0.5 |
| RRF(BM25 + e5 + bge-m3) | 59.77 | 88.26 | 94.29 | 96.83 | 72.29 | 77.61 | 0.6 |
| **RRF(BM25 + e5 + bge-m3 + gemma-2)** | 62.31 | 89.59 | **94.67** | 96.83 | 74.22 | **79.19** | 0.7 |
| ↳ + bge-reranker-v2-m3 | 65.74 | 89.40 | 94.04 | **96.95** | 76.34 | 80.61 | 268 |
| ↳ + Vietnamese_Reranker ‡ | *79.89* | *95.56* | *97.08* | *97.84* | *87.00* | *89.49* | 265 |

(Thời gian RRF chỉ là bước gộp; chưa tính thời gian chạy các bộ truy xuất thành phần.)

### Bảng 4 — Kiểm định paired bootstrap (test, n = 788, B = 10.000)

| So sánh (A − B) | Δ nDCG@10 [CI95] | p | Δ R@1 [CI95] | p |
|---|---|---|---|---|
| **RQ1** BM25 từ − âm tiết, article | +2.35 [+0.80, +3.92] | 0.002 | +2.16 [−0.51, +4.82] | 0.129 |
| BM25 từ − âm tiết, fixed128 | +1.11 [−0.33, +2.55] | 0.125 | +2.73 [+0.38, +5.08] | 0.024 |
| BM25 từ − âm tiết, fixed256 | +2.22 [+0.74, +3.71] | 0.004 | +2.66 [+0.13, +5.33] | 0.049 |
| BM25 từ − âm tiết, sent256 | +2.15 [+0.68, +3.58] | 0.004 | +3.55 [+1.02, +6.09] | 0.007 |
| **RQ2** BM25-từ sent256 − article | +2.97 [+1.38, +4.57] | <0.001 | +6.35 [+3.43, +9.39] | <0.001 |
| e5-large fixed128 − article | +7.80 [+5.67, +9.92] | <0.001 | +6.47 [+3.17, +9.64] | <0.001 |
| bge-m3 fixed128 − article | +1.50 [−0.39, +3.38] | 0.121 | +1.27 [−1.78, +4.31] | 0.442 |
| gemma-emb-2 fixed256 − article | −0.69 [−2.34, +0.96] | 0.427 | −2.41 [−5.33, +0.51] | 0.119 |
| **RQ4** Dense1 − BM25 | +5.33 [+3.03, +7.74] | <0.001 | +2.79 [−1.02, +6.66] | 0.153 |
| RRF(BM25 + e5) − Dense1 | +2.89 [+1.18, +4.53] | <0.001 | +5.14 [+1.84, +8.38] | 0.003 |
| RRF(e5 + bge-m3) − Dense1 | +1.40 [+0.37, +2.42] | 0.007 | +1.40 [−0.51, +3.30] | 0.186 |
| RRF(BM25 + e5 + bge-m3) − Dense1 | +3.28 [+1.85, +4.71] | <0.001 | +4.31 [+1.65, +7.11] | 0.002 |
| RRF(BM25 + 3 dense) − Dense1 | +4.86 [+3.26, +6.40] | <0.001 | +6.85 [+3.81, +9.90] | <0.001 |
| + bge-reranker-v2-m3 − RRF(BM25 + 3 dense) | +1.42 [−0.53, +3.41] | 0.157 | +3.43 [+0.13, +6.85] | 0.042 |
| + Vietnamese_Reranker ‡ − RRF(BM25 + 3 dense) | +10.30 [+8.36, +12.30] | <0.001 | +17.58 [+14.09, +21.07] | <0.001 |

### Bảng 5 — Phân tích phụ: pipeline dựng trên bkai-vi † (lượt 1, chọn trên test)

dense1 = bkai-vi|sent256, dense2 = bkai-vi|fixed256, bm25 = bm25-word|sent256.

| Hệ thống | R@1 | R@10 | R@20 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| dense1 | 76.90 | 97.34 | 98.22 | 85.77 | 88.63 |
| RRF(bm25 + dense1) | 69.92 | 97.72 | 98.86 | 80.43 | 84.63 |
| RRF(dense1 + dense2) | 77.16 | 97.59 | 98.48 | 86.02 | 88.87 |
| RRF(bm25 + dense1 + dense2) | 75.25 | 98.48 | 98.98 | 84.54 | 87.96 |
| dense1 → bge-reranker-v2-m3 | 66.88 | 96.45 | 98.60 | 77.86 | 82.34 |
| RRF(d1 + d2) → Vietnamese_Reranker ‡ | 80.27 | 97.46 | 98.41 | 87.61 | 90.05 |

### Hình đề xuất
- **Hình 1:** nDCG@10 theo chunking cho 5 mô hình dense → e5 dốc lên mạnh; bge-m3 gần phẳng; gemma-emb-2 và aiteam-vi đi xuống khi chunk nhỏ.
- **Hình 2:** Forest plot của Bảng 4 (Δ nDCG@10 với CI95).
- **Hình 3:** Tác động của cùng một bước (thêm BM25 / thêm reranker đa ngôn ngữ) lên pipeline sạch vs. pipeline bkai: dấu ngược nhau.
- **Hình 4:** Thời gian vs. nDCG@10.

## 5. Discussion

### RQ1 — Tách từ
- Tách từ pyvi cải thiện nDCG@10 có ý nghĩa thống kê ở 3/4 chiến lược (+2.15 đến +2.35, p ≤ 0.004); ở fixed128 không có ý nghĩa với nDCG@10 (+1.11, p = 0.12) nhưng có với R@1 (+2.73, p = 0.02).
- R@1 tăng có ý nghĩa ở fixed128, fixed256, sent256 (+2.7 đến +3.6) → lợi ích tập trung ở thứ hạng đầu.
- Giải thích: BM25 trên âm tiết coi "tòa" và "án" là hai đặc trưng độc lập; tách từ khôi phục đơn vị nghĩa ("tòa_án").
- Chi phí: lập chỉ mục chậm hơn 2–3 lần (65–90 s vs. 23–43 s), chỉ chạy offline một lần.

### RQ2 — Chunking
- **BM25:** sent256 tốt hơn nguyên điều +2.97 nDCG@10, R@1 +6.35 (p < 0.001).
- **Dense — lợi ích phụ thuộc giới hạn ngữ cảnh:**
  - e5-large (512 token): +7.80 (p < 0.001) — nguyên điều dài bị cắt cụt khi mã hóa.
  - bge-m3 (1024 token): +1.50, không có ý nghĩa (p = 0.12).
  - gemma-emb-2 (1024 token): −0.69, không có ý nghĩa; fixed128 tệ nhất (68.68).
  - aiteam-vi ‡ (1024 token, lượt 1): nguyên điều tốt nhất; fixed128 giảm 3.85.
  - bkai-vi † (256 token, lượt 1): +4.55.
- → Mô hình ngữ cảnh ≤512 token **cần** chunking; mô hình 1024 token không hưởng lợi và có thể bị hại khi chunk quá nhỏ.
- Chunk theo câu (sent256) ngang hoặc tốt hơn cắt cứng, tạo ít chunk hơn.

### RQ3 — Đa ngôn ngữ vs. chuyên tiếng Việt
- Trong nhóm sạch: e5-large > bge-m3 > gemma-emb-2 (dev chọn: 74.33 / 73.64 / 71.10 trên test). EmbeddingGemma-2 — dù mới hơn — kém e5/bge-m3 khoảng 2.5–3.2 điểm trên văn bản luật tiếng Việt và mã hóa chậm hơn ~2 lần. Nên kiểm tra thêm định dạng prompt (ví dụ thêm title) trước khi kết luận.
- Nhóm chuyên tiếng Việt cao hơn 10–15 điểm (88.63 / 85.39 vs. 75.39), nhưng **không thể kết luận** do bkai-vi đã train trên 80% Zalo Legal train và aiteam-vi không công bố dữ liệu.
- **Cảnh báo:** Zalo Legal (và bản MTEB ZacLegalTextRetrieval) không còn phù hợp để xếp hạng mô hình truy xuất tiếng Việt; cần tập đánh giá sạch (văn bản ban hành sau ngày phát hành mô hình, hoặc ALQAC mới).

### RQ4 — Lai ghép & rerank
- **Pipeline sạch:** mỗi thành phần thêm vào đều có ích và có ý nghĩa thống kê:
  - Dense1 (e5) > BM25: +5.33 (p < 0.001).
  - RRF(BM25 + e5): +2.89 so với e5 — BM25 bổ sung tín hiệu khớp từ khóa (R@1 +5.14).
  - RRF(BM25 + 3 dense sạch): **79.19**, +4.86 so với dense tốt nhất, chi phí gộp không đáng kể.
- **Rerank đa ngôn ngữ (bge-reranker-v2-m3):** +1.42 nDCG@10 không có ý nghĩa (p = 0.16); R@1 +3.43 (p = 0.04); tốn ~270 s GPU cho 788 câu.
- **Vietnamese_Reranker ‡:** +10.30 (p < 0.001), R@1 79.89 — nhưng dữ liệu huấn luyện không công bố; mức điểm ngang bkai-vi † → xếp cùng nhóm "nghi nhiễm".
- **Đối chiếu với pipeline bkai (Bảng 5):** cùng một bước cho kết quả ngược dấu:
  - Thêm BM25: sạch +2.89 ↔ bkai −4.00.
  - Thêm bge-reranker-v2-m3: sạch +1.42 ↔ bkai −6.29.
  - Diễn giải: khi bộ truy xuất đã "thuộc" miền dữ liệu đánh giá, các tín hiệu độc lập (BM25, reranker tổng quát) chỉ kéo thứ hạng về mức của chúng (~80). Đây là một dấu hiệu thực nghiệm của nhiễm dữ liệu.

### Khuyến nghị thực tế
1. Luôn tách từ (pyvi) trước BM25 cho tiếng Việt.
2. Chunk theo câu (~256 âm tiết) cho BM25 và cho embedding ≤512 token; với embedding 1024 token, giữ nguyên điều hoặc chunk lớn.
3. Dùng RRF(BM25 + nhiều dense đa ngôn ngữ) làm mặc định: +4.9 nDCG@10, gần như miễn phí.
4. Chỉ thêm reranker sau khi kiểm chứng trên dev của chính miền; reranker đa ngôn ngữ cải thiện R@1 nhưng không chắc cải thiện nDCG@10.
5. Kiểm tra dữ liệu huấn luyện của mô hình/reranker trước khi tin điểm benchmark.

### Phân tích lỗi (cần làm thêm)
Lấy ~30 câu hỏi mà RRF(BM25 + 3 dense) sai ở top-10, phân loại: (a) diễn đạt khác thuật ngữ luật, (b) điều luật bị sửa đổi/thay thế, (c) cần nhiều điều luật, (d) nhãn có thể sai. Có thể làm từ `perq.pkl`.

## 6. Threats to Validity / Limitations
- **Nhiễm dữ liệu:** bkai-vi train trên 80% Zalo Legal train; aiteam-vi và Vietnamese_Reranker không công bố dữ liệu. Tập dev cũng lấy từ Zalo Legal train → dùng được cho mô hình sạch, nhưng không dùng được để đánh giá bkai-vi.
- EmbeddingGemma-300m chưa đánh giá (repo gated); EmbeddingGemma-2 dùng prompt mặc định "Document" (title: none) — chưa thử thêm tiêu đề điều luật vào trường title.
- Một bộ dữ liệu, một miền; chỉ đánh giá truy xuất (chưa đánh giá câu trả lời LLM).
- ~1 điều luật liên quan/câu hỏi → R@k ≈ hit rate; có thể có false negative trong nhãn.
- Kiểm định chưa hiệu chỉnh đa so sánh (15 phép so sánh × 2 độ đo) → nên báo cáo thêm Holm–Bonferroni; các kết luận chính (p < 0.001) vẫn đứng vững.
- Độ dài chunk tính bằng âm tiết, không phải token của từng mô hình.

## 7. Conclusion
Hai "nút vặn" rẻ — tách từ và chunking — cải thiện truy xuất luật tiếng Việt có ý nghĩa thống kê, nhưng lợi ích của chunking phụ thuộc giới hạn ngữ cảnh của mô hình embedding. Kết hợp BM25 với nhiều mô hình dense đa ngôn ngữ bằng RRF là cấu hình sạch tốt nhất (79.2 nDCG@10) với chi phí không đáng kể. Các mô hình và reranker chuyên tiếng Việt đạt điểm cao hơn nhiều, nhưng bằng chứng từ model card và từ hành vi ngược dấu khi kết hợp với tín hiệu độc lập cho thấy benchmark luật tiếng Việt phổ biến nhất đang bị nhiễm và cần được thay thế bằng một tập đánh giá sạch.

## Declaration of generative AI use
*During the preparation of this work the author(s) used [tool] for [task]. After using this tool, the author(s) reviewed and edited the content as needed and take full responsibility for the content of the publication.*

## Code & data availability
Link GitHub (notebook + rag_bench.py + results_v2.csv + perq.pkl).
