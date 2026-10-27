"""Benchmark chunking, tach tu va truy xuat cho RAG van ban luat tieng Viet.

Du lieu: GreenNode/zalo-ai-legal-text-retrieval-vn (Zalo AI 2021 Legal Text Retrieval).
Danh gia o muc dieu luat: diem cua chunk duoc gop ve dieu luat goc bang max.
Luot 2: tap dev de chon cau hinh, luu diem theo tung cau hoi, kiem dinh paired bootstrap.
"""
import os

# tat thanh tien trinh de output cua Colab la van ban thuan (doc duoc tu DOM)
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import json
import math
import pickle
import random
import re
import time
from collections import defaultdict

import numpy as np

REPO = "GreenNode/zalo-ai-legal-text-retrieval-vn"
CACHE = "/content/cache"
os.makedirs(CACHE, exist_ok=True)
VERSION = 2


# ---------------------------------------------------------------- du lieu
class _Rows(list):
    @property
    def column_names(self):
        return list(self[0].keys()) if self else []


def _pick(cols, *names):
    for n in names:
        if n in cols:
            return n
    raise KeyError(f"khong thay cot {names} trong {cols}")


def _jsonl(fname):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, fname, repo_type="dataset")
    return _Rows(json.loads(l) for l in open(p, encoding="utf-8") if l.strip())


def _qrels(rows):
    rq = _pick(rows.column_names, "query-id", "query_id", "qid")
    rd = _pick(rows.column_names, "corpus-id", "corpus_id", "doc_id")
    rs = "score" if "score" in rows.column_names else None
    out = defaultdict(dict)
    for row in rows:
        if rs is None or row[rs] > 0:
            out[str(row[rq])][str(row[rd])] = 1
    return dict(out)


def load_all():
    """Tra ve corpus, queries (moi cau co text), qrels_test, qrels_train."""
    c, q = _jsonl("corpus.jsonl"), _jsonl("queries.jsonl")
    cid, ctext = _pick(c.column_names, "_id", "id", "corpus-id"), _pick(c.column_names, "text", "content")
    ctitle = "title" if "title" in c.column_names else None
    corpus = {}
    for row in c:
        t = row[ctext] or ""
        if ctitle and row[ctitle]:
            t = f"{row[ctitle]}. {t}"
        corpus[str(row[cid])] = t
    qid, qtext = _pick(q.column_names, "_id", "id", "query-id"), _pick(q.column_names, "text", "query")
    queries = {str(row[qid]): row[qtext] for row in q}
    return corpus, queries, _qrels(_jsonl("qrels/test.jsonl")), _qrels(_jsonl("qrels/train.jsonl"))


def make_splits(queries, qrels_test, qrels_train, min_dev=200, dev_frac=0.3, seed=42):
    """Dev = cau hoi train co text (neu du); neu khong, chia ngau nhien tap test 30/70.
    Tra ve (qrels_by_split, mo_ta). Split 'all' = toan bo 788 cau test (de so voi luot 1)."""
    test_q = sorted(q for q in qrels_test if q in queries)
    train_q = sorted(q for q in qrels_train if q in queries and q not in qrels_test)
    if len(train_q) >= min_dev:
        dev = {q: qrels_train[q] for q in train_q}
        held = {q: qrels_test[q] for q in test_q}
        desc = f"dev = {len(dev)} cau train co text; held-out = toan bo {len(held)} cau test"
    else:
        rng = random.Random(seed)
        shuffled = test_q[:]
        rng.shuffle(shuffled)
        k = int(len(shuffled) * dev_frac)
        dev = {q: qrels_test[q] for q in shuffled[:k]}
        held = {q: qrels_test[q] for q in shuffled[k:]}
        desc = (f"chi {len(train_q)} cau train co text -> chia test {dev_frac:.0%}/{1 - dev_frac:.0%} "
                f"(seed {seed}): dev = {len(dev)}, held-out = {len(held)}")
    splits = {"all": {q: qrels_test[q] for q in test_q}, "dev": dev, "held": held}
    return splits, desc


# ---------------------------------------------------------------- tach tu
def seg_syllable(text):
    return re.findall(r"\w+", text.lower())


def seg_word(text):
    from pyvi import ViTokenizer
    return re.findall(r"\w+", ViTokenizer.tokenize(text).lower())


def segment_corpus(texts, mode, name):
    path = f"{CACHE}/seg_{mode}_{name}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    fn = seg_syllable if mode == "syllable" else seg_word
    if mode == "word":
        from multiprocessing import Pool
        with Pool(os.cpu_count()) as p:
            out = p.map(fn, texts, chunksize=256)
    else:
        out = [fn(t) for t in texts]
    pickle.dump(out, open(path, "wb"))
    return out


# ---------------------------------------------------------------- chunking
def _split_sentences(text):
    parts = re.split(r"(?<=[\.\;\:\?\!])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_corpus(corpus, strategy):
    """strategy: 'article' | 'fixed128' | 'fixed256' | 'sent256'.
    Tra ve (chunk_texts, chunk_doc_ids). Kich thuoc tinh theo am tiet (tu cach bang khoang trang)."""
    texts, owners = [], []
    for did, t in corpus.items():
        if strategy == "article":
            pieces = [t]
        elif strategy.startswith("fixed"):
            size = int(strategy[5:])
            step = int(size * 0.75)  # chong lap 25%
            w = t.split()
            pieces = [" ".join(w[i:i + size]) for i in range(0, max(len(w) - size, 0) + 1, step)] or [t]
            if len(w) > size and (len(w) - size) % step:
                pieces.append(" ".join(w[-size:]))
        elif strategy.startswith("sent"):
            size = int(strategy[4:])
            pieces, cur = [], []
            for s in _split_sentences(t):
                n = len(s.split())
                if cur and sum(len(x.split()) for x in cur) + n > size:
                    pieces.append(" ".join(cur))
                    cur = []
                cur.append(s)
            if cur:
                pieces.append(" ".join(cur))
        else:
            raise ValueError(strategy)
        texts.extend(pieces)
        owners.extend([did] * len(pieces))
    return texts, owners


def aggregate(chunk_scores_idx, owners, k=100):
    """Gop diem chunk ve dieu luat (max). chunk_scores_idx: list[(idx, score)] da sap xep giam dan."""
    best = {}
    for i, s in chunk_scores_idx:
        d = owners[i]
        if d not in best:
            best[d] = s
            if len(best) >= k:
                break
    return best


# ---------------------------------------------------------------- BM25
def bm25_search(chunk_texts, owners, queries, seg_mode, name, k=100, k1=1.2, b=0.75):
    """BM25 (idf kieu Lucene) cai bang ma tran thua de chay nhanh tren hang tram nghin chunk."""
    from sklearn.feature_extraction.text import CountVectorizer
    toks = segment_corpus(chunk_texts, seg_mode, name)
    cv = CountVectorizer(analyzer=lambda x: x)
    X = cv.fit_transform(toks).tocsr().astype(np.float32)
    N = X.shape[0]
    dl = np.asarray(X.sum(1)).ravel()
    df = np.bincount(X.indices, minlength=X.shape[1])
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    rows = np.repeat(np.arange(N), np.diff(X.indptr))
    tf = X.data
    X.data = idf[X.indices] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[rows] / dl.mean()))
    XT = X.T.tocsr()
    vocab = cv.vocabulary_
    fn = seg_syllable if seg_mode == "syllable" else seg_word
    run = {}
    for qid, q in queries.items():
        ids = [vocab[t] for t in fn(q) if t in vocab]
        sc = np.asarray(XT[ids].sum(0)).ravel() if ids else np.zeros(N, dtype=np.float32)
        top = np.argpartition(-sc, min(k * 5, N - 1))[: k * 5]
        top = top[np.argsort(-sc[top])]
        run[qid] = aggregate([(int(i), float(sc[i])) for i in top], owners, k)
    return run


# ---------------------------------------------------------------- dense
DENSE = {
    "e5-large": dict(hf="intfloat/multilingual-e5-large", qp="query: ", dp="passage: ", seg=False, maxlen=512),
    "bge-m3": dict(hf="BAAI/bge-m3", qp="", dp="", seg=False, maxlen=1024),
    "bkai-vi": dict(hf="bkai-foundation-models/vietnamese-bi-encoder", qp="", dp="", seg=True, maxlen=256),
    "aiteam-vi": dict(hf="AITeamVN/Vietnamese_Embedding", qp="", dp="", seg=False, maxlen=1024),
    # EmbeddingGemma khong ho tro float16 -> dung bfloat16; prompt dat qua prompt_name
    "gemma-emb-300m": dict(hf="google/embeddinggemma-300m", qp="", dp="", seg=False, maxlen=1024,
                           qpn="query", dpn="document", dtype="bf16"),
    "gemma-emb-2": dict(hf="google/embeddinggemma-2", qp="", dp="", seg=False, maxlen=1024,
                        qpn="SearchQuery", dpn="Document", dtype="bf16"),
}
# mo hinh "sach": khong co dau hieu da train tren Zalo Legal
CLEAN_DENSE = ["e5-large", "bge-m3", "gemma-emb-2", "gemma-emb-300m"]


def _seg_text(t):
    from pyvi import ViTokenizer
    return ViTokenizer.tokenize(t)


def dense_search(model_key, chunk_texts, owners, queries, name, k=100, bs=128):
    import torch
    from sentence_transformers import SentenceTransformer
    cfg = DENSE[model_key]
    path = f"{CACHE}/emb_{model_key}_{name}.npy"
    dtype = torch.bfloat16 if cfg.get("dtype") == "bf16" else torch.float16
    m = SentenceTransformer(cfg["hf"], device="cuda", model_kwargs={"torch_dtype": dtype})
    m.max_seq_length = cfg["maxlen"]
    prep = (lambda x: _seg_text(x)) if cfg["seg"] else (lambda x: x)
    if os.path.exists(path):
        E = np.load(path)
    else:
        if cfg["seg"]:
            from multiprocessing import Pool
            with Pool(os.cpu_count()) as p:
                segd = p.map(_seg_text, chunk_texts, chunksize=256)
            docs = [cfg["dp"] + t for t in segd]
        else:
            docs = [cfg["dp"] + t for t in chunk_texts]
        # sap theo do dai de batch nhanh hon
        order = np.argsort([len(d) for d in docs])
        Es = m.encode([docs[i] for i in order], batch_size=bs, normalize_embeddings=True,
                      show_progress_bar=False, convert_to_numpy=True, prompt_name=cfg.get("dpn"))
        E = np.empty_like(Es)
        E[order] = Es
        np.save(path, E.astype(np.float16))
    qids = list(queries)
    Q = m.encode([cfg["qp"] + prep(queries[q]) for q in qids], batch_size=bs, show_progress_bar=False,
                 normalize_embeddings=True, convert_to_numpy=True, prompt_name=cfg.get("qpn"))
    Q = Q.astype(np.float32)
    Et = torch.tensor(E, device="cuda", dtype=torch.float16)
    Qt = torch.tensor(Q, device="cuda", dtype=torch.float16)
    S = (Qt @ Et.T).float()
    vals, idx = S.topk(min(k * 5, S.shape[1]), dim=1)
    vals, idx = vals.cpu().numpy(), idx.cpu().numpy()
    run = {q: aggregate(list(zip(idx[j].tolist(), vals[j].tolist())), owners, k) for j, q in enumerate(qids)}
    del m, Et, Qt, S
    torch.cuda.empty_cache()
    return run


# ---------------------------------------------------------------- lai + rerank
def rrf(runs, k=60, top=100):
    out = {}
    for q in runs[0]:
        acc = defaultdict(float)
        for r in runs:
            for rank, d in enumerate(sorted(r.get(q, {}), key=r[q].get, reverse=True)):
                acc[d] += 1.0 / (k + rank + 1)
        out[q] = dict(sorted(acc.items(), key=lambda x: -x[1])[:top])
    return out


def rerank(run, corpus, queries, model="BAAI/bge-reranker-v2-m3", depth=50, maxlen=1024):
    from FlagEmbedding import FlagReranker
    rr = FlagReranker(model, use_fp16=True)
    keys, pairs = [], []
    for q, docs in run.items():
        for d in sorted(docs, key=docs.get, reverse=True)[:depth]:
            keys.append((q, d))
            pairs.append([queries[q], corpus[d]])
    sc = rr.compute_score(pairs, batch_size=128, max_length=maxlen)
    out = defaultdict(dict)
    for (q, d), s in zip(keys, sc):
        out[q][d] = float(s)
    del rr
    import torch
    torch.cuda.empty_cache()
    return dict(out)


# ---------------------------------------------------------------- danh gia
KS = (1, 5, 10, 20)


def per_query(run, qrels):
    """Diem tung cau hoi: {metric: {qid: value}}."""
    res = defaultdict(dict)
    for q, rel in qrels.items():
        ranked = sorted(run.get(q, {}), key=run.get(q, {}).get, reverse=True)
        for k in KS:
            res[f"R@{k}"][q] = len(set(ranked[:k]) & set(rel)) / len(rel)
        res["MRR@10"][q] = next((1 / (i + 1) for i, d in enumerate(ranked[:10]) if d in rel), 0.0)
        dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in rel)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))
        res["nDCG@10"][q] = dcg / idcg
    return dict(res)


def evaluate(run, qrels):
    return {m: round(100 * float(np.mean(list(v.values()))), 2) for m, v in per_query(run, qrels).items()}


RESULTS = []
PERQ = {}  # key -> {metric: {qid: value}} tren toan bo cau hoi co nhan (all + dev)


def log(key, run, splits, t0, **meta):
    """Ghi ket qua cho tung split; luu diem theo cau hoi de kiem dinh bootstrap."""
    allq = {**splits["dev"], **splits["all"]}
    PERQ[key] = per_query(run, allq)
    pickle.dump(PERQ, open(f"{CACHE}/perq.pkl", "wb"))
    rows = []
    for sp, qr in splits.items():
        r = {"key": key, "split": sp, **meta, "n": len(qr),
             **{m: round(100 * float(np.mean([PERQ[key][m][q] for q in qr])), 2) for m in PERQ[key]},
             "time_s": round(time.time() - t0, 1)}
        rows.append(r)
        RESULTS.append(r)
        with open(f"{CACHE}/results_v2.jsonl", "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{key:<48} " + "  ".join(f"{r['split']}:{r['nDCG@10']:.2f}" for r in rows) + f"  ({rows[0]['time_s']}s)")
    return rows


def best_on_dev(keys, metric="nDCG@10", splits=None):
    dev = splits["dev"]
    return max(keys, key=lambda k: np.mean([PERQ[k][metric][q] for q in dev]))


def bootstrap(key_a, key_b, qids, metric="nDCG@10", B=10000, seed=0):
    """Paired bootstrap: a - b tren cung tap cau hoi. Tra ve (delta, ci_lo, ci_hi, p_hai_phia)."""
    a = np.array([PERQ[key_a][metric][q] for q in qids])
    b = np.array([PERQ[key_b][metric][q] for q in qids])
    d = a - b
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return (round(100 * d.mean(), 2), round(100 * np.percentile(means, 2.5), 2),
            round(100 * np.percentile(means, 97.5), 2), round(min(p, 1.0), 4))
