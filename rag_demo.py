
"""
Minimal RAG retrieval + rerank demo for I Ching dream motifs.
- Loads CSV (hexagrams + motif_map)
- Embeds text via simple term-frequency vectors (numpy)
- Retrieves top-k hexagrams for a user query
- Reranks by (a) cosine similarity (b) valence closeness if provided
- Uses motif prior if a known motif keyword is detected
No external deps beyond numpy/pandas.
"""
import json, re, math
import numpy as np
import pandas as pd

def tokenize(text):
    if text is None: return []
    # simple word/char mix tokenizer to handle CN/EN roughly
    text = str(text).lower()
    # split by non-word, also keep CJK chars
    tokens = re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', text)
    return tokens

def build_vocab(corpus_texts, min_freq=1):
    from collections import Counter
    c = Counter()
    for t in corpus_texts:
        c.update(tokenize(t))
    vocab = [w for w,f in c.items() if f>=min_freq]
    vocab_index = {w:i for i,w in enumerate(vocab)}
    return vocab, vocab_index

def embed(text, vocab_index):
    vec = np.zeros(len(vocab_index), dtype=float)
    for tok in tokenize(text):
        idx = vocab_index.get(tok)
        if idx is not None:
            vec[idx] += 1.0
    # l2 normalize
    n = np.linalg.norm(vec)
    if n>0: vec = vec/n
    return vec

def cosine(a,b):
    # assume inputs are l2-normed
    return float(np.dot(a,b))

def load_data(hex_csv, motif_csv):
    hex_df = pd.read_csv(hex_csv)
    motif_df = pd.read_csv(motif_csv)
    return hex_df, motif_df

def build_index(hex_df):
    # concatenate key text fields for retrieval
    docs = (hex_df["name_cn"].fillna('') + ' ' +
            hex_df["theme"].fillna('') + ' ' +
            hex_df["keywords"].fillna('') + ' ' +
            hex_df["emotions"].fillna(''))
    vocab, vocab_index = build_vocab(docs.tolist())
    mat = np.vstack([embed(d, vocab_index) for d in docs])
    return vocab_index, mat

def detect_motif_prior(query, motif_df):
    # crude match: if motif word occurs in query, return its suggested hexes
    hit = None
    for _,row in motif_df.iterrows():
        if str(row["motif"]) in query:
            hit = row
            break
    if hit is None:
        return set()
    try:
        hexes = set(json.loads(hit["suggested_hexagrams"]))
    except Exception:
        hexes = set()
    return hexes

def search(hex_df, motif_df, vocab_index, doc_mat, query, desired_valence=None, topk=5):
    qvec = embed(query, vocab_index)
    # cosine scores
    sims = doc_mat @ qvec
    # initial ranking
    order = np.argsort(-sims)
    # rerank with valence + motif prior
    motif_prior = detect_motif_prior(query, motif_df)
    results = []
    for rank, idx in enumerate(order[:50]):  # consider top 50 for rerank
        row = hex_df.iloc[idx]
        score = sims[idx]
        # add valence rerank: closer to desired_valence is better
        if desired_valence is not None and not np.isnan(row["auspiciousness_score"]):
            score += 0.15 * (1.0 - abs(row["auspiciousness_score"] - desired_valence))
        # motif prior boost
        if int(row["hex_id"]) in motif_prior:
            score += 0.2
        results.append((float(score), idx))
    results.sort(key=lambda x: -x[0])
    top = results[:topk]
    out = []
    for s, idx in top:
        r = hex_df.iloc[idx].to_dict()
        r["rag_score"] = round(float(s), 4)
        out.append(r)
    return out

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--hex", default="iching_hexagrams_64.csv")
    ap.add_argument("--motif", default="iching_motifs_mapping.csv")
    ap.add_argument("--query", required=True, help="dream snippet or motif, e.g., 'Sorry please wait'")
    ap.add_argument("--valence", type=float, default=None, help="desired valence in [-1,1], e.g., -0.3")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    hex_df, motif_df = load_data(args.hex, args.motif)
    vocab_index, doc_mat = build_index(hex_df)

    results = search(hex_df, motif_df, vocab_index, doc_mat,
                     query=args.query, desired_valence=args.valence, topk=args.topk)
    for r in results:
        print(f"[{r['hex_id']:02d}] {r['name_cn']} | theme={r['theme']} | score={r['rag_score']} | val={r['auspiciousness_score']}")
