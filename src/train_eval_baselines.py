
import os, math, argparse, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from tqdm import tqdm

def pick_device(arg_device: str = "auto"):
    try:
        import torch
        if arg_device and arg_device.lower() in ["cpu", "cuda", "mps"]:
            return arg_device.lower()
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"

def load_data(items_csv, train_csv, val_csv, test_csv):
    items = pd.read_csv(items_csv)
    train = pd.read_csv(train_csv)
    val   = pd.read_csv(val_csv)
    test  = pd.read_csv(test_csv)
    return items, train, val, test

def build_index(items):
    item_id_to_idx = {iid:i for i, iid in enumerate(items['item_id'].tolist())}
    idx_to_item_id = {i:iid for iid,i in item_id_to_idx.items()}
    return item_id_to_idx, idx_to_item_id

def user_seen_map(train, item_id_to_idx):
    seen = train.groupby('user_id')['item_id'].apply(list).to_dict()
    return {u: set(item_id_to_idx[i] for i in its if i in item_id_to_idx) for u,its in seen.items()}

def eval_rank(item_vecs, items, train, test, topk=10, alpha=1.0, beta=0.0, popularity=None):
    # item_vecs: dense np array (n_items, d) already L2-normalized row-wise
    item_id_to_idx, idx_to_item_id = build_index(items)
    seen_map = user_seen_map(train, item_id_to_idx)
    pop_vec = None
    if popularity is not None:
        pop_counts = train['item_id'].value_counts()
        pop_vec = np.array([pop_counts.get(idx_to_item_id[i], 0) for i in range(len(idx_to_item_id))], dtype=np.float32)
        pop_vec = (pop_vec - pop_vec.mean()) / (pop_vec.std() + 1e-9)

    results = {'Recall@10':0.0, 'MRR@10':0.0, 'NDCG@10':0.0, 'n_eval':0}
    work = test[test['user_id'].isin(seen_map.keys())].copy()
    for _, row in tqdm(work.iterrows(), total=len(work)):
        u = row['user_id']; tgt = row['test_item']
        if tgt not in item_id_to_idx: continue
        seen = list(seen_map.get(u, []))
        if not seen: continue
        prof = item_vecs[seen].mean(axis=0)
        prof /= (np.linalg.norm(prof) + 1e-9)
        # cosine (rows L2 normalized)
        scores = item_vecs @ prof
        scores[np.array(seen, dtype=int)] = -1e9
        if pop_vec is not None and beta>0:
            scores = alpha*scores + beta*pop_vec
        top_idx = np.argpartition(scores, -topk)[-topk:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        tgt_idx = item_id_to_idx[tgt]
        if tgt_idx in top_idx:
            rank = np.where(top_idx==tgt_idx)[0][0] + 1
            results['Recall@10'] += 1.0
            results['MRR@10'] += 1.0 / rank
            results['NDCG@10'] += 1.0 / np.log2(rank+1)
        results['n_eval'] += 1
    n = max(results['n_eval'], 1)
    for k in ['Recall@10','MRR@10','NDCG@10']:
        results[k] /= n
    return results

def ensure_l2(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms

def build_tfidf(items):
    texts = (items['title'].fillna('') + '. ' + items['description'].fillna('')).tolist()
    vec = TfidfVectorizer(max_features=60000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    X = normalize(X, norm='l2', axis=1)
    return X.toarray().astype(np.float32)

def _should_fp16(device: str) -> bool:
    return device in ("cuda", "mps")

def build_phobert(items, device='cpu'):
    from transformers import AutoTokenizer, AutoModel
    import torch
    def mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-9))
    tok = AutoTokenizer.from_pretrained('vinai/phobert-base')
    mdl = AutoModel.from_pretrained('vinai/phobert-base')
    use_fp16 = _should_fp16(device)
    if use_fp16:
        mdl = mdl.to(device).to(dtype=torch.float16)
    else:
        mdl = mdl.to(device)
    mdl.eval()
    texts = (items['title'].fillna('') + '. ' + items['description'].fillna('')).tolist()
    embs = []
    bs = 16 if device == 'cuda' else (12 if device == 'mps' else 8)
    max_len = 256
    with (torch.no_grad()):
        for i in tqdm(range(0, len(texts), bs)):
            batch = texts[i:i+bs]
            inp = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)
            out = mdl(**inp)
            vec = mean_pool(out.last_hidden_state, inp['attention_mask'])
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            vec = vec.cpu().numpy().astype(np.float32)
            embs.append(vec)
    return np.vstack(embs)

def build_clip_text(items, model_name='openai/clip-vit-base-patch32', device='cpu'):
    from transformers import AutoTokenizer, AutoModel
    import torch
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()
    texts = (items['title'].fillna('') + '. ' + items['description'].fillna('')).tolist()
    embs = []
    bs, max_len = 32, 256
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), bs)):
            batch = texts[i:i+bs]
            inp = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)
            out = mdl(**inp)
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                vec = out.pooler_output
            else:
                vec = out.last_hidden_state[:,0,:]
            vec = torch.nn.functional.normalize(vec, p=2, dim=1).cpu().numpy().astype(np.float32)
            embs.append(vec)
    return np.vstack(embs)

def build_openai(items, model='text-embedding-ada-002', batch_size=128):
    from openai import OpenAI
    client = OpenAI()
    texts = (items['title'].fillna('') + '. ' + items['description'].fillna('')).tolist()
    embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        vecs = [v/(np.linalg.norm(v)+1e-9) for v in vecs]
        embs.append(np.vstack(vecs))
    return np.vstack(embs).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items_csv', default='items_prepared.csv')
    ap.add_argument('--train_csv', default='interactions_train.csv')
    ap.add_argument('--val_csv',   default='interactions_val.csv')
    ap.add_argument('--test_csv',  default='interactions_test.csv')
    # ap.add_argument('--device', default='cpu')
    ap.add_argument('--run', nargs='+', default=['tfidf','phobert','clip','ada2'])
    ap.add_argument('--beta', type=float, default=0.2, help='popularity weight')
    ap.add_argument('--subset', type=int, default=0, help='if >0, evaluate on a random subset of test users for speed')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    items, train, val, test = load_data(args.items_csv, args.train_csv, args.val_csv, args.test_csv)
    # Build item popularity from train (z-score will be done inside eval_rank)
    results = {}

    # Optionally downsample users for speed
    if args.subset and args.subset>0:
        # keep only subset users that have at least one train interaction
        eligible = set(train['user_id'].unique().tolist())
        test = test[test['user_id'].isin(eligible)].copy()
        samp = np.random.default_rng(0).choice(test['user_id'].unique(), size=min(args.subset, len(test['user_id'].unique())), replace=False)
        test = test[test['user_id'].isin(set(samp))].copy()

    if 'tfidf' in args.run:
        print('>> Building TF-IDF embeddings')
        tfidf_vecs = build_tfidf(items)
        tfidf_vecs = ensure_l2(tfidf_vecs)
        results['CB-TFIDF'] = eval_rank(tfidf_vecs, items, train, test, topk=10, alpha=1.0, beta=0.0)
        results['CB-TFIDF+Popularity'] = eval_rank(tfidf_vecs, items, train, test, topk=10, alpha=1.0, beta=args.beta, popularity=True)

    if 'phobert' in args.run:
        print('>> Building PhoBERT embeddings')
        phobert_vecs = build_phobert(items, device=args.device)
        phobert_vecs = ensure_l2(phobert_vecs)
        results['CB-PhoBERT'] = eval_rank(phobert_vecs, items, train, test, topk=10, alpha=1.0, beta=0.0)
        results['CB-PhoBERT+Popularity'] = eval_rank(phobert_vecs, items, train, test, topk=10, alpha=1.0, beta=args.beta, popularity=True)

    if 'clip' in args.run:
        print('>> Building BLIP/CLIP text embeddings')
        clip_vecs = build_clip_text(items, device=args.device)
        clip_vecs = ensure_l2(clip_vecs)
        results['CB-BLIP'] = eval_rank(clip_vecs, items, train, test, topk=10, alpha=1.0, beta=0.0)
        results['CB-BLIP+Popularity'] = eval_rank(clip_vecs, items, train, test, topk=10, alpha=1.0, beta=args.beta, popularity=True)

    if 'ada2' in args.run:
        print('>> Building OpenAI Ada2 embeddings')
        ada_vecs = build_openai(items, model='text-embedding-ada-002')
        results['CB-Ada2'] = eval_rank(ada_vecs, items, train, test, topk=10, alpha=1.0, beta=0.0)
        results['CB-Ada2+Popularity'] = eval_rank(ada_vecs, items, train, test, topk=10, alpha=1.0, beta=args.beta, popularity=True)

    df = pd.DataFrame(results).T
    print('\\n=== RESULTS (top-10) ===')
    print(df)
    df.to_csv('results_baselines.csv')
    print('\\nSaved results to results_baselines.csv')

if __name__ == '__main__':
    main()
