
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy import sparse

items = pd.read_csv("items_prepared.csv")
train_df = pd.read_csv("interactions_train.csv")
test_df  = pd.read_csv("interactions_test.csv")

texts = (items["title"].fillna("") + ". " + items["description"].fillna("")).tolist()
tfidf = TfidfVectorizer(max_features=60000, ngram_range=(1,2))
X = tfidf.fit_transform(texts)
X = normalize(X, norm="l2", axis=1)
tfidf_embs = X.toarray().astype(np.float32)

# Save embeddings and vocab
sparse.save_npz("emb_tfidf_items.npz", X)
with open("tfidf_vectorizer_vocabs.txt","w",encoding="utf-8") as f:
    for t in tfidf.get_feature_names_out(): f.write(t+"\n")

# Build index maps
item_id_to_idx = {iid:i for i, iid in enumerate(items["item_id"].tolist())}
idx_to_item_id = {i:iid for iid,i in item_id_to_idx.items()}
train_seen = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
user_seen_idxs = {u:set(item_id_to_idx[i] for i in its if i in item_id_to_idx) for u,its in train_seen.items()}

def eval_rank(item_vecs, items, train, test, topk=10, alpha=1.0, beta=0.0):
    pop_counts = train["item_id"].value_counts()
    pop_vec = np.array([pop_counts.get(idx_to_item_id[i], 0) for i in range(len(idx_to_item_id))], dtype=np.float32)
    pop_vec = (pop_vec - pop_vec.mean()) / (pop_vec.std()+1e-9)
    results = {"Recall@10":0,"MRR@10":0,"NDCG@10":0,"n_eval":0}
    for _, row in test.iterrows():
        u, tgt = row["user_id"], row["test_item"]
        if u not in user_seen_idxs or tgt not in item_id_to_idx: continue
        seen = list(user_seen_idxs[u]); 
        if not seen: continue
        prof = item_vecs[seen].mean(axis=0)
        prof /= (np.linalg.norm(prof)+1e-9)
        scores = item_vecs @ prof
        scores[np.array(seen,dtype=int)] = -1e9
        if beta>0: scores = alpha*scores + beta*pop_vec
        top_idx = np.argpartition(scores, -topk)[-topk:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        tgt_idx = item_id_to_idx[tgt]
        if tgt_idx in top_idx:
            rank = np.where(top_idx==tgt_idx)[0][0]+1
            results["Recall@10"]+=1; results["MRR@10"]+=1.0/rank; results["NDCG@10"]+=1.0/np.log2(rank+1)
        results["n_eval"]+=1
    n= max(results["n_eval"],1)
    for k in ["Recall@10","MRR@10","NDCG@10"]: results[k]/=n
    return results

print("TF-IDF results:")
print(eval_rank(tfidf_embs, items, train_df, test_df, topk=10, alpha=1.0, beta=0.0))
print("TF-IDF+Popularity results:")
print(eval_rank(tfidf_embs, items, train_df, test_df, topk=10, alpha=1.0, beta=0.2))
