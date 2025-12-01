from multimodal_reranker.dataset.gme_qwen_2_vl import GmeQwen2VL
from itertools import islice
from PIL import Image
from datasets import load_dataset, Dataset
from torchvision import transforms
from tqdm import tqdm
import torch
from itertools import islice
from datasets import load_dataset
from tqdm import tqdm
import torch

def batched(iterable, batch_size):
    """Batch an iterable (like a streamed dataset)."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch

def treat_dataset_full_sim_streamed(dataset_name, device="cuda", chunk_size=6, top_k=25):
    dataset_path = "vidore/" + dataset_name
    split = "test"

    ds_queries = load_dataset(dataset_path, "queries", split=split)
    ds_corpus = load_dataset(dataset_path, "corpus", split=split, streaming=True)
    ds_qrels = load_dataset(dataset_path, "qrels", split=split)
  
    all_queries = ds_queries["query"]
    query_ids = [f"query-{split}-{qid}" for qid in ds_queries["query-id"]]

    query_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_queries), chunk_size)):
            chunk = all_queries[i:i + chunk_size]
            emb = gme.get_text_embeddings(texts=chunk, instruction='Find an image that matches the given text.').cpu()
            query_embeddings.append(emb)
    query_embeddings = torch.cat(query_embeddings, dim=0).to(device)

    corpus_embeddings = []
    all_corpus_filenames = []
    corpus_ids = []

    with torch.no_grad():
        for batch in tqdm(batched(ds_corpus, chunk_size)):
            images = [item["image"] for item in batch]
            filenames = [item["corpus-id"] for item in batch]
            emb = gme.get_image_embeddings(images=images, is_query=False).cpu()
            corpus_embeddings.append(emb)

            all_corpus_filenames.extend(filenames)
            corpus_ids.extend([f"corpus-{split}-{fid}" for fid in filenames])

    corpus_embeddings = torch.cat(corpus_embeddings, dim=0).to(device)

    similarity = query_embeddings @ corpus_embeddings.T  # (N_queries, N_corpus)
    top_scores, top_indices = torch.topk(similarity, k=top_k, dim=1)

    qrels = {}
    for row in ds_qrels:
        qid = f"query-{split}-{row['query-id']}"
        did = f"corpus-{split}-{row['corpus-id']}"
        if row["score"] > 0:
            qrels.setdefault(qid, set()).add(did)

    results = []
    for i, qid in enumerate(query_ids):
        top_idx = top_indices[i].tolist()
        top_score = top_scores[i].tolist()
        top_filenames = [all_corpus_filenames[j] for j in top_idx]

        relevant_filenames = [cid.split(f"corpus-{split}-")[1] for cid in qrels.get(qid, [])]

        results.append({
            "query": all_queries[i],
            "query_id": qid,
            "top_25_image_filenames": top_filenames,
            "top_25_scores": top_score,
            "relevant_image_filenames": relevant_filenames,
        })

    return results

if name == "__main__":
  gme = GmeQwen2VL("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")

  list_datasets = ["esg_reports_v2"]
  full_dataset_dict = {}
  
  for dataset_name in tqdm(list_datasets) :
      print(f"\n--- Processing dataset: {dataset_name} ---")
      full_dataset_dict[dataset_name] = Dataset.from_list(treat_dataset_full_sim_streamed(dataset_name))

  dataset = DatasetDict(full_dataset_dict)
