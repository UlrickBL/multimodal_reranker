from typing import List,Dict,Any
import torchvision
from tqdm import tqdm
from datasets import Dataset

def prepare_multiple_pairs_for_reranker(
    pairs: List[Dict[str, Any]], # Expects a list of dicts, e.g., [{'query': '...', 'image': img}, ...]
    processor
) -> Dict[str, torch.Tensor]:

    queries_list = [pair["query"] for pair in pairs]
    images_list = [pair["image"] for pair in pairs]

    queries_prepared = []
    for question in queries_list :
        queries_prepared += ["<|im_start|>system\nYou will be given an picture and a query. Answer 'Yes' if the answer to the query can be found in the picture, else 'No'<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query : "+question+" \nAre the picture and query related ?<|im_end|>\n<|im_start|>assistant\n"]

    inputs = processor(
                    text=queries_prepared,
                    images=images_list,
                    padding=True,
                    return_tensors="pt",
                )

    return inputs.to("cuda")

def find_by_value(example):
    return example["corpus-id"] in images_path
      
def load_image_from_path(images_path):
    return  image_dataset["test"].filter(find_by_value)["image"]

def load_image_from_name(image_path):
    return [image_filename_to_image.get(filename) for filename in image_path]

def process_example(example,batch_size=32):
    query = example["query"]
    paths = example["top_25_image_filenames"]
    label_paths = set(example["relevant_image_filenames"])  # multiple true labels

    candidate_images = load_image_from_name(paths)

    pairs = [{"query": query, "image": doc} for doc in candidate_images]
    
    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i : i + batch_size]
        inputs = prepare_multiple_pairs_for_reranker(pairs=batch_pairs, processor=processor)
        batch_scores = model(**inputs).to(torch.float32).cpu().numpy()
        all_scores.extend(batch_scores)

    scores = np.array(all_scores)
    
    ranked_items = sorted(zip(scores, paths), key=lambda item: item[0], reverse=True)
    ranked_scores, ranked_image_paths = zip(*ranked_items)
    
    relevance = [1.0 if str(path) in label_paths else 0.0 for path in ranked_image_paths]
    
    top5_scores = np.asarray([ranked_scores[:5]])
    top5_relevance = np.asarray([relevance[:5]])
    
    if top5_relevance.sum() > 0:
        ndcg_5 = ndcg_score(y_true=top5_relevance, y_score=top5_scores)
    else:
        ndcg_5 = 0.0

    return {
        "ndcg_5": ndcg_5,
        "ranked_scores":ranked_scores,
        "ranked_image_paths":ranked_image_paths,
        "ranked_docs": top5_relevance,
        "relevance": relevance
    }

if name=="__main__" :
  
  reranker_dataset = load_dataset("UlrickBL/vidore_benchmark_2_esg_qa_reranker_adapted")
  image_dataset = load_dataset("vidore/economics_reports_v2",'corpus')
  reranker_dataset = reranker_dataset["economics_reports_v2"][:]

  reranker_dataset = Dataset.from_dict(reranker_dataset)

  image_filename_to_image = {
    example["corpus-id"]: example["image"]
    for example in image_dataset["test"]
  }
  
  for idx, example in tqdm(enumerate(reranker_dataset), total=len(reranker_dataset), desc="Processing examples"):
      try:
          out = process_example(example)
          merged = {**example, **out}
          results.append(merged)
      except Exception as e:
          print(f"Error at index {idx}: {e}")
        
  processed_dataset = Dataset.from_list(results)
  
  average_ndcg_5 = np.nanmean(processed_dataset["ndcg_5"])
  print(f"Average NDCG@5: {average_ndcg_5}")

  processed_dataset.to_pandas().to_csv("economics_reports_v2_qwen_yes.csv")
