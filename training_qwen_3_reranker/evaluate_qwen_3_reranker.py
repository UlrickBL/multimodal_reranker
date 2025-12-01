from datasets import Dataset
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from multimodal_reranker.model.qwen_3_reranker import Qwen3Reranker

min_pixels = 256*28*28
max_pixels = 720*28*28
model_qwen = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2", device_map="cuda"
).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct",max_pixels=max_pixels)

base = PeftModel.from_pretrained(model_qwen, "UlrickBL/qwen_vl_reranker_adapter_V3")

model = Qwen3Reranker(base_model=base,processor=processor)

model=model.to("cuda")

reranker_dataset = load_dataset("UlrickBL/REAL-MM-RAG_FinSlides_BEIR_reranker_adapted")
image_dataset = load_dataset("ibm-research/REAL-MM-RAG_FinSlides_BEIR",'corpus')
reranker_dataset = reranker_dataset["REAL_MM_RAG_FinSlides_BEIR"][:]
reranker_dataset = Dataset.from_dict(reranker_dataset)


image_filename_to_image = {
    example["corpus-id"]: example["image"]
    for example in image_dataset["test"]
}

def load_image_from_path(images_path):
    def find_by_value(example):
        return example["corpus-id"] in images_path
    return  image_dataset["test"].filter(find_by_value)["image"]

def load_image_from_name(image_path):
    return [image_filename_to_image.get(filename) for filename in image_path]

def process_example(example):
    query = example["query"]
    paths = example["top_25_image_filenames"]
    label_paths = set(example["relevant_image_filenames"])

    candidate_images = load_image_from_name(paths)

    pairs = [[query, doc] for doc in candidate_images]
    scores = model.compute_score(pairs,batch_size=25)
    
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
    
processed_dataset = reranker_dataset.map(process_example)

average_ndcg_5 = np.nanmean(processed_dataset["ndcg_5"])
print(f"Average NDCG@5: {average_ndcg_5}")

processed_dataset.to_pandas().to_csv("qwen3MM_reranker.csv")