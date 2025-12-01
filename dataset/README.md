---
dataset_info:
  features:
  - name: query
    dtype: string
  - name: query_id
    dtype: string
  - name: top_25_image_filenames
    sequence: int64
  - name: top_25_scores
    sequence: float64
  - name: relevant_image_filenames
    sequence: string
  splits:
  - name: esg_reports_human_labeled_v2
    num_bytes: 26264
    num_examples: 52
  download_size: 15617
  dataset_size: 26264
configs:
- config_name: default
  data_files:
  - split: esg_reports_human_labeled_v2
    path: data/esg_reports_human_labeled_v2-*
---

# Dataset Card for Vidore Reranker Benchmark : vidore_benchmark_2_esg_reports_human_labeled_v2_reranker_adapted

## Dataset Summary

This dataset provides a reranking benchmark based on the VIDORE V2 benchmark, designed to evaluate reranker models in a multimodal retrieval context. The dataset includes a corpus of image data, a set of natural language queries, and the top 25 retrievals (images) returned by a mid-performance multimodal retriever. This setup simulates a realistic retrieval environment where the reranker must learn to surface relevant items that may not already be ranked highly.

## Complete benchmark dataset list

The benchmark is composed of those datasets :
* UlrickBL/vidore_benchmark_economics_reports_v2_reranker_adapted (linked to vidore/economics_reports_v2 corpus)
* UlrickBL/vidore_benchmark_docvqa_reranker_adapted (linked to vidore/docvqa_test_subsampled corpus)
* UlrickBL/vidore_benchmark_2_biomedical_lectures_v2_reranker_adapted (linked to vidore/biomedical_lectures_v2 corpus)
* UlrickBL/vidore_benchmark_2_esg_reports_human_labeled_v2_reranker_adapted (linked to vidore/esg_reports_human_labeled_v2 corpus)
* UlrickBL/vidore_benchmark_2_esg_reports_v2_reranker_adapted (linked to vidore/esg_reports_v2 corpus)

## Dataset Motivation
The purpose of this benchmark is to:

* Evaluate rerankers independently of retriever performance by fixing the retriever outputs.

* Focus on the effectiveness of rerankers in identifying relevant samples from mid-quality retrieval sets.

* Provide detailed statistics on the retrieval and relevance structure to better understand model behavior.

By using a retriever with known mid-level performance on the VIDORE v1 leaderboard, this benchmark offers a challenging but meaningful setting to test reranking capabilities.

The retriever used is : _Alibaba-NLP/gme-Qwen2-VL-2B-Instruct_ (top 23 - 87.8 accuracy)

## Dataset Structure
Each sample in the dataset is associated with:

* query : text query

* query_id : corresponding query id of the original dataset

* top_25_image_filenames : ordered list of retrieved image by the retriever

* top_25_scores : corresponding list with the scores of the top 25 images

* relevant_image_filenames : list of the true labels / relevant images of the original dataset

# How the Dataset Was Constructed
Retriever Used: A multimodal retriever _Alibaba-NLP/gme-Qwen2-VL-2B-Instruct_

The retriever was used to embed the full corpus of images of the associated dataset (vidore/esg_reports_human_labeled_v2).

For each query, the retriever computed similarity and returned the top 25 most similar corpus images.

These 25 candidates were labeled using the ground-truth relevance annotations from VIDORE v2.

Only retrieved items are considered during evaluation — non-retrieved relevant samples are ignored to focus on reranking.

## Dataset Statistics
Here are some key dataset statistics:

| Metric                                               | Value       |
| ---------------------------------------------------- | ----------- |
| Number of queries                                    | 52 |
| Corpus size                                          | 1540 |
| Average # relevant images per query                  | 2.46 |
| Average # retrieved relevant images in top 25        | 1.73 |
| % of queries with at least one relevant retrieved    | 91.25% |
| Avg. position of **first** relevant image            | 3.53 |
| Avg. position of **last** relevant image             | 6.82 |
| NGCD\@5 (Normalized Gain Cumulative Discounted at 5) | 0.6424 |

## Use this dataset

To use this dataset, you can create pairs of queries and images by linking a query and an image from the corpus of the top 25 list and score it with your model to rerank the top 25 list.
