# Case Study for the TextWave System
## Adam Lane

TextWave is a modular question-answering (QA) system designed to evaluate both retrieval and generative components across varying document preprocessing strategies and LLM sizes. It incorporates a curated question set (`questions.tsv`) with difficulty annotations, enabling fine-grained performance analysis. The project explores trade-offs across chunking methods, retrieval strategies (Brute Force, HNSW, LSH), re-ranking pipelines, and standalone LLM-based answer generation. This study documents empirical findings and comparative evaluations for each stage of the QA pipeline.

## **Task 1: Fixed-Length vs. Sentence-Based Chunking for QA Retrieval**

In Task 1, we evaluated the retrieval performance of two chunking strategies (sentence-based and fixed-length) using the `all-MiniLM-L6-v2` embedding model and the Brute Force index. Our results show that fixed-length chunking consistently outperforms sentence chunking across all evaluated metrics. The best performance was achieved with a chunk size of 128 characters and an overlap of 32, yielding the highest Mean Reciprocal Rank (MRR) of 0.8122 and top precision scores (P\@1 = 0.7617). In contrast, sentence chunking reached a maximum MRR of 0.7811 using 5 sentence chunks with minimal overlap. We also observed that moderate chunk sizes (128 to 256 characters) strike a strong balance between contextual richness and specificity, while larger chunks (around 512 characters) slightly degrade performance, likely due to diluted focus.

Additionally, increased overlap generally improves results, as it helps capture transitional context between adjacent chunks. However, we also found that the gains diminish at higher chunk sizes. Sentence chunking, while semantically clean, was more sensitive to chunk length and offered lower consistency in chunk size due to the variability in sentence lengths. Overall, our findings suggest that for embedding-based retrieval tasks, fixed-length chunking with moderate size and overlap provides the most effective configuration when using this model and index type.

<div align="center">

| **Sentences** | **Overlap** | **MRR** | **P\@1** | **P\@3** | **P\@5** |
| ------------- | ----------- | ------- | -------- | -------- | -------- |
| 5             | 1           | 0.7811  | 0.7350   | 0.7099   | 0.6831   |
| 5             | 3           | 0.7781  | 0.7302   | 0.7407   | 0.7234   |
| 10            | 1           | 0.7637  | 0.7169   | 0.6845   | 0.6650   |
| 10            | 3           | 0.7715  | 0.7293   | 0.7026   | 0.6772   |
| 15            | 1           | 0.7534  | 0.7140   | 0.6657   | 0.6480   |
| 15            | 3           | 0.7574  | 0.7121   | 0.6692   | 0.6479   |

</div>

<p align="center"><b>Table 1: Tabulated summary of retrieval results using Sentence Chunking, varying by number of sentences and overlap.</b></p>

<div align="center">

| **Chunk Size**  | **Overlap** | **MRR** | **P\@1** | **P\@3** | **P\@5** |
| --------------- | ----------- | ------- | -------- | -------- | -------- |
| 128             | 16          | 0.7955  | 0.7455   | 0.6988   | 0.6643   |
| 128             | 32          | 0.8122  | 0.7617   | 0.7204   | 0.6906   |
| 256             | 16          | 0.7965  | 0.7541   | 0.7067   | 0.6787   |
| 256             | 32          | 0.7942  | 0.7502   | 0.7080   | 0.6799   |
| 512             | 16          | 0.7808  | 0.7369   | 0.7010   | 0.6776   |
| 512             | 32          | 0.7784  | 0.7340   | 0.6981   | 0.6734   |

</div>

<p align="center"><b>Table 2: Tabulated summary of retrieval results using Fixed-Length Chunking, varying by number of chunk size and overlap.</b></p>

<p align="center">
  <img src="external_plots/Task%201%20-%20Figure%201.png" width="750"/>
</p>

<p align="center"><b>Figure 1: Retrieval Performance of Sentence Chunking.</b></p>

<p align="center">
  <img src="external_plots/Task%201%20-%20Figure%202.png" width="750"/>
</p>

<p align="center"><b>Figure 2: Retrieval Performance of Fixed-Length Chunking.</b></p>

## **Task 2: Impact of Indexing Strategies on Retrieval Performance**

In our evaluation of fixed-length chunking strategies using the `all-MiniLM-L6-v2` embedding model, we compared two indexing methods (HNSW and LSH) across varying chunk sizes and overlaps. The results clearly indicate that HNSW significantly outperforms LSH across all configurations, with HNSW achieving a maximum MRR of 0.8215 and P\@1 of 0.7827 (with the parameters being a chunk size of 128 and overlap of 32). This is compared to LSH’s best MRR of 0.1915 and P\@1 of 0.1258. HNSW also proved more robust to changes in chunk size, maintaining strong performance across all settings, while LSH performance varied more widely and remained consistently lower.

Additionally, increasing the overlap from 16 to 32 characters consistently improved retrieval accuracy in both indexing methods, though gains were more pronounced with HNSW. Larger chunk sizes tended to benefit LSH up to a point, but came with diminishing returns and occasional drops in precision. Overall, these findings suggest that HNSW indexing combined with a chunk size of 128 and overlap of 32 offers the most effective configuration for high quality retrieval in this setup.

<div align="center">

| **Chunk Size** | **Overlap** | **MRR** | **P\@1** | **P\@3** | **P\@5** |
| -------------- | ----------- | ------- | -------- | -------- | -------- |
| 128            | 16          | 0.1364  | 0.0925   | 0.0969   | 0.0967   |
| 128            | 32          | 0.1494  | 0.0915   | 0.1106   | 0.1157   |
| 256            | 16          | 0.1852  | 0.1249   | 0.1217   | 0.1218   |
| 256            | 32          | 0.1696  | 0.1087   | 0.1227   | 0.1142   |
| 512            | 16          | 0.1787  | 0.1153   | 0.1077   | 0.0972   |
| 512            | 32          | 0.1915  | 0.1258   | 0.1274   | 0.1150   |

</div>

<p align="center"><b>Table 3: Tabulated summary of retrieval results using Fixed-Length Chunking, varying by number of chunk size and overlap. The indexing method is LSH.</b></p>

<div align="center">

| **Chunk Size** | **Overlap** | **MRR** | **P\@1** | **P\@3** | **P\@5** |
| -------------- | ----------- | ------- | -------- | -------- | -------- |
| 128            | 16          | 0.8072  | 0.7722   | 0.6984   | 0.6641   |
| 128            | 32          | 0.8215  | 0.7827   | 0.7181   | 0.6890   |
| 256            | 16          | 0.7996  | 0.7617   | 0.7032   | 0.6770   |
| 256            | 32          | 0.8061  | 0.7741   | 0.7096   | 0.6801   |
| 512            | 16          | 0.7851  | 0.7502   | 0.7026   | 0.6782   |
| 512            | 32          | 0.7892  | 0.7541   | 0.7007   | 0.6734   |

</div>

<p align="center"><b>Table 4: Tabulated summary of retrieval results using Fixed-Length Chunking, varying by number of chunk size and overlap. The indexing method is HNSW.</b></p>

<p align="center">
  <img src="external_plots/Task%202%20-%20Figure%201.png" width="750"/>
</p>

<p align="center"><b>Figure 3: Retrieval Performance of Fixed-Length Chunking and LSH.</b></p>

<p align="center">
  <img src="external_plots/Task%202%20-%20Figure%202.png" width="750"/>
</p>

<p align="center"><b>Figure 4: Retrieval Performance of Fixed-Length Chunking and HNSW.</b></p>

## **Task 3: Improving Retrieval with Re-Ranking Pipelines**

To evaluate the impact of reranking strategies on retrieval performance, we tested six reranker types using our best performing chunking configuration from Task 2 (chunk size of 128 and overlap of 32). The baseline system (no reranking) achieved a strong MRR of 0.8215 and P\@1 of 0.7827, demonstrating the effectiveness of fixed-length chunking with `MiniLM` embeddings. However, all rerankers (except for Sequential) improved upon these results, with the Hybrid reranker emerging as the most effective strategy, yielding the highest MRR (0.8311), P\@1 (0.8027), and P\@3 (0.7305).

TFIDF and Cross-encoder also performed well, offering competitive accuracy with slightly lower computational cost compared to the Hybrid method. Interestingly, the Bag-of-Words (Bow) reranker performed nearly on par with Cross-encoder, despite its simplicity. In contrast, the Sequential reranker underperformed, achieving the lowest MRR (0.8051) and P\@5 (0.6780), suggesting it may not be well-suited to our current architecture or question set. Overall, these results highlight the substantial benefit of reranking, with Hybrid standing out as the most robust option.

<div align="center">

| **Chunk Size** | **Overlap** | **Reranker Type** | **MRR**    | **P\@1**   | **P\@3**   | **P\@5** |
| -------------- | ----------- | ----------------- | ---------- | ---------- | ---------- | -------- |
| 128            | 32          | None              | 0.8215     | 0.7827     | 0.7181     | 0.6890   |
| 128            | 32          | Cross-encoder     | 0.8234     | 0.7846     | 0.7242     | 0.6890   |
| 128            | 32          | Bow               | 0.8232     | 0.7836     | 0.7235     | 0.6890   |
| 128            | 32          | Hybrid            | 0.8311 	| 0.8027     | 0.7305     | 0.6864   |
| 128            | 32          | TFIDF             | 0.8248     | 0.7855     | 0.7299     | 0.6890   |
| 128            | 32          | Sequential        | 0.8051     | 0.7846     | 0.7134     | 0.6780   |

</div>

<p align="center"><b>Table 5: Tabulated summary of retrieval results using Fixed-Length Chunking with a chunk size of 128 and overlap of 32. The indexing method is still HNSW, and all reranking types are compared.</b></p>

<p align="center">
  <img src="external_plots/Task%203%20-%20Figure%201.png" width="750"/>
</p>

<p align="center"><b>Figure 5: Retrieval Performance by Reranker  (HNSW and Fixed-Length).</b></p>

## **Task 4: Evaluating Mistral Models in Isolation**

evaluation results for the Mistral models across small, medium, and large sizes reveal consistently low performance, particularly in terms of Exact Match (EM) scores, which approach zero across all difficulty levels. Even in the best case (the small and medium models on medium difficulty questions), EM reached only 0.0033, while all other EM scores, including for "easy" and "hard" questions, were flatly zero. This suggests that the models very rarely, if ever, produced answers that exactly matched the reference answers provided in the dataset.

The F1 scores, which provide a more forgiving partial credit metric based on overlap, show somewhat better, though still weak performance. Here, the scores ranged from 0.0089 to 0.1129, with medium difficulty questions surprisingly scoring higher than easy or hard questions across all model sizes. Notably, the small and large models performed slightly better than the medium model in terms of overall F1, though these differences are marginal.

<div align="center">

| **Model Size** | **Overall EM** | **Easy EM** | **Medium EM** | **Hard EM** |
| -------------- | -------------- | ----------- | ------------- | ----------- |
| Small          | 0.0010         | 0.0000      | 0.0033        | 0.0000      |
| Medium         | 0.0010         | 0.0000      | 0.0033        | 0.0000      |
| Large          | 0.0000         | 0.0000      | 0.0000        | 0.0000      |

</div>

<p align="center"><b>Table 6: Tabulated summary of the Exact Match (EM) evaluation results for the Mistral models across small, medium, and large sizes.</b></p>

<div align="center">

| **Model Size** | **Overall F1** | **Easy F1** | **Medium F1** | **Hard F1** |
| -------------- | -------------- | ----------- | ------------- | ----------- |
| Small          | 0.0538         | 0.0144      | 0.1129        | 0.0861  |
| Medium         | 0.0393         | 0.0089      | 0.0822        | 0.0644  |
| Large          | 0.0536         | 0.0155      | 0.1119        | 0.0823  |

</div>

<p align="center"><b>Table 7: Tabulated summary of the F1 evaluation results for the Mistral models across small, medium, and large sizes.</b></p>

<p align="center">
  <img src="external_plots/Task%204%20-%20Figure%201.png" width="750"/>
</p>

<p align="center"><b>Figure 6: Exact Match (EM) Scores by Model Size and Question Difficulty.</b></p>

<p align="center">
  <img src="external_plots/Task%204%20-%20Figure%202.png" width="750"/>
</p>

<p align="center"><b>Figure 7: F1 Scores by Model Size and Question Difficulty.</b></p>

These results point to several critical points, which may help explain the poor performance we observed:

1. All three Mistral models, when evaluated in a stand-alone context (without retrieval-augmented generation), demonstrate minimal capacity to generate exact or even closely aligned answers to open-ended questions. This highlights the inherent challenge of answering questions from memory alone, especially on broad, knowledge-dependent datasets.

2. The near-zero EM scores strongly indicate that the models are not generating answers in a form that matches the reference exactly. This could be due to paraphrasing, verbosity, or incomplete answers, even if the general information is correct. Token-level F1 shows that some overlap exists (particularly for medium difficulty questions) but still not at a level that would be considered practically useful.

3. For general users expecting accurate and reliable answers, these findings suggest major shortcomings in relying on these type of foundation models alone for open-domain question answering. A user with no external retrieval or grounding mechanism is likely to encounter vague and imprecise outputs, which may not align with expectations.

4. It is also possible that some of the poor results stem from evaluation mismatches. For example, gold answers may be phrased differently than valid model outputs, resulting in unfairly low EM or F1 scores. This is an issue that arises when answers may be semantically correct but lexically different, and manual or semantic evaluations might reveal more nuanced performance.

5. What's interesting to observs it that there’s no strong correlation between difficulty level and model performance. "Easy" questions scored lowest on average F1, possibly indicating that such questions require more specific answers (such as dates or names) that are harder to guess or infer without external knowledge. This is coutner-intuitive to the assumption that models should perform better on easier questions.

In summary, these findings underline the limitations of current base LLMs in various QA tasks. They highlight the need for futher fine-tuning to increase performance to a usable standard. From a research and user perspective, these results justify further exploration into more robust evaluation models and processes.

## **Task 5: Evaluating RAG Pipelines Across Mistral Models**

Here we introduced several architectural enhancements to our QA pipeline, including fixed-length chunking, HNSW indexing, and a hybrid reranking mechanism. However, when compared to the baseline results from Task 4, these additions did not lead to any meaningful performance improvements. The EM scores remained stagnant across all model sizes, with no model achieving more than 0.0010 overall EM. Similarly, F1 scores either slightly declined or stayed effectively the same. For example, the small model’s F1 decreased marginally from 0.0538 in Task 4 to 0.0533 in Task 5, while the large model dropped from 0.0536 to 0.0525. These trends held consistently across question difficulty levels, with easy, medium, and hard question categories showing no measurable gain.

<div align="center">

| **Model Size**  | **Overall EM** | **Easy EM** | **Medium EM** | **Hard EM** |
| --------------- | -------------- | ----------- | ------------- | ----------- |
| Small  	      | 0.0000         | 0.0000      | 0.0000        | 0.0000      |
| Medium 	      | 0.0010         | 0.0000      | 0.0033        | 0.0000      |
| Large  	      | 0.0000         | 0.0000      | 0.0000        | 0.0000      |

</div>

<p align="center"><b>Table 8: Tabulated summary of the Exact Match (EM) evaluation results for the Mistral models across small, medium, and large sizes. This now includes the optimal strategies seen earlier.</b></p>

<div align="center">

| **Model Size**  | **Overall F1** | **Easy F1** | **Medium F1** | **Hard F1** |
| --------------- | -------------- | ----------- | ------------- | ----------- |
| Small  	      | 0.0533         | 0.0143      | 0.1091        | 0.0890      |
| Medium 	      | 0.0391         | 0.0094      | 0.0815        | 0.0645      |
| Large  	      | 0.0525         | 0.0156      | 0.1098        | 0.0812      |

</div>

<p align="center"><b>Table 9: Tabulated summary of the F1 evaluation results for the Mistral models across small, medium, and large sizes. This now includes the optimal strategies seen earlier.</b></p>

<p align="center">
  <img src="external_plots/Task%205%20-%20Figure%201.png" width="750"/>
</p>

<p align="center"><b>Figure 8: Exact Match (EM) Scores by Model Size and Question Difficulty.</b></p>

<p align="center">
  <img src="external_plots/Task%205%20-%20Figure%202.png" width="750"/>
</p>

<p align="center"><b>Figure 9: F1 Scores by Model Size and Question Difficulty.</b></p>

Similar to the previous task, there are several likely sources of error that help explain these outcomes:

1. The reranker, though intended to prioritize semantically relevant chunks, may have elevated passages that were topically similar but failed to contain precise answer spans. 

2. The use of fixed-size chunking may have fragmented important contextual cues or split answer spans across chunk boundaries, rendering them less useful for the downstream QA process.

3. While HNSW offers efficient retrieval, its approximate nature, combined with potential embedding mismatches between the retriever and reranker, could lead to the omission of highly relevant chunks. The added complexity of the pipeline (where retrieval, reranking, and question answering each rely on different models )may also have introduced subtle misalignments in representation space.

4. Finally, there was no explicit supervision ensuring that retrieved contexts contained the gold answer, which further limited the potential for improvements in EM or F1.

In conclusion, although Task 5 implemented sophisticated retrieval and reranking mechanisms, the end-to-end QA performance did not improve over the simpler baseline from Task 4. This suggests that future work should focus on improving chunk selection strategies (such as overlapping windows), refining the reranker with domain-specific supervision, and enforcing answer containment during retrieval to maximize effectiveness.