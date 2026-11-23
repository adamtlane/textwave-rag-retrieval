import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Add 'textwave' project root to sys.path.
TEXTWAVE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if TEXTWAVE_DIR not in sys.path:
    sys.path.insert(0, TEXTWAVE_DIR)

import csv
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from modules.utils.metrics import Matching
from modules.extraction.preprocessing import DocumentProcessing
from modules.extraction.embedding import Embedding
from modules.retrieval.index.bruteforce import BruteForceIndex
from modules.retrieval.index.hnsw import HNSWIndex
from modules.retrieval.index.lsh import LSHIndex
from modules.retrieval.reranker import Reranker

# Start the user configuration parameters.

ROOT_DIR = TEXTWAVE_DIR
QUESTIONS_TSV = os.path.join(ROOT_DIR, "qa_resources", "question.tsv")
CORPUS_DIR = os.path.join(ROOT_DIR, "storage")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CHUNKING_STRATEGY = "fixed-length" # Choose between "sentence" or "fixed-length".

SENTENCE_NUM_SENTENCES = 10 # Parameters for sentence-based chunking.
SENTENCE_OVERLAP = 3

FIXED_CHUNK_SIZE = 128 # Parameters for fixed-length chunking.
FIXED_OVERLAP = 32

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_TYPE = "hnsw" # Choose between "brute", "lsh", "hnsw".

TOP_K = 5
RERANK_TYPE = "sequential" # The options are "None", "cross_encoder", "bow", "hybrid", "tfidf", "sequential".

if CHUNKING_STRATEGY == "sentence": # Unique tag for filename based on chunking strategy and parameters.
    PARAM_STR = f"{SENTENCE_NUM_SENTENCES}s_{SENTENCE_OVERLAP}o"
elif CHUNKING_STRATEGY == "fixed-length":
    PARAM_STR = f"{FIXED_CHUNK_SIZE}c_{FIXED_OVERLAP}o"
else:
    raise ValueError(f"Unknown chunking strategy: {CHUNKING_STRATEGY}")

# Start all necessary helper functions.

def load_questions(tsv_path: str) -> List[Dict]:
    """
    Load questions from a TSV file.

    Parameters:
        tsv_path (str): Path to the questions file.

    Returns:
        List[Dict]: A list of question dictionaries.
    """
    questions = []

    with open(tsv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            questions.append(row)

    return questions

def load_corpus_files(corpus_dir: str) -> List[str]:
    """
    Load filenames of all cleaned text files in the corpus directory.

    Parameters:
        corpus_dir (str): Directory containing `.txt.clean` files.

    Returns:
        List[str]: Sorted list of filenames.
    """
    files = [f for f in os.listdir(corpus_dir) if f.endswith(".txt.clean")]
    files.sort()

    return files

def preprocess_corpus(dp: DocumentProcessing, files: List[str], strategy: str) -> Dict[str, List[str]]:
    """
    Apply chunking to all corpus files using the selected strategy.

    Parameters:
        dp (DocumentProcessing): Chunking utility instance.
        files (List[str]): List of corpus filenames.
        strategy (str): Chunking strategy ("sentence" or "fixed-length").

    Returns:
        Dict[str, List[str]]: Mapping from filename to list of chunks.
    """
    chunks_per_file = {}

    for filename in tqdm(files, desc=f"Chunking corpus ({strategy})"):
        file_path = os.path.join(CORPUS_DIR, filename)
        if strategy == "sentence":
            chunks = dp.sentence_chunking(file_path, num_sentences=SENTENCE_NUM_SENTENCES, overlap_size=SENTENCE_OVERLAP)
        elif strategy == "fixed-length":
            chunks = dp.fixed_length_chunking(file_path, chunk_size=FIXED_CHUNK_SIZE, overlap_size=FIXED_OVERLAP)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        chunks_per_file[filename] = chunks

    return chunks_per_file

def flatten_chunks(chunks_per_file: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """
    Flatten dictionary of chunk lists into two aligned lists: chunks and source filenames.

    Parameters:
        chunks_per_file (Dict[str, List[str]]): Chunked content by filename.

    Returns:
        Tuple[List[str], List[str]]: Flat list of chunks and corresponding source filenames.
    """
    all_chunks = []
    all_sources = []
    for filename, chunks in tqdm(chunks_per_file.items(), desc="Flattening chunks"):
        all_chunks.extend(chunks)
        all_sources.extend([filename] * len(chunks))

    return all_chunks, all_sources

def rerank_results(questions: List[Dict], raw_results: List[List[Dict]]) -> List[List[Dict]]:
    """
    Apply reranking to initial retrieved results using the configured reranking method.

    Parameters:
        questions (List[Dict]): List of question dictionaries.
        raw_results (List[List[Dict]]): Raw retrieval results for each question.

    Returns:
        List[List[Dict]]: Reranked results per question.
    """
    if len(questions) != len(raw_results):
        raise ValueError("Mismatch between number of questions and number of raw results.")

    reranked_results = []
    reranker = Reranker(type=RERANK_TYPE)

    for i, q in enumerate(questions):
        query = q["Question"]
        context = raw_results[i]

        if not context:
            reranked_results.append([])
            continue

        texts = [doc["text"] if isinstance(doc, dict) else str(doc) for doc in context]
        ranked_docs, ranked_indices, ranked_scores = reranker.rerank(query, texts)

        reranked_results.append([
            {
                "text": ranked_docs[j],
                "doc_id": context[ranked_indices[j]]["doc_id"],
                "score": ranked_scores[j]
            }
            for j in range(len(ranked_docs))
        ])

    return reranked_results

def evaluate_retrieval(
    questions: List[Dict],
    index,
    embedding: Embedding,
    top_k: int
) -> Dict[str, float]:
    """
    Evaluate retrieval effectiveness using metrics like MRR and Precision@K.

    Parameters:
        questions (List[Dict]): List of question dictionaries.
        index: The retrieval index used for searching.
        embedding (Embedding): Embedding model instance.
        top_k (int): Number of top documents to retrieve.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    reciprocal_ranks = []
    precision_at_1 = []
    precision_at_3 = []
    precision_at_5 = []
    all_raw_results = []

    skipped = 0
    for q in tqdm(questions, desc="Evaluating retrieval"):
        question_text = q.get("Question", "")
        article_file_raw = q.get("ArticleFile")
        if not article_file_raw or not question_text:
            skipped += 1
            all_raw_results.append([])
            continue

        results = index.search(question_text, top_k=top_k)
        all_raw_results.append(results if results else [])

    print(f"Skipped {skipped} questions with missing data.")

    if RERANK_TYPE and RERANK_TYPE.lower() != "none":
        print(f"Applying reranking method: {RERANK_TYPE}")
        all_raw_results = rerank_results(questions, all_raw_results)

    for q, results in zip(questions, all_raw_results):
        article_file_raw = q.get("ArticleFile")
        if not article_file_raw:
            continue

        article_file = os.path.basename(article_file_raw) + ".txt.clean"
        retrieved_doc_ids = [res['doc_id'] for res in results]
        relevant_positions = [pos for pos, doc_id in enumerate(retrieved_doc_ids) if doc_id == article_file]

        reciprocal_ranks.append(1.0 / (relevant_positions[0] + 1) if relevant_positions else 0.0)
        precision_at_1.append(1.0 if relevant_positions and relevant_positions[0] == 0 else 0.0)
        precision_at_3.append(
            sum(1 for pos in range(min(3, len(retrieved_doc_ids))) if retrieved_doc_ids[pos] == article_file) / 3.0
        )
        precision_at_5.append(
            sum(1 for pos in range(min(5, len(retrieved_doc_ids))) if retrieved_doc_ids[pos] == article_file) / 5.0
        )

    return {
        "MRR": np.mean(reciprocal_ranks),
        "P@1": np.mean(precision_at_1),
        "P@3": np.mean(precision_at_3),
        "P@5": np.mean(precision_at_5),
        "num_questions": len(questions)
    }

def save_results(metrics: Dict[str, float], strategy: str, param_str: str):
    """
    Save evaluation metrics to CSV.

    Parameters:
        metrics (Dict[str, float]): Evaluation results.
        strategy (str): Chunking strategy name.
        param_str (str): Unique string encoding experiment parameters.
    """
    filename = f"evaluation_{strategy}_{param_str}_{INDEX_TYPE}_results.csv"
    out_path = os.path.join(RESULTS_DIR, filename)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            writer.writerow([k, v])

    print(f"Results saved to {out_path}")

def embedding_with_progress(embedder, texts: List[str], batch_size: int = 64) -> List[np.ndarray]:
    """
    Encode text in batches with a progress bar.

    Parameters:
        embedder: Embedding function.
        texts (List[str]): Text chunks to embed.
        batch_size (int): Batch size.

    Returns:
        List[np.ndarray]: List of embeddings.
    """
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding corpus chunks"):
        batch = texts[i : i + batch_size]
        emb_batch = embedder(batch)
        embeddings.extend(emb_batch)

    return embeddings

def main():
    """Main execution function: preprocess, index, retrieve, and evaluate."""
    print("=== Evaluating Retrieval Performance for Chunking Strategy ===")
    print(f"Strategy: {CHUNKING_STRATEGY}")
    print(f"Embedding: {EMBEDDING_MODEL_NAME}")
    print(f"Index type: {INDEX_TYPE}")
    print(f"Reranking method: {RERANK_TYPE or 'None'}")
    print("Output folder:", RESULTS_DIR)

    if CHUNKING_STRATEGY == "sentence":
        print(f"Params: num_sentences={SENTENCE_NUM_SENTENCES}, overlap={SENTENCE_OVERLAP}")
    else:
        print(f"Params: chunk_size={FIXED_CHUNK_SIZE}, overlap={FIXED_OVERLAP}")

    dp = DocumentProcessing()
    embedding = Embedding(EMBEDDING_MODEL_NAME)

    print("Loading questions...")
    questions = load_questions(QUESTIONS_TSV)
    print(f"Loaded {len(questions)} questions.")

    print("Loading corpus files...")
    corpus_files = load_corpus_files(CORPUS_DIR)
    print(f"Found {len(corpus_files)} corpus files.")

    chunks_per_file = preprocess_corpus(dp, corpus_files, CHUNKING_STRATEGY) # Chunk and embed documents.
    all_chunks, chunk_sources = flatten_chunks(chunks_per_file)

    print("Building embedding index...")
    dim = embedding.encode(all_chunks[0]).shape[0] # Infer embedding dimension.

    if INDEX_TYPE == "brute": # Select index type.
        index = BruteForceIndex(dim=dim, embedder=None, metric="cosine")
    elif INDEX_TYPE == "lsh":
        index = LSHIndex(dim=dim, nbits=10, embedder=None)
    elif INDEX_TYPE == "hnsw":
        index = HNSWIndex(dim=dim, efConstruction=200, M=16, embedder=None)
    else:
        raise ValueError(f"Unknown index type: {INDEX_TYPE}")

    index.set_embedder(embedding.encode) # Attach embedding model.

    if hasattr(index, "build"): # Use bulk build or per-document addition.
        print("Using batch `build()` method...")
        def embedder_with_progress(texts):
            return embedding_with_progress(embedding.encode, texts)
        index.set_embedder(embedder_with_progress)
        index.build(all_chunks, chunk_sources)
    else:
        print("Using iterative `add()` method...")
        for chunk, source in tqdm(zip(all_chunks, chunk_sources), desc="Indexing", total=len(all_chunks)):
            index.add(chunk, doc_id=source)

    print("Evaluating retrieval...")
    metrics = evaluate_retrieval(questions, index, embedding, TOP_K)

    print("\n=== Retrieval Performance ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    save_results(metrics, CHUNKING_STRATEGY, PARAM_STR)

if __name__ == "__main__":
    main()