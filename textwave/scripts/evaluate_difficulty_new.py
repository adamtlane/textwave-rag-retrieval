import string
import re
import sys
import os
import csv
import json
import time
import argparse
from collections import defaultdict
from tqdm import tqdm
from typing import List
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Add 'textwave' project root to sys.path.
TEXTWAVE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if TEXTWAVE_DIR not in sys.path:
    sys.path.insert(0, TEXTWAVE_DIR)

from modules.generator.question_answering import QAGeneratorMistral
from modules.retrieval.reranker import Reranker
from modules.retrieval.index.hnsw import HNSWIndex
from modules.extraction.embedding import Embedding 

# Star the User Configuration Parameters.

MODEL_NAMES = ["mistral-large-latest"]

DEFAULT_NUM_QUESTIONS_LIMIT = 1051 # Set None to use all.
DEFAULT_BATCH_SIZE = 50
MAX_RETRIES = 5
RETRY_BACKOFF = 5 # In seconds, exponential backoff.
SLEEP_BETWEEN_BATCHES = 2 # The cooldown between batches.

TSV_PATH = os.path.join(TEXTWAVE_DIR, "qa_resources", "question.tsv")
OUTPUT_DIR = os.path.join(TEXTWAVE_DIR, "qa_eval_results")

CORPUS_DIR = os.path.join(TEXTWAVE_DIR, "storage")
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBED_DIM = 384 # The embedding dimension for all-MiniLM-L6-v2.

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 40
HNSW_TOP_K = 15 # Retrieve the top 15 from HNSW before reranking.
CHUNK_SIZE = 250 # Number of characters per chunk.
CHUNK_OVERLAP = 50 # NUmber of overlap chars between chunks.

API_KEY = os.environ.get("MISTRAL_API_KEY")
if not API_KEY:
    raise EnvironmentError("MISTRAL_API_KEY not found in environment variables.")

# Start the Utility Functions.

def normalize_answer(text):
    """
    Lowercases, removes punctuation and the articles a, an, and the, and collapses
    whitespace, following the standard SQuAD evaluation, so that "Yes." matches "yes".
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def exact_match(pred, gold):
    """
    Determines whether the predicted answer exactly matches the gold answer after normalization.
    """
    return normalize_answer(pred) == normalize_answer(gold)

def compute_token_f1(pred, gold):
    """
    Computes the F1 score between normalized predicted and gold answers at the token level.

    Returns:
        float: F1 score between 0.0 and 1.0
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    common = set(pred_tokens) & set(gold_tokens)
    overlap = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1

def save_results_to_csv(results, model_name, difficulty, output_dir=OUTPUT_DIR):
    """
    Saves evaluation results to a CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name.replace('/', '-')}_{difficulty}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Expected Answer", "Generated Answer", "Exact Match", "F1 Score"])
        for r in results:
            writer.writerow([r["question"], r["gold"], r["pred"], str(r["is_correct"]), f"{r['f1']:.4f}"])

    print(f"[Saved CSV] {filepath}")

def save_summary_to_json(summary, model_name, output_dir=OUTPUT_DIR):
    """
    Saves overall evaluation summary to a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name.replace('/', '-')}_summary.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, mode="w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"[Saved JSON Summary] {filepath}")

def summarize(results):
    """
    Aggregates evaluation metrics (Exact Match and F1) by difficulty level and overall.

    Returns:
        dict: summary statistics with keys [overall, easy, medium, hard]
    """
    stats = defaultdict(list)

    for r in results:
        stats[r["difficulty"]].append((r["is_correct"], r["f1"]))
        stats["overall"].append((r["is_correct"], r["f1"]))

    summary = {}

    for key in ["overall", "easy", "medium", "hard"]:
        values = stats.get(key, [])
        if not values:
            summary[key] = {"exact_match": 0.0, "f1": 0.0}
            continue
        exacts, f1s = zip(*values)
        summary[key] = {
            "exact_match": sum(exacts) / len(exacts),
            "f1": sum(f1s) / len(f1s)
        }

    return summary

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    """
    Splits a long text into overlapping chunks of approximately chunk_size characters.

    Args:
        text (str): Input document.
        chunk_size (int): Desired size of each chunk in characters.
        overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_length:
            break
        start = end - overlap

    return chunks

def load_corpus() -> (List[str], List[str]):
    """
    Loads and reads all cleaned text documents from the corpus directory.

    Returns:
        List[str]: List of document texts.
        List[str]: Corresponding list of filenames.
    """
    files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt.clean")]
    files.sort()
    corpus_docs = []

    print(f"[Corpus] Loading {len(files)} corpus files...")
    for filename in tqdm(files, desc="Loading corpus files"):
        path = os.path.join(CORPUS_DIR, filename)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            corpus_docs.append(f.read())

    return corpus_docs, files

embedding_model = Embedding(EMBEDDING_MODEL_NAME) # Embedding wrapper.

def embedder_fn(texts: List[str]) -> np.ndarray:
    """
    Encodes a list of text strings into embeddings using the configured embedding model.

    Args:
        texts (List[str]): Text inputs to embed.

    Returns:
        np.ndarray: Embedding vectors of shape (n_samples, EMBED_DIM)
    """
    print(f"[Embedding] Encoding {len(texts)} texts...")
    embeddings = embedding_model.model.encode(texts, batch_size=64, show_progress_bar=True)

    return np.array(embeddings).astype('float32')

# Start the Main Evaluation Logic with Reranking & HNSW Search.

def build_hnsw_index():
    """
    Constructs an HNSW index from corpus documents.

    Returns:
        HNSWIndex: Trained index.
        List[str]: Chunked texts.
        List[str]: Chunk identifiers.
    """
    print("[Corpus] Loading corpus documents...")
    corpus_docs, doc_ids = load_corpus()

    print("[Chunking] Splitting corpus documents into chunks...")
    chunk_texts = []
    chunk_ids = []

    for doc_idx, text in enumerate(tqdm(corpus_docs, desc="Chunking corpus")):
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_texts.append(chunk)
            chunk_ids.append(f"{doc_ids[doc_idx]}_chunk{chunk_idx}")

    embeddings = embedder_fn(chunk_texts)

    print("[Index] Building HNSW index...")
    hnsw_index = HNSWIndex(
        dim=EMBED_DIM,
        embedder=embedder_fn,
        metric='cosine',
        M=HNSW_M,
        efConstruction=HNSW_EF_CONSTRUCTION
    )
    hnsw_index.build(chunk_texts, chunk_ids)
    print("[Index] HNSW index built.")

    return hnsw_index, chunk_texts, chunk_ids

def evaluate_model(model_name, num_questions_limit, batch_size):
    """
    Runs evaluation pipeline for a given model using HNSW-based retrieval and reranking.

    Args:
        model_name (str): Name of the QA model to use.
        num_questions_limit (int): Max number of QA examples to evaluate.
        batch_size (int): Number of examples per batch.

    Returns:
        List[dict]: Evaluation results with EM and F1 scores.
    """
    generator = QAGeneratorMistral(api_key=API_KEY, generator_model=model_name)
    reranker = Reranker(type="cross_encoder")
    all_results = []

    hnsw_index, chunk_texts, chunk_ids = build_hnsw_index()

    with open(TSV_PATH, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if num_questions_limit:
        rows = rows[:num_questions_limit]

    total_batches = (len(rows) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(rows))
        batch_rows = rows[start:end]

        print(f"\n[Batch {batch_idx+1}/{total_batches}] Processing {len(batch_rows)} questions...")

        for row in batch_rows:
            question = row["Question"].strip()
            gold_answer = row["Answer"].strip()
            raw_difficulty = row.get("DifficultyFromAnswerer")

            if raw_difficulty is None:
                print(f"[Warning] Skipping due to missing difficulty: {question[:60]}...")
                continue

            difficulty = raw_difficulty.strip().lower()
            pred = ""

            try:
                candidates = hnsw_index.search(question, top_k=HNSW_TOP_K)
            except Exception as e:
                print(f"[Error] HNSW search failed for question: {question[:60]}... -> {e}")
                candidates = []

            candidate_texts = [c['text'] for c in candidates]

            if candidate_texts:
                reranked_docs, _, _ = reranker.rerank(question, candidate_texts)
                context_for_qa = reranked_docs[:5]
            else:
                context_for_qa = []

            for attempt in range(MAX_RETRIES):
                try:
                    pred = generator.generate_answer(query=question, context=context_for_qa)
                    break
                except Exception as e:
                    if "capacity" in str(e).lower() or "429" in str(e):
                        wait = min(RETRY_BACKOFF * (2 ** attempt), 60)
                        print(f"[Retry {attempt+1}] 429 error or capacity issue. Sleeping {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"[Error] {model_name} failed on question: {question[:60]}... -> {e}")
                        break

            result = {
                "question": question,
                "gold": gold_answer,
                "pred": pred,
                "difficulty": difficulty,
                "is_correct": exact_match(pred, gold_answer),
                "f1": compute_token_f1(pred, gold_answer)
            }
            all_results.append(result)

        if SLEEP_BETWEEN_BATCHES:
            print(f"[Pause] Sleeping {SLEEP_BETWEEN_BATCHES}s between batches to avoid overload...")
            time.sleep(SLEEP_BETWEEN_BATCHES)

    return all_results

# Start the Entry Point.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QA Generation Performance with Reranking and HNSW")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--limit", type=int, default=DEFAULT_NUM_QUESTIONS_LIMIT, help="Question limit")
    args = parser.parse_args()

    print("Task 5 Results - QA with HNSW + Reranking Context")
    print("--------------------------------------------------")
    print(f"Loaded questions from: {TSV_PATH}\n")

    for model_name in MODEL_NAMES:
        print(f"\n--- Evaluating: {model_name} ---")
        results = evaluate_model(model_name, args.limit, args.batch)
        summary = summarize(results)

        by_difficulty = defaultdict(list)
        for r in results:
            by_difficulty[r["difficulty"]].append(r)

        for difficulty, records in by_difficulty.items():
            save_results_to_csv(records, model_name, difficulty)

        print("\nExact Match Results:")
        print(f"  Overall EM : {summary['overall']['exact_match']:.4f}")
        print(f"  Easy EM    : {summary['easy']['exact_match']:.4f}")
        print(f"  Medium EM  : {summary['medium']['exact_match']:.4f}")
        print(f"  Hard EM    : {summary['hard']['exact_match']:.4f}")

        print("\nF1 Score Results:")
        print(f"  Overall F1 : {summary['overall']['f1']:.4f}")
        print(f"  Easy F1    : {summary['easy']['f1']:.4f}")
        print(f"  Medium F1  : {summary['medium']['f1']:.4f}")
        print(f"  Hard F1    : {summary['hard']['f1']:.4f}")

        save_summary_to_json(summary, model_name)