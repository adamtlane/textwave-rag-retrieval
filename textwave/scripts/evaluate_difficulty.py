import sys
import os
import csv
import json
import time
import argparse
from collections import defaultdict
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Add 'textwave' project root to sys.path.
TEXTWAVE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if TEXTWAVE_DIR not in sys.path:
    sys.path.insert(0, TEXTWAVE_DIR)

from modules.generator.question_answering import QAGeneratorMistral

# Star the User Configuration.

MODEL_NAMES = ["mistral-large-latest"]

DEFAULT_NUM_QUESTIONS_LIMIT = 1051 # Maximum number of questions to evaluate (None = all).
DEFAULT_BATCH_SIZE = 50 # Number of questions to evaluate per batch.
MAX_RETRIES = 5 # Max API retry attempts on failure.
RETRY_BACKOFF = 5 # Retry exponential backoff multiplier in seconds.
SLEEP_BETWEEN_BATCHES = 2 # Optional cooldown between batches to avoid overload.

TSV_PATH = os.path.join(TEXTWAVE_DIR, "qa_resources", "question.tsv") # Path to QA dataset.
OUTPUT_DIR = os.path.join(TEXTWAVE_DIR, "qa_eval_results") # Output directory for CSV/JSON results.

API_KEY = os.environ.get("MISTRAL_API_KEY") # Load Mistral API key from environment.
if not API_KEY:
    raise EnvironmentError("MISTRAL_API_KEY not found in environment variables.")

# Start the Utility Functions.

def exact_match(pred, gold):
    """
    Check if the predicted answer exactly matches the gold answer (case and space insensitive).
    """

    return pred.strip().lower() == gold.strip().lower()

def compute_token_f1(pred, gold):
    """
    Compute token-level F1 score between predicted and gold answers.
    Tokenization is whitespace-based and case-insensitive.
    """
    pred_tokens = pred.lower().strip().split()
    gold_tokens = gold.lower().strip().split()

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
    Save evaluation results to a CSV file for a given model and difficulty level.

    Args:
        results (list): List of dictionaries with evaluation data.
        model_name (str): Name of the evaluated model.
        difficulty (str): Difficulty level (easy, medium, hard).
        output_dir (str): Directory to save the output file.
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
    Save a summary of evaluation metrics (EM, F1) by difficulty level as JSON.

    Args:
        summary (dict): Aggregated summary metrics.
        model_name (str): Name of the evaluated model.
        output_dir (str): Output directory for JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name.replace('/', '-')}_summary.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, mode="w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"[Saved JSON Summary] {filepath}")

def summarize(results):
    """
    Compute average exact match and F1 scores across all examples, grouped by difficulty level.

    Args:
        results (list): List of evaluation records with 'is_correct', 'f1', and 'difficulty'.

    Returns:
        dict: Summary statistics with average EM and F1 for overall, easy, medium, and hard questions.
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

# Star the Main Evaluation Logic.

def evaluate_model(model_name, num_questions_limit, batch_size, sanity_mode):
    """
    Evaluate a QA model on a dataset, computing accuracy and F1 per difficulty level.

    Args:
        model_name (str): Name of the generator model.
        num_questions_limit (int or None): Max number of questions to evaluate.
        batch_size (int): Batch size for processing.
        sanity_mode (bool): If True, evaluate only 5 questions for debugging.

    Returns:
        list: List of per-example evaluation records.
    """
    generator = QAGeneratorMistral(api_key=API_KEY, generator_model=model_name)
    all_results = []

    with open(TSV_PATH, encoding='utf-8') as f: # Load TSV of QA data.
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    
    if sanity_mode: # Apply constraints.
        rows = rows[:5]
        batch_size = 1
    elif num_questions_limit:
        rows = rows[:num_questions_limit]

    total_batches = (len(rows) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(rows))
        batch_rows = rows[start:end]

        print(f"\n[Batch {batch_idx+1}/{total_batches}] Processing {len(batch_rows)} questions...")
        for row in tqdm(batch_rows, desc=f"{model_name} | Q{start+1}-{end}"):
            question = row["Question"].strip()
            gold_answer = row["Answer"].strip()
            raw_difficulty = row.get("DifficultyFromAnswerer")

            if raw_difficulty is None: # Skip if difficulty not labeled.
                print(f"[Warning] Skipping due to missing difficulty: {question[:60]}...")
                continue

            difficulty = raw_difficulty.strip().lower()
            pred = ""

            for attempt in range(MAX_RETRIES): # Retry on failure (for example, 429 errors).
                try:
                    pred = generator.generate_answer(query=question, context=[])
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

        if SLEEP_BETWEEN_BATCHES and not sanity_mode: # Optional cooldown.
            print(f"[Pause] Sleeping {SLEEP_BETWEEN_BATCHES}s between batches to avoid overload...")
            time.sleep(SLEEP_BETWEEN_BATCHES)

    return all_results

if __name__ == "__main__": # Entry Point.
    parser = argparse.ArgumentParser(description="Evaluate QA Generation Performance")
    parser.add_argument("--sanity", action="store_true", help="Run a quick sanity check with 5 examples")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--limit", type=int, default=DEFAULT_NUM_QUESTIONS_LIMIT, help="Question limit")
    args = parser.parse_args()

    print("Task 4 Results - Standalone QA (No RAG)")
    print("---------------------------------------")
    print(f"Loaded questions from: {TSV_PATH}\n")

    for model_name in MODEL_NAMES:
        print(f"\n--- Evaluating: {model_name} ---")
        results = evaluate_model(model_name, args.limit, args.batch, args.sanity)
        summary = summarize(results)
        
        by_difficulty = defaultdict(list) # Group results by difficulty level for per-difficulty CSV export.
        for r in results:
            by_difficulty[r["difficulty"]].append(r)

        for difficulty, records in by_difficulty.items():
            save_results_to_csv(records, model_name, difficulty)

        print("\nExact Match Results:") # Print the summary.
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