"""
Prepare the TextWave corpus and question set from the CMU Question-Answer Dataset.

Download the dataset from https://www.cs.cmu.edu/~ark/QA-data/ and extract it.
The release has three portions, each laid out the same way:

    Question_Answer_Dataset_v1.2/
    ├── S08/  data/set1/a1.txt.clean ...  question_answer_pairs.txt
    ├── S09/  data/set1/a1.txt.clean ...  question_answer_pairs.txt
    └── S10/  data/set1/a1.txt.clean ...  question_answer_pairs.txt

This script writes the two inputs TextWave expects:

    textwave/storage/S08_set1_a1.txt.clean ...   (all 150 articles from S08, S09, S10)
    textwave/qa_resources/question.tsv           (the S08 questions used in the evaluation)

Usage (from the repository root):

    python textwave/scripts/prepare_data.py path/to/Question_Answer_Dataset_v1.2
"""

import os
import shutil
import sys

TEXTWAVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(TEXTWAVE_DIR, "storage")
QA_PATH = os.path.join(TEXTWAVE_DIR, "qa_resources", "question.tsv")
PORTIONS = ("S08", "S09", "S10")
QUESTION_PORTION = "S08"


def copy_articles(root: str) -> int:
    """Copy <portion>/data/setN/aM.txt.clean to storage/<portion>_setN_aM.txt.clean."""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    copied = 0
    for portion in PORTIONS:
        data_dir = os.path.join(root, portion, "data")
        if not os.path.isdir(data_dir):
            sys.exit(f"Missing {data_dir}. Point this script at the extracted "
                     f"Question_Answer_Dataset_v1.2 folder.")
        for set_name in sorted(os.listdir(data_dir)):
            set_dir = os.path.join(data_dir, set_name)
            if not os.path.isdir(set_dir):
                continue
            for name in sorted(os.listdir(set_dir)):
                if name.endswith(".txt.clean"):
                    shutil.copy(os.path.join(set_dir, name),
                                os.path.join(STORAGE_DIR, f"{portion}_{set_name}_{name}"))
                    copied += 1
    return copied


def write_questions(root: str) -> int:
    """Rewrite ArticleFile "data/set3/a4" as "S08_set3_a4" to match the corpus names.

    The file is processed line by line rather than with a CSV parser, because some
    questions contain unbalanced quote characters that a CSV reader would merge.
    """
    pairs_path = os.path.join(root, QUESTION_PORTION, "question_answer_pairs.txt")
    raw = open(pairs_path, "rb").read()
    try:                      # the release is usually Latin-1; accept UTF-8 too
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    os.makedirs(os.path.dirname(QA_PATH), exist_ok=True)
    rows = 0
    with open(QA_PATH, "w", encoding="utf-8", newline="") as dst:
        for i, line in enumerate(text.splitlines(keepends=True)):
            ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            fields = line[:len(line) - len(ending)].split("\t")
            if i > 0 and fields and fields[-1].strip().startswith("data/"):
                article = fields[-1].strip()[len("data/"):].replace("/", "_")
                fields[-1] = f"{QUESTION_PORTION}_{article}"
                rows += 1
            dst.write("\t".join(fields) + ending)
    return rows


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    n_articles = copy_articles(sys.argv[1])
    n_questions = write_questions(sys.argv[1])
    print(f"Copied {n_articles} articles to {STORAGE_DIR}")
    print(f"Wrote {n_questions} question rows to {QA_PATH}")
