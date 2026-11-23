# TextWave RAG Retrieval and Question Answering

A modular retrieval augmented generation (RAG) system for question answering over Wikipedia articles. It compares chunking strategies, FAISS index types, and rerankers for retrieval, then measures how well Mistral models answer questions from the retrieved context. It is served through a Flask API.

## Results

### Retrieval

Fixed length chunking (128 characters with 32 characters of overlap) and an HNSW index gave the strongest baseline. A hybrid reranker raised it further.

| Reranker | MRR | P@1 | P@3 | P@5 |
|---|---|---|---|---|
| None (baseline) | 0.822 | 0.783 | 0.718 | 0.689 |
| TF-IDF | 0.825 | 0.786 | 0.730 | 0.689 |
| Cross-encoder | 0.823 | 0.785 | 0.724 | 0.689 |
| **Hybrid** | **0.831** | **0.803** | **0.731** | 0.686 |

### Answer generation

The original course results showed low answer quality across Mistral small, medium, and large models, with exact match near zero and overall F1 between 0.04 and 0.05, and adding the best retrieval configuration did not change it. A later review found two causes in the code rather than in retrieval.

1. **The retrieved context never reached the model.** The line that added the passages to the prompt was commented out, so every run, including the retrieval augmented one, was effectively closed book. This is why the two settings scored almost the same.
2. **The answer format did not match the references.** The prompt asked for full sentence answers while the dataset's reference answers are short, such as "yes" or "1921", and the scoring did not strip punctuation, so "Yes." did not match "yes".

The generator now passes the retrieved passages and asks for short answers, and scoring follows the standard SQuAD normalization (lowercase, no punctuation or articles). These fixes have not yet been re-run against the Mistral API, so updated generation numbers are still to come. The [case study](textwave/CaseStudy.MD) records the original results.

## How it works

1. **Preprocessing.** Articles are split into sentence based or fixed length chunks with configurable overlap.
2. **Embedding and indexing.** Chunks are embedded with `all-MiniLM-L6-v2` and stored in a brute force, HNSW, or LSH FAISS index.
3. **Reranking.** Candidates can be reordered with TF-IDF, bag of words, a cross-encoder, a hybrid, or a sequential reranker.
4. **Generation.** `QAGeneratorMistral` sends the retrieved passages to a Mistral model and asks for a short answer drawn only from them. With no passages it answers closed book.
5. **Evaluation.** Scripts in `textwave/scripts/` compute MRR and precision at K for retrieval, and SQuAD style exact match and F1 by question difficulty for answers.

## Running it

The corpus and questions come from the [CMU Question-Answer Dataset](https://www.cs.cmu.edu/~ark/QA-data/) (Smith, Heilman, Hwa, and others) and are not redistributed here. Download and extract it, then build TextWave's inputs.

```bash
pip install -r requirements.txt
python textwave/scripts/prepare_data.py path/to/Question_Answer_Dataset_v1.2
```

Answer generation needs a Mistral API key. Put `MISTRAL_API_KEY=...` in a `.env` file at the repository root, which Git ignores. Then start the API, locally or in Docker.

```bash
python textwave/app.py
# or
docker build -f textwave/Dockerfile -t textwave-app .
docker run -p 5000:5000 --env-file .env textwave-app
```

[textwave/README.md](textwave/README.md) has the request format for the `/generate` endpoint, and [notebooks/demo_generator.ipynb](notebooks/demo_generator.ipynb) walks through the generator.

## Built with

Python, FAISS, Sentence Transformers, Hugging Face Transformers, Mistral API, scikit-learn, Flask, Docker

## Context

Built for EN.705.603 Creating AI-Enabled Systems at Johns Hopkins University (Summer 2025). Course starter code and unit tests are not included.
