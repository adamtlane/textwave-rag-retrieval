from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import torch
import numpy as np

class Reranker:
    """
    A flexible reranker class supporting multiple reranking strategies, including:
    cross-encoder, TF-IDF, bag-of-words, hybrid, and sequential.

    Depending on the strategy, this class scores and ranks a list of candidate texts
    given a user query.
    """

    def __init__(self, type, cross_encoder_model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2', corpus_directory=''):
        """
        Initializes the reranker with the specified strategy and optional cross-encoder model.

        Args:
            type (str): Reranking strategy ('cross_encoder', 'tfidf', 'bow', 'hybrid', 'sequential').
            cross_encoder_model_name (str): Name of the HuggingFace cross-encoder model. Default is TinyBERT.
            corpus_directory (str): Unused parameter placeholder for potential future use.
        """
        self.type = type
        self.cross_encoder_model_name = cross_encoder_model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cross_encoder_model = None
        self.tokenizer = None

        if self.type in ['cross_encoder', 'hybrid', 'sequential']:
            try:
                self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model_name).to(self.device)
            except AttributeError:
                self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model_name)

    def rerank(self, query, context, distance_metric="cosine", seq_k1=None, seq_k2=None):
        """
        Dispatches reranking to the appropriate strategy.

        Args:
            query (str): Input query string.
            context (List[str]): List of candidate document strings.
            distance_metric (str): Distance metric for vector-based methods. Default is "cosine".
            seq_k1 (int, optional): Top-k for TF-IDF stage in sequential reranking.
            seq_k2 (int, optional): Top-k for cross-encoder stage in sequential reranking.

        Returns:
            Tuple[List[str], List[int], List[float]]: Ranked documents, their original indices, and scores.
        """
        if self.type == "cross_encoder":
            return self.cross_encoder_rerank(query, context, top_k=len(context))
        elif self.type == "tfidf":
            return self.tfidf_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "bow":
            return self.bow_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "hybrid":
            return self.hybrid_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "sequential":
            return self.sequential_rerank(query, context, seq_k1, seq_k2, distance_metric=distance_metric)
        else:
            raise ValueError(f"Unsupported reranker type: {self.type}")

    def cross_encoder_rerank(self, query, context, top_k=None):
        """
        Reranks documents using a cross-encoder model.

        Args:
            query (str): Input query.
            context (List[str]): List of candidate documents.
            top_k (int, optional): Not used in current implementation.

        Returns:
            Tuple[List[str], List[int], List[float]]: Ranked documents, their indices, and scores.
        """
        if not self.cross_encoder_model or not self.tokenizer:
            raise RuntimeError("Cross-encoder model not initialized.")

        pairs = [[query, doc] for doc in context] # Create input pairs.
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad(): # Predict relevance scores.
            logits = self.cross_encoder_model(**inputs).logits
            if logits.dim() == 2 and logits.size(1) > 1:
                scores = logits[:, 0].tolist()
            else:
                scores = logits.view(-1).tolist()

        if len(scores) != len(context):
            raise RuntimeError(f"Tokenized {len(pairs)} pairs, but got {len(scores)} scores.")

        indices = np.argsort(scores)[::-1] # Sort by descending score.
        ranked_docs = [context[i] for i in indices]
        ranked_scores = [round(scores[i], 4) for i in indices]

        return ranked_docs, indices.tolist(), ranked_scores

    def tfidf_rerank(self, query, context, distance_metric="cosine"):
        """
        Reranks using TF-IDF vector similarity.

        Args:
            query (str): Query string.
            context (List[str]): Candidate texts.
            distance_metric (str): Metric for pairwise distances. Default is "cosine".

        Returns:
            Tuple[List[str], List[int], List[float]]: Ranked documents, their indices, and distances.
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([query] + context)
        distances = pairwise_distances(tfidf_matrix[0], tfidf_matrix[1:], metric=distance_metric).flatten()

        indices = np.argsort(distances, kind="stable")
        ranked_docs = [context[i] for i in indices]
        ranked_distances = [distances[i] for i in indices]

        return ranked_docs, indices.tolist(), ranked_distances

    def bow_rerank(self, query, context, distance_metric="cosine"):
        """
        Reranks using Bag-of-Words (BoW) vector similarity.

        Args:
            query (str): Query string.
            context (List[str]): Candidate texts.
            distance_metric (str): Distance metric. Default is "cosine".

        Returns:
            Tuple[List[str], List[int], List[float]]: Ranked documents, their indices, and distances.
        """
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform([query] + context)
        distances = pairwise_distances(bow_matrix[0], bow_matrix[1:], metric=distance_metric).flatten()

        indices = np.argsort(distances, kind="stable")
        ranked_docs = [context[i] for i in indices]
        ranked_distances = [distances[i] for i in indices]

        return ranked_docs, indices.tolist(), ranked_distances

    def hybrid_rerank(self, query, context, distance_metric="cosine", tfidf_weight=0.3):
        """
        Combines TF-IDF and cross-encoder scores using weighted average.

        Args:
            query (str): Input query.
            context (List[str]): Candidate documents.
            distance_metric (str): Distance metric for TF-IDF.
            tfidf_weight (float): Weight for TF-IDF score. Default is 0.3.

        Returns:
            Tuple[List[str], List[int], List[float]]: Ranked documents, indices, and scores.
        """
        tfidf_docs, _, tfidf_scores = self.tfidf_rerank(query, context, distance_metric=distance_metric)
        cross_docs, _, cross_scores = self.cross_encoder_rerank(query, context)

        tfidf_dict = {doc: score for doc, score in zip(tfidf_docs, tfidf_scores)}
        cross_dict = {doc: score for doc, score in zip(cross_docs, cross_scores)}

        combined_docs = set(tfidf_dict.keys()).intersection(cross_dict.keys())
        doc_to_score = {
            doc: tfidf_weight * tfidf_dict[doc] + (1 - tfidf_weight) * cross_dict[doc]
            for doc in combined_docs
        }

        ranked = sorted(doc_to_score.items(), key=lambda x: x[1], reverse=True)
        ranked_docs = [doc for doc, _ in ranked]
        ranked_scores = [round(score, 4) for _, score in ranked]
        ranked_indices = [context.index(doc) for doc in ranked_docs]

        return ranked_docs, ranked_indices, ranked_scores

    def sequential_rerank(self, query, context, seq_k1, seq_k2, distance_metric="cosine"):
        """
        Applies two-stage reranking: TF-IDF followed by cross-encoder on top-K.

        Args:
            query (str): Input query.
            context (List[str]): Candidate texts.
            seq_k1 (int): Top-k results to keep after TF-IDF.
            seq_k2 (int): Top-k results to keep after cross-encoder.
            distance_metric (str): Distance metric for TF-IDF. Default is "cosine".

        Returns:
            Tuple[List[str], List[int], List[float]]: Reranked documents, their original indices, and scores.
        """
        
        tfidf_docs, _, _ = self.tfidf_rerank(query, context, distance_metric=distance_metric) # TF-IDF top seq_k1.
        tfidf_top_docs = tfidf_docs[:seq_k1]
        
        cross_docs, cross_local_indices, cross_scores = self.cross_encoder_rerank(query, tfidf_top_docs, top_k=seq_k2) # Cross-encoder top seq_k2 from TF-IDF selection.
       
        doc_to_global_idx = {doc: i for i, doc in enumerate(context)}  # Convert local indices to global context indices.
        cross_global_indices = [doc_to_global_idx[doc] for doc in cross_docs]

        return cross_docs, cross_global_indices, cross_scores


if __name__ == "__main__": # Example usage and debugging output.
    query = "What are the health benefits of green tea?"
    documents = [
        "Green tea contains antioxidants that may help prevent cardiovascular disease.",
        "Coffee is also rich in antioxidants but can increase heart rate.",
        "Drinking water is essential for hydration.",
        "Green tea may also aid in weight loss and improve brain function."
    ]

    print("\nTF-IDF Reranking:")
    reranker = Reranker(type="tfidf")
    docs, indices, scores = reranker.rerank(query, documents)

    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")

    print("\nBag-of-Words Reranking:")
    reranker = Reranker(type="bow")
    docs, indices, scores = reranker.rerank(query, documents)

    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")

    print("\nCross-Encoder Reranking:")
    reranker = Reranker(type="cross_encoder")
    docs, indices, scores = reranker.rerank(query, documents)

    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")

    print("\nHybrid Reranking:")
    reranker = Reranker(type="hybrid")
    docs, indices, scores = reranker.rerank(query, documents)

    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")