import numpy as np
import math
from collections import Counter
from typing import List, Union
import re
import unicodedata

class TF_IDF:
    """
    A Term Frequency–Inverse Document Frequency (TF-IDF) transformer.

    This class supports fitting on a corpus to compute IDF values, and
    transforming documents into TF-IDF vector representations.
    """

    def __init__(self):
        """
        Initializes the TF_IDF transformer.

        Attributes:
            vocabulary_ (dict or None): A mapping of terms to column indices in the TF-IDF matrix.
            idf_ (np.ndarray or None): An array of inverse document frequency values for each term.
        """
        self.vocabulary_ = None
        self.idf_ = None

    def _tokenize(self, document: str) -> List[str]:
        """
        Tokenizes a document into lowercase alphanumeric words.

        Parameters:
            document (str): The text document to tokenize.

        Returns:
            List[str]: A list of word tokens.
        """
        return re.findall(r'\b\w+\b', document.lower())

    def fit(self, corpus: List[str]) -> "TF_IDF":
        """
        Learns vocabulary and computes the smoothed IDF values from the given corpus.

        Parameters:
            corpus (List[str]): A list of documents used to compute IDF values.

        Returns:
            TF_IDF: The instance itself (for chaining or reuse).
        """
        doc_count = {} # Document frequency per token.
        vocab = set() # All unique tokens.
        
        for doc in corpus: # Count how many documents contain each token.
            tokens = set(self._tokenize(doc)) # Unique tokens in document.
            vocab.update(tokens)
            for token in tokens:
                doc_count[token] = doc_count.get(token, 0) + 1

        self.vocabulary_ = {word: i for i, word in enumerate(sorted(vocab))} # Assign index to each vocabulary term.
        num_docs = len(corpus)

        self.idf_ = np.zeros(len(self.vocabulary_))

        for word, idx in self.vocabulary_.items(): # Compute smoothed IDF for each term (IDF = log((1 + N) / (1 + df)) + 1).
            df = doc_count.get(word, 0)
            self.idf_[idx] = math.log((1 + num_docs) / (1 + df)) + 1 

        return self

    def transform(self, documents: Union[List[str], str]) -> np.ndarray:
        """
        Transforms one or more documents into TF-IDF vectors.

        Uses term frequency as raw counts (not normalized), multiplied by precomputed IDF.

        Parameters:
            documents (Union[List[str], str]): A single string or a list of documents.

        Returns:
            np.ndarray: A 2D array (or 1D if single input) representing the TF-IDF vectors.
        """
        if self.vocabulary_ is None or self.idf_ is None:
            raise AttributeError("TF_IDF instance is not fitted yet. Call 'fit' first.")

        single_input = False

        if isinstance(documents, str): # Wrap single document in a list.
            documents = [documents]
            single_input = True

        tfidf_matrix = np.zeros((len(documents), len(self.vocabulary_))) # Initialize TF-IDF matrix.

        for i, doc in enumerate(documents): # Count the term frequency.
            tokens = self._tokenize(doc)
            tf_counter = Counter(tokens)
            total_terms = len(tokens)

            for token, count in tf_counter.items(): # Compute TF-IDF for each term in vocabulary.
                if token in self.vocabulary_:
                    tf = count # Raw term frequency.
                    idx = self.vocabulary_[token]
                    tfidf_matrix[i, idx] = tf * self.idf_[idx]

        return tfidf_matrix[0] if single_input else tfidf_matrix # Return a single vector if input was a single string.

if __name__ == "__main__":
    
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Never jump over the lazy dog quickly.",
        "A quick movement of the enemy will jeopardize six gunboats.",
    ] # Sample corpus for building TF-IDF model.

    tfidf = TF_IDF().fit(corpus) # Instantiate and fit TF-IDF transformer.
    vec = tfidf.transform("The quick dog jumps high over the lazy fox.") # Transform a new test document into its TF-IDF vector.

    print("TF-IDF vector:\n", vec) # Display the result.