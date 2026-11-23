import os
import re
from typing import List
from glob import glob
import nltk

try:
    nltk.data.find('tokenizers/punkt') # Ensure NLTK sentence tokenizer is available.
except LookupError:
    nltk.download('punkt')

class DocumentProcessing:
    """
    A class used for processing documents including reading files,
    trimming whitespace, and splitting documents into sentence-based or
    fixed-length character chunks.
    """

    def __init__(self):
        """
        Initializes the DocumentProcessing class.
        Currently, no instance attributes are used.
        """
        pass

    def __read_text_file(self, file_path: str) -> str:
        """
        Reads the content of a UTF-8 encoded text file.

        Parameters:
            file_path (str): The path to the text file to be read.

        Returns:
            str: The content of the file if successfully read;
                 otherwise, an error message describing the issue.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            return f"The file at {file_path} was not found."
        except Exception as e:
            return f"An error occurred: {e}"

    def trim_white_space(self, text: str) -> str:
        """
        Removes excessive whitespace from the input text by reducing all
        whitespace sequences to a single space and trimming leading/trailing spaces.

        Parameters:
            text (str): The raw text to be cleaned.

        Returns:
            str: A cleaned string with normalized spacing.
        """
        return ' '.join(text.split())

    def sentence_chunking(self, document_filename: str, num_sentences: int, overlap_size: int = 0) -> List[str]:
        """
        Splits a document into sentence-based chunks, each containing a fixed number
        of sentences, optionally with overlap.

        Parameters:
            document_filename (str): Path to the document to be chunked.
            num_sentences (int): Number of sentences per chunk.
            overlap_size (int): Number of overlapping sentences between consecutive chunks.

        Returns:
            List[str]: A list of text chunks, each made up of consecutive sentences.
        """
        text = self.__read_text_file(document_filename)

        if isinstance(text, str): # If reading failed, return a single-item list with error message.
            text = self.trim_white_space(text)
            sentences = nltk.sent_tokenize(text)

            chunks = []
            i = 0

            while i < len(sentences):
                chunk = ' '.join(sentences[i:i + num_sentences]) # Create a chunk of num_sentences.
                chunks.append(chunk)
                i += (num_sentences - overlap_size) # Slide the window by (num_sentences - overlap_size).

            return chunks

        return [text] # Return error message as a list.

    def fixed_length_chunking(self, document_filename: str, chunk_size: int, overlap_size: int = 2) -> List[str]:
        """
        Splits a document into fixed-length character chunks, with optional overlap.

        Parameters:
            document_filename (str): Path to the document to be chunked.
            chunk_size (int): Number of characters in each chunk.
            overlap_size (int): Number of overlapping characters between chunks.

        Returns:
            List[str]: A list of character-based text chunks.
        """
        if overlap_size >= chunk_size:
            raise ValueError("Overlap size must be smaller than chunk size.")

        text = self.__read_text_file(document_filename)

        if not isinstance(text, str): # If reading failed, return a single-item list with error message.
            return [text]

        text = self.trim_white_space(text)

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            if end >= text_length: # Stop if the chunk reaches or exceeds the end of the document.
                break

            start = end - overlap_size # Move the window forward with specified overlap.

        return chunks
    
    def chunk_text(self, document_filename: str, strategy: str = "sentence", **kwargs) -> List[str]:
        """
        Chunk text from a file using the specified strategy.

        Parameters:
            document_filename (str): Path to the file.
            strategy (str): Chunking strategy: "sentence" or "fixed".
            kwargs: Additional arguments for chunking methods.

        Returns:
            List[str]: List of text chunks.
        """
        if strategy == "sentence":
            num_sentences = kwargs.get("num_sentences", 5)
            overlap_size = kwargs.get("overlap_size", 0)
            return self.sentence_chunking(document_filename, num_sentences, overlap_size)
        elif strategy == "fixed-length":
            chunk_size = kwargs.get("chunk_size", 300)
            overlap_size = kwargs.get("overlap_size", 2)
            return self.fixed_length_chunking(document_filename, chunk_size, overlap_size)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

if __name__ == "__main__":
    processing = DocumentProcessing()

    sentence_chunks = processing.sentence_chunking(
        "storage/corpus/S08_set3_a1.txt.clean", num_sentences=5, overlap_size=3) # Example usage: sentence-based chunking.
    print("Sentence-based chunks:")

    for idx, chunk in enumerate(sentence_chunks):
        print(f"[{idx}] {chunk}\n")

    
    fixed_chunks = processing.fixed_length_chunking(
        "storage/corpus/S08_set3_a1.txt.clean", chunk_size=300, overlap_size=50) # Example usage: fixed-length character chunking.
    print("Character-based chunks:")

    for idx, chunk in enumerate(fixed_chunks):
        print(f"[{idx}] {chunk[:75]}...\n") # Preview only first 75 characters.