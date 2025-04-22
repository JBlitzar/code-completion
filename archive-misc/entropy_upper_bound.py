import os
from collections import Counter
import math


def calculate_topk_upper_bound(file_path, k=5):
    """
    Calculates the upper bound for top-k accuracy based on the tokenized text file.
    
    Args:
        file_path (str): Path to the input text file.
        k (int): Top-k accuracy value to compute.
    
    Returns:
        float: The upper bound for top-k accuracy.
    """
    try:
        # Read the file and tokenize by spaces
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        tokens = text.split()  # Tokenize by spaces
        
        # Calculate token frequencies
        token_counts = Counter(tokens)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return 0

        # Convert frequencies to probabilities
        token_probabilities = {token: count / total_tokens for token, count in token_counts.items()}
        
        # Calculate entropy
        entropy = -sum(p * math.log2(p) for p in token_probabilities.values())
        
        # Calculate top-k accuracy upper bound
        sorted_tokens = sorted(token_probabilities.items(), key=lambda x: x[1], reverse=True)
        top_k_prob = sum(prob for _, prob in sorted_tokens[:k])

        # Print entropy and top-k accuracy upper bound
        print(f"Entropy: {entropy:.4f} bits")
        print(f"Top-{k} Accuracy Upper Bound: {top_k_prob:.4f}")
        return top_k_prob
    except Exception as e:
        print(f"Error: {e}")
        return None


# Example usage
file_path = os.path.expanduser(
    "~/torch_datasets/github-python/corpus/data/corpus_processed.txt"
)

top_k_accuracy = calculate_topk_upper_bound(file_path, k=5)
if top_k_accuracy is not None:
    print(f"Upper Bound for Top-5 Accuracy: {top_k_accuracy:.4f}")
