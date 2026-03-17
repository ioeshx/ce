import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    parser = argparse.ArgumentParser(description="Find top K concepts similar to target concepts.")
    parser.add_argument("--input_csv", type=str, default="data/instance.csv", help="Path to input CSV")
    parser.add_argument("--output_csv", type=str, default="data/top{}_instance.csv", help="Path to output CSV")
    parser.add_argument("--targets", type=str, nargs="+", default=["Snoopy", "Mickey", "Spongebob", "Hello Kitty", "Pikachu"], help="Target concepts separated by space (default: Snoopy Mickey Spongebob)")
    parser.add_argument("--k", type=int, default=70, help="Number of top concepts to save")
    parser.add_argument("--model", type=str, default="clip-ViT-B-32", help="Sentence-transformer model to use (default: clip-ViT-B-32)")

    args = parser.parse_args()
    args.output_csv = args.output_csv.format(args.k)  # Format output path with K

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"Reading instance concepts from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    if 'concept' not in df.columns:
        raise ValueError("CSV must contain a column named 'concept'")
    concepts = df['concept'].astype(str).tolist()

    print("Encoding concepts...")
    concept_embeddings = model.encode(concepts, show_progress_bar=True)
    print(f"Encoding target concepts: {args.targets}")
    target_embeddings = model.encode(args.targets)

    # Calculate cosine similarity between all concepts and all target concepts
    # Similarities shape: (num_concepts, num_targets)
    similarities = cosine_similarity(concept_embeddings, target_embeddings)
    
    # Calculate the average similarity score across all target concepts 
    avg_similarities = np.mean(similarities, axis=1)

    # Add the average similarities to the DataFrame
    df['similarity_score'] = avg_similarities

    # Sort DataFrame by the average similarity score in descending order
    df_sorted = df.sort_values(by='similarity_score', ascending=False)
    
    # Extract the top K rows
    df_top_k = df_sorted.head(args.k)
    
    # Output back to CSV matching the original format (e.g. keeping 'id' and 'concept')
    # If you want to include the similarity score in output, we can leave it. 
    # Here we drop similarity_score to keep the format strictly identical to the original instance.csv if favored,
    # but let's keep the original columns from the source intact. 
    original_cols = [col for col in df.columns] # if col != 'similarity_score'
    df_top_k[original_cols].to_csv(args.output_csv, index=False)

    print(f"Top {args.k} concepts successfully saved to {args.output_csv}\n")
    print(df_top_k[['concept', 'similarity_score']])

if __name__ == "__main__":
    main()
