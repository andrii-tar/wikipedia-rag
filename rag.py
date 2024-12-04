import torch
import pandas as pd
from llm import invoke_model
from sentence_transformers import SentenceTransformer

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

from rank_bm25 import BM25Okapi

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        return "Error: File not found."
    except Exception as e:
        return f"Error: {e}"

def split_text(text, chunk_size=500) -> [str]:
    overlap_size = int(chunk_size * 0.1)

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(' '.join(words[start:end]).lower())
        start += chunk_size - overlap_size

    return chunks

def embed_chunks(chunks):
    return model.encode(chunks)

def retrieve_semantic(query, embeddings, chunks, top_k=5) -> tuple:
    query_embedding = embed_chunks([query.lower()])[0]

    similarities = model.similarity([query_embedding], embeddings)

    values, indices = torch.topk(similarities, k=top_k)

    top_k_chunks = [chunks[i] for i in indices.flatten().tolist()]

    chunks_info = []
    for i in indices.flatten().tolist():
        chunks_info.append({
            "chunk": chunks[i],
            "similarity": similarities[0, i].item(),
            "chunk_id": i
        })

    return "\n\n".join(top_k_chunks), chunks_info

def retrieve_bm25(query, chunks, top_k=5):
    tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    tokenized_question = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_question)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    context = " ".join([chunks[i] for i in top_indices])

    chunks_info = []
    for idx in top_indices:
        chunks_info.append({
            "chunk_id": idx,
            "score": int(scores[idx]),
            "content": chunks[idx],
        })

    return context, chunks_info

def retrieve_context(query, embeddings, chunks, top_k=5, is_retrieve_bm25=False) -> tuple:
    if(is_retrieve_bm25):
        return retrieve_bm25(query, chunks, top_k)
    else:
        return retrieve_semantic(query, embeddings, chunks, top_k)


def process_and_split_csv(input_csv, chunk_size=500, output_csv="chunked_articles.csv"):
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} rows from {input_csv}.")

        chunked_data = []

        for _, row in df.iterrows():
            article_id = row['ID']
            body = row['Body']

            if not isinstance(body, str) or not body.strip():
                continue

            chunks = split_text(body, chunk_size)
            for chunk in chunks:
                chunked_data.append({"ID": article_id, "Chunk": chunk})

        chunked_df = pd.DataFrame(chunked_data)
        print(f"Created DataFrame with {len(chunked_df)} chunks.")

        chunked_df.to_csv(output_csv, index=False)

        return chunked_df, chunked_df["Chunk"].tolist()

    except Exception as e:
        print(f"Error: {e}")