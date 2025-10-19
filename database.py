import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
import shutil
import nltk
import json
import numpy as np

from torch import embedding

DATASET_PATH = r"/home/thaituan_uit/RAG_UIT_project/news.json"
CHROMA_PATH = "chroma_database"

model_name = "hiieu/halong_embedding"
model_kwargs = {
        'device': 'cuda',
        'trust_remote_code':True
}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100,
    length_function = len,
    add_start_index = True
)

def main():
    docs = load_docs()
    chunks = splitting_text(docs)
    save_database(chunks)

def split_into_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def load_docs():
    try:
        documents = []
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            list_keys = ('articles', 'data', 'items', 'news', 'documents')
            records = None
            for key in list_keys:
                if key in data and isinstance(data[key], list):
                    records = data[key]
                    break
            if records is None:
                if all(isinstance(v, dict) for v in data.values()):
                    records = list(data.values())
                else:
                    records = [data]
        else:
            records = [data]

        for idx, rec in enumerate(records):
            content = ''
            metadata = {"source": DATASET_PATH, "ids": idx}

            if isinstance(rec, dict):
                # Try common text fields
                for field in ('content', 'text', 'body', 'article', 'description', 'summary', 'full_text', 'url', 'date', 'title'):
                    if field in rec and isinstance(rec[field], str) and rec[field].strip():
                        content = rec[field].strip()
                        break

                # If no single large text field found, join all string fields
                if not content:
                    parts = [v.strip() for v in rec.values() if isinstance(v, str) and v.strip()]
                    content = "\n".join(parts)

                for meta_key in ('title', 'url', 'date'):
                    if meta_key in rec:
                        metadata[meta_key] = rec[meta_key]
            else:
                content = str(rec)

            documents.append(Document(page_content=content, metadata=metadata))

        return documents
    except Exception as e:
        print(f"An error occurred while loading documents: {e}")
        return []
    
def semantic_chunking(text, threshold=0.75, model="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name, device=model_kwargs['device'])
    sents = nltk.sent_tokenize(text)
    embeddings = model.encode(sents)
    
    chunks = []
    curr_chunk = [sents[0]]
    res_chunks = []
    
    for i in range(1, len(sents)):
        cos_sim = cosine_similarity(np.matrix(embeddings[i-1]), np.matrix(embeddings[i]))[0][0]
        
        if cos_sim > threshold:
            curr_chunk.append(sents[i])
        else:
            curr_chunk.append(" ".join(curr_chunk))
            curr_chunk = [sents[i]]
    
    chunks.append(" ".join(curr_chunk))
    
    for chunk in chunks:
        if len(chunk) > 1000:
            res_chunks.extend(text_splitter.split_text(chunk))
        else:
            res_chunks.append(chunk)
    
    return res_chunks

def splitting_text(documents):
    chunks = text_splitter.split_documents(documents)
    print(f"Split  {len(documents)} documents into {len(chunks)} chunks")
    return chunks

def save_database(chunks):
    if(os.path.exists(CHROMA_PATH)):
        shutil.rmtree(CHROMA_PATH)

    embedding = HuggingFaceEmbeddings(model_name = model_name, model_kwargs = model_kwargs)

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding, collection_metadata={"hnsw:space": "cosine"})

    max_batch_size = 5461

    for batch in split_into_batches(chunks, max_batch_size):
        db.add_documents(batch)
        
    db = Chroma.from_documents(
        chunks,
        embedding,
        persist_directory=CHROMA_PATH
    )

    print(f"Saved {len(chunks)} to db {CHROMA_PATH}")

if __name__ == "__main__":
    main()

