# scripts/ingest_data.py
import os
import json
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader
)
from langchain_text_splitters import CharacterTextSplitter

from src.data_loader import AnimeDataLoader
from src.vector_store import VectorStore

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
vector_store = VectorStore()


def ingest_csv(path):
    processed_path = os.path.join(
        PROCESSED_DIR, f"processed_{os.path.basename(path)}"
    )

    processed_file = AnimeDataLoader(
        original_file=path,
        processed_file=processed_path
    ).load_and_process()

    docs = CSVLoader(processed_file, encoding="utf-8").load()
    return splitter.split_documents(docs)


def ingest_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    if isinstance(data, list):
        for item in data:
            text = " ".join(f"{k}: {v}" for k, v in item.items())
            docs.append(Document(page_content=text))
    else:
        text = " ".join(f"{k}: {v}" for k, v in data.items())
        docs.append(Document(page_content=text))

    return splitter.split_documents(docs)


def ingest_pdf(path):
    docs = PyPDFLoader(path).load()
    return splitter.split_documents(docs)


def ingest_txt(path):
    docs = TextLoader(path, encoding="utf-8").load()
    return splitter.split_documents(docs)


def ingest_file(path):
    if path.endswith(".csv"):
        return ingest_csv(path)
    if path.endswith(".json"):
        return ingest_json(path)
    if path.endswith(".pdf"):
        return ingest_pdf(path)
    if path.endswith(".txt"):
        return ingest_txt(path)
    return []


if __name__ == "__main__":
    for file in os.listdir(RAW_DIR):
        full_path = os.path.join(RAW_DIR, file)
        documents = ingest_file(full_path)

        if documents:
            vector_store.add_documents(documents)
            print(f"Ingested: {file}")
