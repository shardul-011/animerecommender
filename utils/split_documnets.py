from langchain_text_splitters import CharacterTextSplitter

def split_documents(docs, chunk_size=1000, overlap=100):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(docs)
