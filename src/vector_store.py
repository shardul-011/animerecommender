import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

class VectorStore():
    def __init__(self,csv_path:str, persist_directory:str="chroma_db"):
        self.csv_path = csv_path
        self.persist_directory = persist_directory
        self.embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
    def build_and_save_vectorstore(self):
        loader=CSVLoader(
            file_path=self.csv_path,
            encoding="utf-8",
            metadata_columns=[]
        )
        data=loader.load()
        
        text_splitter=CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
            )
        
        texts=text_splitter.split_documents(data)
        
        db=Chroma.from_documents(texts,self.embedding_model,persist_directory=self.persist_directory)
        db.persist()  ##to saave locally in our system
        
    def load_vectorstore(self):
        if not os.path.exists(self.persist_directory):
            raise RuntimeError(
            f"Vector store not found at {self.persist_directory}. "
            "Run build_and_save_vectorstore() first."
        )
        return Chroma(
            persist_directory=self.persist_directory,embedding_function=self.embedding_model)
        
    
    
    