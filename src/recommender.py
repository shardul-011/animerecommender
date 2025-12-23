# src/recommender.py

from langchain_groq import ChatGroq
from src.prompt_template import get_anime_prompt

class AnimeRecommender:
    def __init__(self, retriever, api_key: str, model_name: str):
        self.llm = ChatGroq(api_key=api_key, model_name=model_name, temperature=0)
        self.prompt = get_anime_prompt()
        self.retriever = retriever

        # Correct RAG chain for LangChain 1.1.2
        self.qa_chain = (
            {
                "context": lambda x: self.retriever.invoke(x["question"]),
                "question": lambda x: x["question"]
            }
            | self.prompt
            | self.llm
        )

    def get_recommendation(self, question: str) -> str:
        result = self.qa_chain.invoke({"question": question})
        return result
