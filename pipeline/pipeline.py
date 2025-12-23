
from src.vector_store import VectorStore
from src.recommender import AnimeRecommender
from config.config import GROQ_API_KEY, MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger=get_logger(__name__)

class AnimeRecommenderPipeline():
    def __init__(self,persist_directory="chroma"):
        try:
            logger.info("Initializing Animerecommnder Pipeline")
            
            vector_store=VectorStore(csv_path="", persist_directory=persist_directory)
            retriever=vector_store.load_vectorstore().as_retriever()
            
            self.reccomender=AnimeRecommender(retriever,GROQ_API_KEY,MODEL_NAME)
            
            logger.info("AnimeRecommender Pipeline Initialized Successfully")
            
        except Exception as e:
            logger.error(f"error initializing AnimeRecommender Pipeline {str(e)}")
            raise CustomException("Error initializing AnimeRecommender Pipeline",e)
        
    def recommend(self,query:str)->str:
            try:
                logger.info("Getting anime recommendation")
                recommendation=self.reccomender.get_recommendation(query)
                if hasattr(recommendation, "content"):
                    return recommendation.content
                elif isinstance(recommendation, dict):
                    return recommendation.get("result", "")
                else:
                    return str(recommendation)
                
            except Exception as e:
                logger.error(f"error getting anime recommendation {str(e)}")
                raise CustomException("Error getting anime recommendation",e)