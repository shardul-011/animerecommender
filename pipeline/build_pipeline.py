from utils.logger import get_logger
from utils.custom_exception import CustomException
from dotenv import load_dotenv
from src.recommender import AnimeRecommender
from src.vector_store import VectorStore
from src.data_loader import AnimeDataLoader

load_dotenv()
logger=get_logger(__name__)

def main():
    try:
        logger.info("Starting to Built Pipeline")
        
        loader=AnimeDataLoader(original_file="animerecomender/data/anime_with_synopsis.csv",processed_file="animerecomender/data/processed_anime.csv")
        processes_Csv=loader.load_and_process()
        
        logger.info("Data Loaded and Processed Successfully")
        
        vector_store=VectorStore(processes_Csv)
        vector_store.build_and_save_vectorstore()
        

        logger.info("Vector Store Built and Saved Successfully")
        
    except Exception as e:
        logger.error(f"Error in Building Pipeline: {str(e)}")
        raise CustomException("Error in Building Pipeline",e)