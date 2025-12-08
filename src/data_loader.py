import pandas as pd

class AnimeDataLoader:
    def __init__(self,original_file:str,processed_file:str):
        self.original_file = original_file
        self.processed_file = processed_file
    
    def load_and_process(self):
        
        df=pd.read_csv(self.original_file,encoding="utf-8", on_bad_lines='skip').dropna()
        missing={'Name','Genres','sypnopsis'}
        processed_column=missing-set(df.columns)
        
        if missing:
            raise ValueError(f"Missing columns in the dataset: {processed_column}")
        
        df['combined_info']=(
            "Title:" + df['Name'] + "Overview:" + df['synopsis'] + "Genres:" + df['Genres']
        )
        
        df[['combined_info']].to_csv(self.processed_file,index=False,header=True,encoding="utf-8")
        return self.processed_file
        
        
        
    