from config import GROQ_API_KEY, MODEL_NAME
from groq import Groq

def test_groq_api_key():
    
    if GROQ_API_KEY is None or GROQ_API_KEY == "":
        raise ValueError("GROQ_API_KEY is not set properly.")
    else:
        print("GROQ_API_KEY is set properly.")
        
        client=Groq(api_key=GROQ_API_KEY)
        
        response=client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role":"user","content":"Hello, how are you?"}
            ]
        )
        
        print("Model response:", response.choices[0].message.content)
test_groq_api_key()