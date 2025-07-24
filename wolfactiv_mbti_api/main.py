

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import openai
from supabase import create_client, Client

# Charger les variables d'environnement depuis .env
load_dotenv()

# Récupérer les clés API
openai.api_key = os.getenv("OPENAI_API_KEY")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_url = "https://oimzzeyjjovxdhuscmqw.supabase.co"
supabase: Client = create_client(supabase_url, supabase_key)

# Initialiser FastAPI
app = FastAPI()

# Modèle Pydantic pour la requête
class QuizRequest(BaseModel):
    email: str

# Fonction d'inférence MBTI avec OpenAI
def infer_mbti_from_answers(answers_text: str) -> str:
    prompt = f'''
Tu es un expert en psychologie MBTI.
Voici des réponses à des questions de personnalité :
"{answers_text}"

Analyse-les et indique uniquement le type MBTI (parmi INFP, ESTJ, ENTP...) sans explication.
'''
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Tu es un expert MBTI."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message["content"].strip()

# Endpoint principal
@app.post("/analyze_mbti")
def analyze_mbti(data: QuizRequest):
    result = supabase.table("quiz_results").select("*").eq("email", data.email).order("submitted_at", desc=True).limit(1).execute()
    
    if not result.data:
        return {"error": "Aucune donnée trouvée pour cet email"}
    
    user_data = result.data[0]
    answers = user_data.get("personality_answers", "")
    
    if not answers:
        return {"error": "Champs 'personality_answers' vide"}
    
    mbti = infer_mbti_from_answers(answers)

    return {
        "email": data.email,
        "mbti": mbti,
        "original_answers": answers
    }

# Petit endpoint de test
@app.get("/")
def read_root():
    return {"message": "Hello from Wolfactiv MBTI API"}
