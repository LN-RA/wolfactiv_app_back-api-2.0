from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from supabase import create_client, Client
from pathlib import Path
from openai import OpenAI
import pandas as pd
from .recommender import get_u_final, calculate_similarities
from pathlib import Path

BASE_DIR = Path(__file__).parent
EXCEL_PATH = (BASE_DIR / "encoding_perso.xlsx").resolve()  # fichier à côté de main.py

# Charger les variables d'environnement depuis .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Clés API
openai_api_key = os.getenv("OPENAI_API_KEY")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_url = "https://oimzzeyjjovxdhuscmqw.supabase.co"
supabase: Client = create_client(supabase_url, supabase_key)

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

# OUVERT pour les tests (avant d'avoir le front)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # à verrouiller ensuite
    allow_credentials=False,  # doit rester False si allow_origins == ["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
def health():
    return {"status": "ok"}

class QuizRequest(BaseModel):
    email: str

# Fonction d'inférence MBTI
def infer_mbti_from_answers(answers_text: str) -> str:
    prompt = f'''
Tu es un expert en psychologie MBTI.
Voici des réponses à des questions de personnalité :
"{answers_text}"

Analyse-les et indique uniquement le type MBTI (parmi INFP, ESTJ, ENTP...) sans explication.
''' 
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Tu es un expert MBTI."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip().upper()

# Récupération du vecteur MBTI
def get_vector_from_mbti(mbti: str) -> dict:
    try:
        if not EXCEL_PATH.exists():
            raise FileNotFoundError(f"Fichier Excel introuvable: {EXCEL_PATH}")

        df = pd.read_excel(EXCEL_PATH)  # ✅ chemin relatif

        if "MBTI" not in df.columns:
            raise ValueError("Colonne 'MBTI' absente dans le fichier.")

        row = df[df["MBTI"].str.upper() == mbti.upper()]
        if row.empty:
            raise ValueError(f"Type MBTI {mbti} non trouvé.")

        vector = row.iloc[0, 1:].astype(float).to_dict()
        return vector

    except Exception as e:
        raise RuntimeError(f"Erreur dans le chargement du vecteur MBTI : {e}")
# Ajustement du vecteur
def adjust_vector(vector: dict, disliked: list, happy: str, strong: bool, strong_odor: str) -> dict:
    for note in disliked:
        if note in vector:
            vector[note] = 0.0
    if happy in vector:
        vector[happy] += 1.0
    if strong and strong_odor in vector:
        vector[strong_odor] += 1.0
    return vector

# Endpoint principal
@app.post("/analyze_mbti")

def analyze_mbti(data: QuizRequest):
    try:
        print("🔍 Recherche quiz pour :", data.email)
        result = supabase.table("quiz_results")\
            .select("*")\
            .eq("email", data.email)\
            .order("submitted_at", desc=True)\
            .limit(1)\
            .execute()

        if not result.data:
            return {"error": "Aucune donnée pour cet email"}

        user_data = result.data[0]
        answers = user_data.get("personality_answers", "")
        if isinstance(answers, list):
            answers = "\n".join(answers)

        if not answers:
            return {"error": "Réponses manquantes"}

        mbti = infer_mbti_from_answers(answers)
        print("🧬 MBTI :", mbti)

        # Données sensorielles
        disliked = user_data.get("disliked_odors", [])
        happy = user_data.get("happy_memory_odor", "")
        strong = user_data.get("strong_memory", False)
        strong_odor = user_data.get("strong_memory_odor", "")

        # Vecteur brut et ajusté
        vector = get_vector_from_mbti(mbti)
        vector = adjust_vector(vector, disliked, happy, strong, strong_odor)
        print("📊 Vecteur final :", vector)

        # Sauvegarde
        supabase.table("resultats_mbti").insert({
            "email": data.email,
            "mbti_result": mbti,
            "vector": list(vector.values())
        }).execute()

        # Étape 1 : conversion en vecteur numpy
        vector_np = list(vector.values())

        # Étape 2 : projection avec la matrice S
        u_final = get_u_final(vector_np)

        # Étape 3 : calcul des similarités
        top_perfumes = calculate_similarities(u_final)

        # Réponse complète
        return {
            "email": data.email,
            "mbti": mbti,
            "vector": vector,
            "top_perfumes": top_perfumes
        }

    except Exception as e:
        print("❌ Erreur dans analyze_mbti:", e)
        return {
            "error": "Erreur interne",
            "details": str(e)
        }
    
    ...
