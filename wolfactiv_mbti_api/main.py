# --- imports ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os
import pandas as pd
from supabase import create_client, Client
from openai import OpenAI
from .recommender import get_u_final, calculate_similarities

# === chargement ENV (AVANT de lire os.getenv) ===
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# === clés/projets ===
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://oimzzeyjjovxdhuscmqw.supabase.co")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # ⚠️ service role (serveur seulement)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Client Supabase côté serveur (bypass RLS) : on peut l'utiliser pour tout ici
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

EXCEL_PATH = (BASE_DIR / "encoding_perso.xlsx").resolve()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # OK pour DEV, restreins ensuite
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- helpers ---
class QuizRequest(BaseModel):
    email: str

def persona_from_mbti(mbti: str) -> tuple[str, str]:
    mapping = {
        "INFP": ("Le/La Rêveur·se", "Suis l’étoile qui te guide."),
        "ENTJ": ("Le/La Stratège", "Construis, décide, avance."),
        # ... complète ta table
    }
    return mapping.get(
        mbti.upper(),
        ("Votre Personnage", "Votre citation personnalisée sera bientôt disponible.")
    )

def infer_mbti_from_answers(answers_text: str) -> str:
    prompt = f'''
Tu es un expert en psychologie MBTI.
Voici des réponses à des questions de personnalité :
"{answers_text}"

Indique uniquement le type MBTI (INFP, ESTJ, ENTP...).
'''
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Tu es un expert MBTI."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content.strip().upper()

def get_vector_from_mbti(mbti: str) -> dict:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Fichier Excel introuvable: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    if "MBTI" not in df.columns:
        raise ValueError("Colonne 'MBTI' absente dans le fichier.")
    row = df[df["MBTI"].str.upper() == mbti.upper()]
    if row.empty:
        raise ValueError(f"Type MBTI {mbti} non trouvé.")
    return row.iloc[0, 1:].astype(float).to_dict()

def adjust_vector(vector: dict, disliked: list, happy: str, strong: bool, strong_odor: str) -> dict:
    for note in disliked or []:
        if note in vector: vector[note] = 0.0
    if happy and happy in vector: vector[happy] += 1.0
    if strong and strong_odor and strong_odor in vector: vector[strong_odor] += 1.0
    return vector

@app.get("/health")
def health():
    return {"status": "ok"}

# --- endpoint principal ---
@app.post("/analyze_mbti")
def analyze_mbti(data: QuizRequest):
    try:
        # 1) Dernier quiz pour cet email
        r = supabase.table("quiz_results")\
            .select("*")\
            .eq("email", data.email)\
            .order("submitted_at", desc=True)\
            .limit(1)\
            .execute()
        if not r.data:
            return {"error": "Aucune donnée pour cet email"}

        user_data = r.data[0]
        answers = user_data.get("personality_answers", "")
        if isinstance(answers, list):
            answers = "\n".join(answers)
        if not answers:
            return {"error": "Réponses manquantes"}

        # 2) MBTI + vecteur ajusté
        mbti = infer_mbti_from_answers(answers)
        disliked = user_data.get("disliked_odors", []) or []
        happy = user_data.get("happy_memory_odor", "") or ""
        strong = bool(user_data.get("strong_memory", False))
        strong_odor = user_data.get("strong_memory_odor", "") or ""

        vector = get_vector_from_mbti(mbti)
        vector = adjust_vector(vector, disliked, happy, strong, strong_odor)

        # 3) Recos parfums
        u_final = get_u_final(list(vector.values()))
        top_perfumes = calculate_similarities(u_final)  # ← list[dict]

        # 4) Persona + radar
        character_name, quote = persona_from_mbti(mbti)
        # Si tu n'as pas de mapping spécifique pour le radar, tu peux envoyer le dict tel quel :
        radar_data = vector  # sinon construis un objet avec tes axes

        # 5) Historique dans resultats_mbti (facultatif)
        supabase.table("resultats_mbti").insert({
            "email": data.email,
            "mbti_result": mbti,
            "vector": list(vector.values())
        }).execute()

        # 6) Upsert dans results (lu par le front)
        payload = {
            "email": data.email,
            "mbti_result": mbti,
            "vector": list(vector.values()),
            "character_name": character_name,
            "quote": quote,
            "top_perfumes": top_perfumes,  # JSONB
            "radar_data": radar_data       # JSONB
        }
        supabase.table("results").upsert(payload, on_conflict="email").execute()

        # 7) Retour API (pour affichage immédiat)
        return {
            "email": data.email,
            "mbti": mbti,
            "vector": vector,
            "character_name": character_name,
            "quote": quote,
            "top_perfumes": top_perfumes,
            "radar_data": radar_data
        }

    except Exception as e:
        print("❌ Erreur dans analyze_mbti:", e)
        return {"error": "Erreur interne", "details": str(e)}

