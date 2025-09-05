from pathlib import Path
import os
import numpy as np
import pandas as pd
import difflib

# --- Localisation portable des fichiers de données ---------------------------
PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", PKG_DIR / "data"))

S_MATRIX_PATH = Path(os.getenv("S_MATRIX_PATH", DATA_DIR / "similarite_matrice.csv"))
PARFUMS_PATH  = Path(os.getenv("PARFUMS_PATH",  DATA_DIR / "parfums_enrichi.csv"))
ENCODING_XLSX = Path(os.getenv("ENCODING_XLSX", DATA_DIR / "encoding_perso.xlsx"))  # si utilisé ailleurs

def read_csv_robust(path: Path, **kwargs) -> pd.DataFrame:
    """
    Lecture CSV robuste:
      - auto-détecte le séparateur (; ou ,)
      - tente utf-8-sig puis latin-1
    """
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin-1", **kwargs)

# -----------------------------------------------------------------------------


def get_u_final(u_vector):
    print("📥 u_vector (input):", u_vector)

    # Chargement de la matrice de similarité (index en 1ère colonne)
    S_df = read_csv_robust(S_MATRIX_PATH, index_col=0)
    S = S_df.to_numpy()

    # u en float, aplati
    u = np.array(u_vector, dtype=float).reshape(-1)

    print("✅ Matrice S (shape):", S.shape)
    print("✅ Vecteur u (shape):", u.shape)

    if S.shape[1] != u.shape[0]:
        raise ValueError(f"Incompatibilité dimensions: S.shape={S.shape}, u.shape={u.shape}")

    return S @ u


def _pick_col(df: pd.DataFrame, candidates):
    """Choisit la première colonne existante parmi candidates (insensible à la casse),
       sinon essaie un fuzzy match sur le premier candidat."""
    # mapping minuscule -> nom réel
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    match = difflib.get_close_matches(candidates[0], df.columns, n=1, cutoff=0.6)
    return match[0] if match else None


def calculate_similarities(u_final):
    # Chargement du fichier des parfums enrichis
    df_parfums = read_csv_robust(PARFUMS_PATH)

    # Nettoyage des colonnes (espaces et espaces insécables)
    df_parfums.columns = (
        df_parfums.columns
        .str.strip()
        .str.replace(r'[\u202f\u00a0]', '', regex=True)
    )

    # Colonnes utiles (avec tolérance de noms)
    brand_col = _pick_col(df_parfums, ['Marque', 'brand'])
    name_col  = _pick_col(df_parfums, ['Nom du Parfum', 'Nom', 'Parfum', 'name'])
    image_col = _pick_col(df_parfums, ['Image', 'images parfums', 'images_parfums', 'image'])
    url_col   = _pick_col(df_parfums, ['URL', 'Lien de redirection', 'lien', 'link', 'url'])

    # Familles olfactives à matcher (tolérance sur les noms réels de colonnes)
    familles_olfactives = [
        'Epicee', 'Ambree', 'Boisee Mousse', 'Hesperidee', 'Florale', 'Aromatique',
        'Cuir', 'Boisee', 'Balsamique', 'Florale Fraiche', 'Verte', 'Florale Rosee',
        'Musquee', 'Fruitee', 'Florale Poudree', 'Marine', "Fleur D'Oranger",
        'Conifere Terpenique', 'Aldehydee'
    ]

    correspondance = {}
    colonnes_fichier = df_parfums.columns.tolist()
    for famille in familles_olfactives:
        match = difflib.get_close_matches(famille, colonnes_fichier, n=1, cutoff=0.6)
        if match:
            correspondance[famille] = match[0]

    if not correspondance:
        raise ValueError("Aucune colonne de familles olfactives trouvée dans le CSV.")

    note_columns = df_parfums[[v for v in correspondance.values()]]

    # Adapter la taille de u_final si besoin (tronquer / compléter par des zéros)
    ufinal = np.array(u_final, dtype=float).reshape(-1)
    n = note_columns.shape[1]
    if ufinal.shape[0] < n:
        ufinal = np.pad(ufinal, (0, n - ufinal.shape[0]))
    elif ufinal.shape[0] > n:
        ufinal = ufinal[:n]

    # Similarité cosinus
    def cosine_similarity(v1, v2):
        v1 = pd.to_numeric(v1, errors='coerce').fillna(0).values.astype(float)
        v2 = v2.astype(float)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        return 0.0 if norm1 == 0 or norm2 == 0 else float(np.dot(v1, v2) / (norm1 * norm2))

    similarities = []
    for i, row in note_columns.iterrows():
        sim = cosine_similarity(row, ufinal)

        marque = str(df_parfums.loc[i, brand_col]) if brand_col else ""
        nom    = str(df_parfums.loc[i, name_col])  if name_col  else ""
        parfum_name = f"{marque} - {nom}".strip(" -")

        image = df_parfums.loc[i, image_col] if image_col else ""
        url   = df_parfums.loc[i, url_col]   if url_col   else ""

        similarities.append({
            "parfum": parfum_name,
            "similarité": round(sim * 100, 2),
            "image": image,
            "url": url
        })

    similarities.sort(key=lambda x: x["similarité"], reverse=True)
    return similarities[:5]
