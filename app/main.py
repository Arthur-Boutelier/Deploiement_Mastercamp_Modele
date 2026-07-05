"""
API d'analyse de radiographies thoraciques.

Expose un point d'entree /predict qui recoit une radiographie, une vue, un age
et un sexe, puis renvoie :
  - la detection binaire d'anomalie,
  - le score et le niveau de confiance,
  - une heatmap des zones determinantes,
  - les messages d'avertissement adaptes.
"""

import logging

from fastapi import FastAPI, File, Form, UploadFile, HTTPException

from .. import config
from .model import modele
from .preprocessing import pretraiter_image, charger_image_depuis_octets
from .heatmap import generer_heatmap
from .schemas import ReponsePrediction, ReponseSante

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xray_api")

app = FastAPI(
    title="Assistant Radiologue Virtuel",
    description="Outil pedagogique de detection d'anomalie sur radiographie thoracique.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Correspondances des entrees
# ---------------------------------------------------------------------------
VUES = {
    "AP": "frontale AP",
    "PA": "frontale PA",
    "FRONTAL": "frontale",
    "LATERAL": "laterale",
}

SEXES = {
    "M": "homme", "H": "homme", "HOMME": "homme",
    "F": "femme", "FEMME": "femme",
}


@app.on_event("startup")
def demarrage():
    logger.info("Chargement du modele...")
    modele.charger()
    logger.info("Modele charge sur %s.", modele.device)


@app.get("/health", response_model=ReponseSante)
def sante():
    return ReponseSante(
        statut="ok",
        modele_charge=modele.model is not None,
        peripherique=modele.device,
    )


@app.post("/predict", response_model=ReponsePrediction)
async def predire(
    fichier: UploadFile = File(..., description="Image de la radiographie"),
    vue: str = Form(..., description="Vue : AP, PA, FRONTAL ou LATERAL"),
    age: int = Form(..., description="Age du patient"),
    sexe: str = Form(..., description="Sexe : M ou F"),
    inclure_heatmap: bool = Form(False),
    inclure_probabilites: bool = Form(False),
):
    # --- Validation des entrees ---
    vue_norm = VUES.get(vue.strip().upper())
    if vue_norm is None:
        raise HTTPException(status_code=400, detail=f"Vue invalide : {vue}. Attendu : AP, PA, FRONTAL, LATERAL.")

    sexe_norm = SEXES.get(sexe.strip().upper())
    if sexe_norm is None:
        raise HTTPException(status_code=400, detail=f"Sexe invalide : {sexe}. Attendu : M ou F.")

    if age < 0 or age > 120:
        raise HTTPException(status_code=400, detail="Age hors des bornes acceptables (0-120).")

    if modele.model is None:
        raise HTTPException(status_code=503, detail="Le modele n'est pas encore charge.")

    # --- Lecture et pretraitement de l'image ---
    try:
        octets = await fichier.read()
        image = charger_image_depuis_octets(octets)
        image = pretraiter_image(image)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Image illisible : {exc}")

    # --- Prediction ---
    score, _details = modele.score_anomalie(image, vue_norm, sexe_norm, age)
    anomalie = score >= config.SEUIL_ANOMALIE
    confiance = score if anomalie else 1.0 - score

    reponse = ReponsePrediction(
        anomalie_detectee=bool(anomalie),
        score_anomalie=round(float(score), 4),
        niveau_de_confiance=round(float(confiance), 4),
        seuil_utilise=config.SEUIL_ANOMALIE,
        strategie=config.STRATEGIE_ANOMALIE,
        avertissement=config.AVERTISSEMENT_GENERAL,
        alerte=config.ALERTE_ANOMALIE if anomalie else config.MESSAGE_NORMAL,
    )

    # --- Probabilites detaillees (optionnel) ---
    if inclure_probabilites:
        probs = modele.probabilites_par_classe(image, vue_norm, sexe_norm, age)
        reponse.probabilites_par_classe = {k: round(v, 4) for k, v in probs.items()}

    # --- Heatmap (optionnel, hors budget temps critique) ---
    if inclure_heatmap:
        try:
            heatmap_b64, grille = generer_heatmap(modele, image, vue_norm, sexe_norm, age, score)
            reponse.heatmap_base64 = heatmap_b64
            reponse.heatmap_grille = grille
        except Exception as exc:  # noqa: BLE001
            logger.warning("Echec de generation de la heatmap : %s", exc)

    return reponse


@app.post("/heatmap")
async def heatmap_seule(
    fichier: UploadFile = File(..., description="Image de la radiographie"),
    vue: str = Form(...),
    age: int = Form(...),
    sexe: str = Form(...),
):
    """Endpoint dedie a la heatmap, appele separement de la decision.

    La decision (/predict) reste ainsi rapide et respecte le budget de 10 s ;
    la heatmap, plus couteuse, est demandee a la demande par le client.
    """
    vue_norm = VUES.get(vue.strip().upper())
    sexe_norm = SEXES.get(sexe.strip().upper())
    if vue_norm is None or sexe_norm is None:
        raise HTTPException(status_code=400, detail="Vue ou sexe invalide.")
    if modele.model is None:
        raise HTTPException(status_code=503, detail="Le modele n'est pas encore charge.")

    try:
        octets = await fichier.read()
        image = charger_image_depuis_octets(octets)
        image = pretraiter_image(image)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Image illisible : {exc}")

    score, _ = modele.score_anomalie(image, vue_norm, sexe_norm, age)
    heatmap_b64, grille = generer_heatmap(modele, image, vue_norm, sexe_norm, age, score)
    return {"heatmap_base64": heatmap_b64, "heatmap_grille": grille}
