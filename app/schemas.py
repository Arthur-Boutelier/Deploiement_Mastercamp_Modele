"""
Schemas de reponse de l'API (Pydantic).
"""

from typing import Dict, List, Optional
from pydantic import BaseModel


class ReponsePrediction(BaseModel):
    anomalie_detectee: bool
    score_anomalie: float          # probabilite d'anomalie, entre 0 et 1
    niveau_de_confiance: float     # confiance de la decision prise, entre 0 et 1
    seuil_utilise: float
    strategie: str

    avertissement: str
    alerte: str

    probabilites_par_classe: Optional[Dict[str, float]] = None
    heatmap_base64: Optional[str] = None
    heatmap_grille: Optional[List[List[float]]] = None


class ReponseSante(BaseModel):
    statut: str
    modele_charge: bool
    peripherique: str
