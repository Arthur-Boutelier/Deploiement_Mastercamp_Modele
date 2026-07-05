import os

# ---------------------------------------------------------------------------
# Modele
# ---------------------------------------------------------------------------
MODELE_ID = os.environ.get("MODELE_ID", "Arthurbtlr/Modele_finetune_final_V8")

CHARGER_EN_4BIT = os.environ.get("CHARGER_EN_4BIT", "true").lower() == "true"


COLONNES_MALADIES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]

IDX_NO_FINDING = COLONNES_MALADIES.index("No Finding")

def construire_prompt_binaire(vue, sexe, age, maladie):
    """Prompt binaire pour une pathologie donnee.

    IMPORTANT : ce texte doit etre strictement identique a celui utilise
    lors de l'entrainement du modele, faute de quoi les scores seront fausses.
    """
    return (
        "Tu es un systeme d'analyse de radiographies thoraciques.\n"
        f"Tu analyses une radiographie {vue} d'un(e) {sexe} de {age} ans.\n\n"
        f"Question : la pathologie suivante est-elle visible sur cette radiographie : {maladie} ?\n"
        "Reponds uniquement par un seul mot, sans ponctuation : oui ou non."
    )



STRATEGIE_ANOMALIE = os.environ.get("STRATEGIE_ANOMALIE", "complement_no_finding")
SEUIL_ANOMALIE = float(os.environ.get("SEUIL_ANOMALIE", "0.04"))
CLASSES_FIABLES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Edema", "Consolidation", "Atelectasis", "Pleural Effusion", "Support Devices",
]

HEATMAP_GRILLE = int(os.environ.get("HEATMAP_GRILLE", "6"))

HEATMAP_LOT = int(os.environ.get("HEATMAP_LOT", "4"))

HEATMAP_ALPHA = float(os.environ.get("HEATMAP_ALPHA", "0.45"))

TAILLE_IMAGE = int(os.environ.get("TAILLE_IMAGE", "320"))


AVERTISSEMENT_GENERAL = (
    "Cet outil est exclusivement pedagogique. Il ne constitue en aucun cas "
    "un diagnostic medical."
)

ALERTE_ANOMALIE = (
    "Attention : une anomalie potentielle a ete detectee. Ceci n'est qu'un outil "
    "pedagogique. Veuillez prendre rendez-vous avec votre radiologue afin de "
    "confirmer cette analyse."
)

MESSAGE_NORMAL = (
    "Aucune anomalie evidente n'a ete detectee. Cet outil pedagogique ne remplace "
    "pas l'avis d'un professionnel de sante."
)
