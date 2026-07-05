"""
Generation de la heatmap par occultation (occlusion sensitivity), traitee PAR LOT.

Principe : on masque successivement chaque region de l'image et on mesure la
baisse du score d'anomalie. Une baisse importante signifie que la region etait
determinante pour la decision. Les images occultees sont evaluees par lots afin
de rester dans un budget de temps compatible avec un usage en temps reel.

La heatmap n'est PAS sur le chemin critique de la decision : elle est calculee
uniquement sur demande.
"""

import base64
from io import BytesIO

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
matplotlib.colormaps  # force l'initialisation du registre de palettes

from .. import config


def generer_heatmap(modele, image, vue, sexe, age, score_baseline):
    """Retourne (image_png_base64, carte_normalisee) de la heatmap superposee.

    - modele : instance de ModeleRadio deja chargee.
    - image : image PIL pretraitee (RGB).
    - score_baseline : score d'anomalie de l'image non occultee.
    """
    largeur, hauteur = image.size
    n = config.HEATMAP_GRILLE
    pas_x = largeur // n
    pas_y = hauteur // n

    valeur_neutre = int(np.array(image).mean())

    # 1) Construction de toutes les images occultees
    images_occultees = []
    positions = []
    for i in range(n):
        for j in range(n):
            zone = np.array(image).copy()
            y0, y1 = i * pas_y, ((i + 1) * pas_y if i < n - 1 else hauteur)
            x0, x1 = j * pas_x, ((j + 1) * pas_x if j < n - 1 else largeur)
            zone[y0:y1, x0:x1] = valeur_neutre
            images_occultees.append(Image.fromarray(zone))
            positions.append((i, j))

    # 2) Evaluation par lots
    scores = []
    for debut in range(0, len(images_occultees), config.HEATMAP_LOT):
        lot = images_occultees[debut : debut + config.HEATMAP_LOT]
        scores.extend(list(modele.scores_anomalie_lot(lot, vue, sexe, age)))

    # 3) Importance = baisse du score par rapport a la reference
    importance = np.zeros((n, n), dtype=np.float32)
    for (i, j), score_occulte in zip(positions, scores):
        importance[i, j] = max(0.0, score_baseline - float(score_occulte))

    maxi = importance.max()
    if maxi > 0:
        importance = importance / maxi

    overlay = _superposer(image, importance)
    return _png_base64(overlay), importance.tolist()


def _superposer(image, carte):
    """Superpose la carte d'importance sur l'image en niveaux de gris."""
    largeur, hauteur = image.size

    carte_img = Image.fromarray(np.uint8(carte * 255)).resize(
        (largeur, hauteur), resample=Image.BILINEAR
    )
    carte_norm = np.array(carte_img) / 255.0

    couleurs = matplotlib.colormaps["jet"](carte_norm)[:, :, :3]

    fond = np.array(image.convert("RGB")) / 255.0
    melange = (1 - config.HEATMAP_ALPHA) * fond + config.HEATMAP_ALPHA * couleurs
    melange = np.uint8(np.clip(melange, 0, 1) * 255)
    return Image.fromarray(melange)


def _png_base64(image):
    """Encode une image PIL en chaine base64 (PNG)."""
    tampon = BytesIO()
    image.save(tampon, format="PNG")
    return base64.b64encode(tampon.getvalue()).decode("utf-8")
