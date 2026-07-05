"""
Pretraitement des radiographies.

ATTENTION : la fonction ci-dessous doit reproduire EXACTEMENT le nettoyage
applique pour generer les images "_cleaned" utilisees a l'entrainement. Une
divergence de pretraitement degrade fortement les predictions. Adapte le corps
de `pretraiter_image` a ta chaine de nettoyage reelle.
"""

from PIL import Image

from .. import config


def pretraiter_image(image):
    """Applique le pretraitement standard a une image PIL et retourne une image RGB.

    Le pretraitement se limite au redimensionnement, conformement a la chaine
    utilisee a l'entrainement. Aucune conversion en niveaux de gris ni
    egalisation d'histogramme n'est appliquee.
    """
    image = image.convert("RGB")               # garantit trois canaux
    image = image.resize((config.TAILLE_IMAGE, config.TAILLE_IMAGE))
    return image


def charger_image_depuis_octets(octets):
    """Construit une image PIL a partir de donnees binaires."""
    from io import BytesIO
    return Image.open(BytesIO(octets))
