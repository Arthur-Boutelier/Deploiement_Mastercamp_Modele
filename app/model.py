"""
Encapsulation du modele : chargement, calcul des probabilites par classe,
et calcul du score d'anomalie.
"""

import torch

from .. import config


class ModeleRadio:
    """Charge le modele une seule fois et expose les methodes d'inference."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.id_oui = None
        self.id_non = None

    # -- Chargement --------------------------------------------------------
    def charger(self):
        """Charge le modele et le tokeniseur. A appeler au demarrage de l'API."""
        from unsloth import FastVisionModel

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=config.MODELE_ID,
            load_in_4bit=config.CHARGER_EN_4BIT,
        )
        FastVisionModel.for_inference(self.model)

        # Padding a gauche : indispensable pour lire le dernier token
        try:
            self.tokenizer.tokenizer.padding_side = "left"
        except AttributeError:
            self.tokenizer.padding_side = "left"

        self.id_oui = self._premier_id("oui")
        self.id_non = self._premier_id("non")
        if self.id_oui == self.id_non:
            raise RuntimeError("Les tokens 'oui' et 'non' partagent le meme identifiant.")

    def _premier_id(self, mot):
        ids = self.tokenizer.tokenizer.encode(mot, add_special_tokens=False)
        return ids[0]

    # -- Inference ---------------------------------------------------------
    def proba_oui(self, image, prompt):
        """Retourne P(oui) normalisee entre oui et non pour une image et un prompt."""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        texte = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.tokenizer(
            image, texte, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            sortie = self.model(**inputs)

        logits = sortie.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p_oui = probs[self.id_oui].item()
        p_non = probs[self.id_non].item()
        return p_oui / (p_oui + p_non + 1e-9)

    def probabilites_par_classe(self, image, vue, sexe, age):
        """Retourne un dictionnaire {pathologie: P(oui)} pour les 14 classes."""
        resultats = {}
        for maladie in config.COLONNES_MALADIES:
            prompt = config.construire_prompt_binaire(vue, sexe, age, maladie)
            resultats[maladie] = self.proba_oui(image, prompt)
        return resultats

    def score_anomalie(self, image, vue, sexe, age):
        """Calcule le score d'anomalie selon la strategie configuree.

        Retourne (score, details) ou details contient les probabilites utiles.
        """
        if config.STRATEGIE_ANOMALIE == "complement_no_finding":
            prompt = config.construire_prompt_binaire(vue, sexe, age, "No Finding")
            p_normal = self.proba_oui(image, prompt)
            return 1.0 - p_normal, {"P_normal": p_normal}

        if config.STRATEGIE_ANOMALIE == "max_pathologies":
            scores = {}
            for maladie in config.CLASSES_FIABLES:
                prompt = config.construire_prompt_binaire(vue, sexe, age, maladie)
                scores[maladie] = self.proba_oui(image, prompt)
            return max(scores.values()), scores

        raise ValueError(f"Strategie inconnue : {config.STRATEGIE_ANOMALIE}")

    def scores_anomalie_lot(self, images, vue, sexe, age):
        """Score d'anomalie pour un lot d'images, en une seule passe avant.

        Utilise pour la heatmap. Ne prend en charge que la strategie
        'complement_no_finding' (une seule classe interrogee : No Finding).
        Retourne un tableau numpy de scores d'anomalie.
        """
        import numpy as np

        prompt = config.construire_prompt_binaire(vue, sexe, age, "No Finding")
        batch_images, textes = [], []
        for image in images:
            batch_images.append([image])
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]
            textes.append(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True))

        inputs = self.tokenizer(
            batch_images, textes, add_special_tokens=False,
            return_tensors="pt", padding=True,
        ).to(self.device)

        with torch.no_grad():
            sortie = self.model(**inputs)

        logits = sortie.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p_oui = probs[:, self.id_oui]
        p_non = probs[:, self.id_non]
        p_normal = (p_oui / (p_oui + p_non + 1e-9)).cpu().numpy()

        del inputs, sortie, logits, probs
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return 1.0 - p_normal


# Instance unique, partagee par toute l'application
modele = ModeleRadio()
