# Assistant Radiologue Virtuel - API

API pédagogique d'analyse de radiographies thoraciques. Elle reçoit une radiographie, une vue, un âge et un sexe, puis renvoie une détection binaire d'anomalie, un niveau de confiance, une heatmap des zones déterminantes et des messages d'avertissement adaptés.

## Architecture

```
app/
  config.py         Configuration clinique, prompt, seuils, messages
  preprocessing.py  Pretraitement des images (a aligner sur ton nettoyage)
  model.py          Chargement du modele et calcul du score d'anomalie
  heatmap.py        Heatmap par occultation (occlusion sensitivity)
  schemas.py        Schemas de reponse (Pydantic)
  main.py           Application FastAPI (points d'entree /predict et /health)
requirements.txt
Dockerfile
```

## Fonctionnement

- La détection d'anomalie repose sur la stratégie validée en évaluation : `anomalie = 1 - P(image normale)`, avec un seuil calibré au point de fonctionnement haute sensibilité.
- La heatmap masque successivement chaque région de l'image et mesure la baisse du score d'anomalie. Les régions dont l'occultation fait le plus chuter le score sont surlignées.

## Points à adapter avant la mise en production

- **`config.MODELE_ID`** : renseigne le dépôt ou le chemin de ton modèle fine-tuné.
- **`config.SEUIL_ANOMALIE`** : reporte la valeur exacte issue de ta calibration.
- **`preprocessing.pretraiter_image`** : reproduis fidèlement le nettoyage utilisé pour générer tes images `_cleaned`. C'est la cause première d'écart de performance entre l'entraînement et l'API.

## Lancement en local

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Documentation interactive disponible sur `http://localhost:8000/docs`.

### Exemple d'appel

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "fichier=@radio.jpg" \
  -F "vue=AP" \
  -F "age=54" \
  -F "sexe=M" \
  -F "inclure_heatmap=true"
```

## Déploiement sur Azure

L'inférence Gemma requiert idéalement un GPU. Trois options adaptées :

- **Azure Container Apps** (avec profil GPU) : construire l'image, la pousser sur Azure Container Registry, puis créer une application conteneur.
- **Azure Machine Learning - Managed Online Endpoint** : solution native pour servir un modèle sur GPU, avec mise à l'échelle gérée.
- **Azure Container Instances (ACI)** avec GPU : plus simple, adapté à une démonstration.

Exemple avec Azure Container Registry et Container Apps :

```bash
# 1. Construire et pousser l'image
az acr build --registry MonRegistre --image xray-api:latest .

# 2. Deployer sur Azure Container Apps
az containerapp create \
  --name xray-api \
  --resource-group MonGroupe \
  --image MonRegistre.azurecr.io/xray-api:latest \
  --target-port 8000 \
  --ingress external \
  --cpu 4 --memory 16Gi
```

Pour un déploiement CPU (démonstration lente), l'image Docker fournie fonctionne telle quelle. Pour un déploiement GPU, remplace la ligne `FROM` du Dockerfile par une image de base CUDA et installe `torch` avec le support GPU correspondant.

## Avertissement

Cet outil est strictement pédagogique et ne constitue en aucun cas un dispositif médical ni un diagnostic. Toute suspicion d'anomalie doit être confirmée par un professionnel de santé.
