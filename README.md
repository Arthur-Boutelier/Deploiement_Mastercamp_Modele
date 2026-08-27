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

## Avertissement

Cet outil est strictement pédagogique et ne constitue en aucun cas un dispositif médical ni un diagnostic. Toute suspicion d'anomalie doit être confirmée par un professionnel de santé.
