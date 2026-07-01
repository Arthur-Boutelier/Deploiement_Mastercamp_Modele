from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os
from huggingface_hub import login


login("hf_TonTokenSecretWriteIci") 

app = FastAPI(title="MedGemma API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En production, remplace "*" par l'URL de ton site web Azure
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Vérification/Téléchargement du modèle...")
model_path = hf_hub_download(
    repo_id="TonPseudo/MedGemma-4", 
    filename="unsloth.Q4_K_M.gguf", 
    token=os.environ.get("HF_TOKEN") # Le token sera défini dans les variables d'environnement Azure
)

# Chargement du modèle en mémoire RAM
print("Chargement en mémoire...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048, # Taille du contexte (les tokens max)
    n_threads=4 # Optimisation pour les CPU d'Azure
    # Note: Pour un vrai modèle visuel (VLM), il faudrait utiliser une classe supportant la vision (ex: Llama-cpp avec vision)
)

@app.post("/predict")
async def predict_radiography(
    image: UploadFile = File(...), 
):
    # 1. Réception et lecture de l'image (Étape 2 de ta pipeline : Prétraitement)
    image_bytes = await image.read()
    
    # ---> C'est ici que tu appelleras tes fonctions de redimensionnement/normalisation <---
    
    # 2. Préparation du prompt multimodal (Étape 3 : Inférence)
    prompt = f"Instruction: Analyse cette radiographie et donne le JSON.\nInput: {texte_radiologue}\nOutput:"
    
    # (Dans un vrai cas VLM, l'image traitée est passée à l'inférence ici)
    output = llm(
        prompt,
        max_tokens=200,
        stop=["\n", "}"] # Arrête la génération à la fin du JSON
    )
    
    # 3. Renvoi de la réponse structurée (Étape 5 et 6)
    return {
        "filename": image.filename,
        "content_type": image.content_type,
        "response": output["choices"][0]["text"]
    }