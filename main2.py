from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import xgboost as xgb
import io

app = FastAPI(title="API Détection de Billets - M2")

# Configuration CORS pour autoriser Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle avec la méthode native XGBoost
MODEL_PATH = "xgboost1.json"  # Assure-toi d'avoir sauvegardé ton modèle en .json
model = xgb.XGBClassifier()

try:
    model.load_model(MODEL_PATH)
    print("✅ Modèle XGBoost chargé avec succès (format JSON).")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    model = None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas disponible sur le serveur.")

    try:
        # 1. Lecture du fichier CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), sep=";")
        
        # 2. Validation des colonnes
        required = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
        if not all(col in df.columns for col in required):
            raise HTTPException(status_code=400, detail=f"Colonnes manquantes. Requis : {required}")

        X = df[required]
        
        # 3. Prédictions et calcul du score de confiance
        predictions = model.predict(X)
        probs = model.predict_proba(X)[:, 1] 
        
        # 4. Préparation du résultat
        df_results = df.copy()
        df_results['prediction'] = [int(p) for p in predictions]
        df_results['score_confiance'] = [round(prob * 100, 2) for prob in probs]
        
        stats = {
            "total_analyses": len(df_results),
            "vrais_billets": int((df_results['prediction'] == 1).sum()),
            "faux_billets": int((df_results['prediction'] == 0).sum())
        }
        
        return {
            "statistiques": stats,
            "table_predictions": df_results.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de traitement : {str(e)}")

# Pour lancer : uvicorn main:app --reload