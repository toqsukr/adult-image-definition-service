from fastapi import FastAPI, File, UploadFile
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os
from fastapi.responses import JSONResponse
from io import BytesIO
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler() 
    ]
)

app = FastAPI()

model_dir = "./models"
latest_model = "nsfw_nude_classifier_model.h5"

model_path = os.path.join(model_dir, latest_model)
model = load_model(model_path)

def predict_image(image):
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0][0]
    return float(prediction) 

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logging.info(f"Received file: {file.filename}")

        image_data = BytesIO(await file.read())
        
        image = load_img(image_data, target_size=(224, 224))
        
        probability = predict_image(image)
        
        probability_rounded = round(probability, 2)

        logging.info(f"File: {file.filename}, Prediction: {probability_rounded}")
        
        return JSONResponse(content={"probability": probability_rounded})
    except Exception as e:
        logging.error(f"Error processing file: {file.filename}. Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)