from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import shutil
import tempfile
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import uvicorn

model = None
genre_map = None

image_size = (224, 224)
model_file = "logo_genre_classifier.pt"
final_dataset_file = "ml_genre_dataset.csv"
device = torch.device("cpu")

class LogoClassifier(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        
        self.base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        num_ftrs = 512 * 7 * 7
        
        self.classifier_head = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.base_model.classifier = self.classifier_head
        
    def forward(self, x):
        return self.base_model(x)
    
data_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_genre_core(image_path: str):
    
    global model, genre_map
    
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = data_transforms(image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_index = torch.max(probabilities, 0)
            
        predicted_genre = genre_map[predicted_index.item()]
        
        return predicted_genre, confidence.item()
    
    except Exception as e:
        print(f"Inference error during core prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global model, genre_map
    
    print("Startup: Initializing classification module...")
    
    try:
        df = pd.read_csv(final_dataset_file)
        genre_map = sorted(df['primary_genre'].unique())
        num_classes = len(genre_map)
        print(f"Startup: loaded {num_classes} genre classes")
    except FileNotFoundError:
        print(F"Dataset file not found at {final_dataset_file}. Cannot initialize genre map")
        genre_map = None
    except Exception as e:
        print(f"Error loading genre map: {e}")
        genre_map = None
        
    if genre_map:
        
        try:
            model = LogoClassifier(len(genre_map))
            model.to(device)
            state_dict = torch.load(model_file, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Startup: successfully loaded trained model from {model_file}")
        except Exception as e:
            print(f"Could not load model or state dict: {e}")
            model = None
            
    yield
    
    print("Shutdown: cleaning up model resources")
    

app = FastAPI(
    title="Logo Classification API",
    description="Metal genre classification using VGG16 transfer learning",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict")
async def predict_genre_endpoint(image: UploadFile = File(...)):
    
    if model is None or genre_map is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not yet initialized or failed to load during startup"
        )
        
    temp_dir = tempfile.gettempdir()
    file_location = os.path.join(temp_dir, image.filename)
    
    try:
        with open(file_location, "wb") as file_object:
            shutil.copyfileobj(image.file, file_object)
            
        predicted_genre, confidence = predict_genre_core(file_location)
        
        return JSONResponse(content={
            "filename": image.filename,
            "predicted_genre": predicted_genre,
            "confidence": confidence
        })
        
    finally:
        
        if os.path.exists(file_location):
            os.remove(file_location)
            
            
if __name__ == "__main__":
    uvicorn.run("api.app", host="0.0.0.0", port=8080, log_level="info")
        