import os
import io
import uuid
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage with disk persistence
DATA_STORE = {}
MODEL_STORE = {} # Store trained models and feature names
STORAGE_DIR = "process_storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

def get_dataframe(file_id: str) -> pd.DataFrame:
    """Retrieve DataFrame from memory or disk."""
    if file_id in DATA_STORE:
        return DATA_STORE[file_id]
    
    file_path = os.path.join(STORAGE_DIR, f"{file_id}.pkl")
    if os.path.exists(file_path):
        try:
            df = pd.read_pickle(file_path)
            DATA_STORE[file_id] = df
            return df
        except Exception as e:
            print(f"Failed to load {file_id} from disk: {e}")
            return None
    return None

def save_dataframe(file_id: str, df: pd.DataFrame):
    """Save DataFrame to memory and disk."""
    DATA_STORE[file_id] = df
    file_path = os.path.join(STORAGE_DIR, f"{file_id}.pkl")
    df.to_pickle(file_path)

class ProcessRequest(BaseModel):
    file_id: str
    numeric_columns: List[str]
    scaler_type: str  # "standard" or "minmax"

class SplitRequest(BaseModel):
    file_id: str
    target_column: str
    test_size: float
    selected_columns: Optional[List[str]] = None

class TrainRequest(BaseModel):
    file_id: str
    model_type: str # "logistic" or "decision_tree"
    target_column: str
    test_size: float = 0.2
    selected_columns: Optional[List[str]] = None

class PredictRequest(BaseModel):
    file_id: str
    inputs: dict

@app.get("/")
def read_root():
    return {"message": "No-Code ML API is running"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Cleanup column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        file_id = str(uuid.uuid4())
        save_dataframe(file_id, df)
        
        preview = df.head(5).to_dict(orient="records")
        columns = df.columns.tolist()
        
        # Identify numeric columns for convenience
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": columns,
            "numeric_columns": numeric_cols,
            "preview": preview
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/api/preprocess")
def preprocess_data(request: ProcessRequest):
    df = get_dataframe(request.file_id)
    if df is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    df = df.copy()
    
    try:
        scaler = None
        if request.scaler_type == "standard":
            scaler = StandardScaler()
        elif request.scaler_type == "minmax":
            scaler = MinMaxScaler()
        
        if scaler and request.numeric_columns:
            df[request.numeric_columns] = scaler.fit_transform(df[request.numeric_columns])
        
        # Update store (or we could create a new version to allow undo)
        save_dataframe(request.file_id, df)
        
        return {
            "message": f"Applied {request.scaler_type} scaling to {len(request.numeric_columns)} columns",
            "preview": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

@app.post("/api/split")
def split_data(request: SplitRequest):
    df = get_dataframe(request.file_id)
    if df is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    if request.target_column not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found")
    
    # Define X (features)
    if request.selected_columns:
        # If user explicitly selected columns in Preprocessing, use them for X
        # But ensure target is removed if it's in there
        # We also need to ensure these columns actually exist in df
        valid_cols = [c for c in request.selected_columns if c in df.columns]
        if request.target_column in valid_cols:
             valid_cols.remove(request.target_column)
        
        if not valid_cols:
             # Fallback if selection is weird
             val_cols = df.drop(columns=[request.target_column]).select_dtypes(include=[np.number]).columns.tolist()

        X = df[valid_cols]
    else:
        # Default behavior: Use all numeric columns except target
        X = df.drop(columns=[request.target_column]).select_dtypes(include=[np.number])
    
    y = df[request.target_column]
    
    # Ensure X is numeric (in case selected_columns included strings)
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0) # handle NaNs
    
    # Ensure we have data
    if X.empty or len(X) == 0:
        raise HTTPException(status_code=400, detail="No valid feature columns found for splitting. Please select numeric columns.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=request.test_size, random_state=42)
        
        return {
            "message": "Split successful",
            "train_shape": X_train.shape,
            "test_shape": X_test.shape,
            "distribution": {
                "train": len(X_train),
                "test": len(X_test)
            }
        }
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Split error: {str(e)}")

@app.post("/api/train")
def train_model(request: TrainRequest):
    df = get_dataframe(request.file_id)
    if df is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Selection of features
    if request.selected_columns:
        # Use only columns the user explicitly chose
        X = df[request.selected_columns]
    else:
        # Fallback to all numeric columns excluding target
        X = df.drop(columns=[request.target_column]).select_dtypes(include=[np.number])
    
    y = df[request.target_column]
    
    # Fill NaNs for robustness
    X = X.fillna(0)
    
    # Check if we have enough data
    if len(X) < 10:
         raise HTTPException(status_code=400, detail="Not enough data to train")
    
    # Use the requested test_size
    test_size = request.test_size
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = None
    if request.model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif request.model_type == "decision_tree":
        model = DecisionTreeClassifier()
    else:
        raise HTTPException(status_code=400, detail="Unknown model type")
    
    try:
        model.fit(X_train, y_train)
        
        # Save model and features for prediction
        MODEL_STORE[request.file_id] = {
            "model": model,
            "features": X.columns.tolist(),
            "target": request.target_column
        }
        
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        # Convert classification report to safe dict
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Calculate feature metadata for user input validation
        feature_metadata = {}
        for col in X.columns:
            col_data = X[col]
            meta = {
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "is_binary": False
            }
            # Check for binary/categorical-like (few unique values)
            unique_vals = col_data.unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                 meta["is_binary"] = True
            
            feature_metadata[col] = meta

        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "model_type": request.model_type,
            "features": X.columns.tolist(), # Return features for frontend form
            "feature_metadata": feature_metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/api/predict")
def predict_result(request: PredictRequest):
    if request.file_id not in MODEL_STORE:
        raise HTTPException(status_code=404, detail="Model not found for this file. Please train first.")
    
    stored = MODEL_STORE[request.file_id]
    model = stored["model"]
    features = stored["features"]
    
    try:
        # Prepare input data in the correct order
        input_data = [float(request.inputs.get(f, 0)) for f in features]
        prediction = model.predict([input_data])[0]
        
        # If prediction is numpy type, convert to native python type
        if hasattr(prediction, "item"):
            prediction = prediction.item()
            
        return {
            "prediction": prediction,
            "feature_values": request.inputs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/download/{file_id}")
def download_data(file_id: str):
    df = get_dataframe(file_id)
    if df is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Convert DF to CSV string
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=processed_data_{file_id}.csv"
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
