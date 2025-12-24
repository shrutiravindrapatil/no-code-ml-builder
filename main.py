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
    scaler_type: str  # "standard", "minmax", or "none"
    encoding_type: Optional[str] = "none" # "label" or "none"

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
        save_dataframe(f"{file_id}_original", df.copy())
        
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
    # Try to load the original data to ensure we don't apply scaling on top of already scaled data
    # (Enables toggling between methods)
    df = get_dataframe(f"{request.file_id}_original")
    if df is None:
         # Fallback for older existing sessions or safety
         df = get_dataframe(request.file_id)
         
    if df is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    df = df.copy()
    
    try:
        # 1. Handle Encoding (Convert Words to Numbers)
        if request.encoding_type == "label":
            # Identify columns that are object/category type
            # We only convert the ones that are in 'numeric_columns' (selected columns) or ALL?
            # User expectation: "if dataset consist any categorial data"
            # It's safer to convert all categorical data in the dataframe to ensure they are usable
            obj_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in obj_cols:
                # Use factorize which is simple label encoding
                codes, _ = pd.factorize(df[col])
                df[col] = codes

        # 2. Handle Scaling
        scaler = None
        if request.scaler_type == "standard":
            scaler = StandardScaler()
        elif request.scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif request.scaler_type == "none":
            scaler = None
        
        if scaler and request.numeric_columns:
            # We should only scale columns that are actually numeric.
            # If encoding happened, more columns might be numeric now.
            # But we only want to scale the columns the user SELECTED (request.numeric_columns).
            
            # Filter selected columns for those that are actually numeric in the DF
            # (If encoding was skipped, string cols will be skipped here)
            valid_cols = [c for c in request.numeric_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
            
            if valid_cols:
                df[valid_cols] = scaler.fit_transform(df[valid_cols])
        
        # Update store (overwrites the current working copy)
        save_dataframe(request.file_id, df)
        
        # Recalculate which columns are numeric now, to send back to frontend
        current_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        return {
            "message": f"Applied {request.scaler_type} scaling to {len(request.numeric_columns)} columns",
            "preview": df.head(5).to_dict(orient="records"),
            "numeric_columns": current_numeric_cols # Send back updated list
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
    
    # Handling Categorical Data based on Model Type
    encoders = {}
    cat_columns = X.select_dtypes(include=['object', 'category']).columns
    
    if request.model_type == "logistic":
        if len(cat_columns) > 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Logistic Regression cannot handle text data in columns: {list(cat_columns)}. Please go back to Preprocessing and choose 'Convert Words to Numbers'."
            )
            
    elif request.model_type == "decision_tree":
        # Auto-encode categorical columns for Decision Tree
        for col in cat_columns:
            codes, uniques = pd.factorize(X[col])
            X[col] = codes
            encoders[col] = uniques.tolist()
    
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
            "target": request.target_column,
            "encoders": encoders 
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
    encoders = stored.get("encoders", {}) # Get encoders if they exist
    
    try:
        # Prepare input data in the correct order
        input_list = []
        for f in features:
            val = request.inputs.get(f, 0)
            
            # If this feature has an encoder, we expect valid string input
            if f in encoders:
                mapper = encoders[f] # This is a list of unique values
                if val in mapper:
                    # Find index
                    val = mapper.index(val)
                else:
                    # User entered a category not seen during training?
                    # Fallback to -1 or 0? 
                    # If model trained with 0..N, -1 might go down a specific path.
                    val = -1
            else:
                # Assume numeric
                val = float(val)
            
            input_list.append(val)

        prediction = model.predict([input_list])[0]
        
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
