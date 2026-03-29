import sys
import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil

# Ensure we can import from the root
sys.path.append(os.path.abspath("."))

# Ensure chumpy is available (it's a mock in the root)
import chumpy

from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util

app = FastAPI(title="MDM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.inference import MDMInference

app = FastAPI(title="MDM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model pointer
inference = None
MODEL_PATH = "save/my_humanml_trans_enc_512_bert-50steps/models_to_upload/humanml_trans_dec_512_bert/model000600000.pt"

@app.on_event("startup")
async def startup_event():
    global inference
    print("Initializing MDM Inference...")
    inference = MDMInference(MODEL_PATH, device='cpu')
    print("Inference engine ready.")

class GenerateRequest(BaseModel):
    prompt: str
    num_frames: int = 120 # Increased from 60

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": inference is not None}

@app.post("/generate")
async def generate_motion(req: GenerateRequest):
    if inference is None:
        raise HTTPException(status_code=503, detail="Inference engine not loaded")
    
    try:
        print(f"Generating motion for: {req.prompt}")
        # joints shape: [1, joints, 3, frames]
        joints = inference.generate(req.prompt, num_frames=req.num_frames)
        
        # Convert to list for JSON response
        joints_list = joints[0].tolist()
        
        return {
            "status": "success",
            "prompt": req.prompt,
            "joints": joints_list,
            "fps": inference.fps,
            "num_joints": len(joints_list)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
