AI Motion Generator
A web-based generative AI application that converts natural language text prompts into 3D human motion animations. Built on top of the MDM (Human Motion Diffusion Model) research paper, extended with a custom Flask backend and browser-based frontend.

What it does
Type a sentence like "a person walks forward and picks up a box" and the system generates a realistic 3D skeletal animation of that motion. The output can be rendered as a stick figure video or a full SMPL body mesh (importable into Blender/Maya).

How it works — core logic
1. Text understanding
The text prompt is passed through OpenAI's CLIP encoder, which converts it into a fixed-size embedding vector. This vector is what "conditions" the generation — it tells the model what kind of motion to produce.
2. Diffusion-based generation
The generation process is based on Gaussian diffusion:

During training: real motion capture sequences have noise gradually added until they become pure Gaussian noise. The model learns to reverse this process.
During inference: the model starts from random Gaussian noise and iteratively denoises it over 50 or 1000 steps, guided by the CLIP text embedding, until a clean motion sequence emerges.

3. Motion representation
Motions are represented as sequences of 3D joint positions — 22 body joints × 3 coordinates (x, y, z) × T time frames (up to ~196 frames at 20 fps, roughly 9.8 seconds).
4. The model architecture
The denoising model is a Transformer (not a CNN). At each diffusion step:

Input: noisy motion sequence + timestep embedding + CLIP text embedding
Output: predicted clean motion x0
The Transformer attends across both time (motion frames) and the text conditioning

5. Rendering
Raw output is a .npy file of joint positions. This is visualized as a stick figure .mp4. Optionally, render_mesh.py runs SMPLify to fit a full 3D body mesh over the joints and exports per-frame .obj files.

Project structure
AI_Motion_Generator/
│
├── backend/              # Flask REST API server
│   ├── app.py            # Main server: loads model, exposes /generate endpoint
│   └── ...               # Model loading utilities, device management
│
├── frontend/             # Browser-based UI
│   ├── index.html        # Input form and video preview area
│   ├── script.js         # Sends prompt to backend, displays result
│   └── style.css         # UI styling
│
├── model/
│   └── mdm.py            # Core Transformer model definition (CLIP + diffusion)
│
├── diffusion/            # Gaussian diffusion math (noise schedule, forward/reverse)
│
├── train/
│   └── train_mdm.py      # Training loop (supports EMA, WandB, TensorBoard)
│
├── sample/
│   ├── generate.py       # Run inference: text/action prompt → motion + video
│   └── edit.py           # Motion editing (in-between, upper-body editing)
│
├── visualize/
│   └── render_mesh.py    # Convert stick figure to SMPL 3D mesh (.obj per frame)
│
├── data_loaders/         # HumanML3D / KIT dataset loading pipelines
│
├── eval/                 # Evaluation scripts (FID, R-precision metrics)
│
├── utils/                # Losses (velocity, foot contact, geometry), rotations, misc
│
├── prepare/              # Shell scripts to download weights and datasets
│
├── chumpy/               # SMPL body model dependency (included locally)
│
├── assets/               # Example outputs (GIFs, .tex results table)
│
├── DiP.md                # Instructions for the ultra-fast 50-step DiP variant
├── environment.yml       # Conda environment (Python 3.7, PyTorch, CLIP, spaCy)
└── cog.yaml              # Replicate.com deployment config

Backend
The backend is a Flask Python server (backend/app.py).
On startup: loads the pretrained MDM model weights into GPU/CPU memory so they are ready for fast inference.
Endpoint: POST /generate

Input (JSON): { "text_prompt": "a person waves their hand", "motion_length": 6.0 }
Process: runs the full diffusion sampling pipeline (equivalent to sample/generate.py) — starts from Gaussian noise, denoises for 50 steps conditioned on the CLIP-encoded prompt, outputs joint positions
Output: returns the generated animation video file (.mp4) or raw joint data


Frontend
The frontend is a plain HTML/CSS/JavaScript web interface (frontend/).

index.html — a text input box, motion length slider, and a "Generate" button; a video element below shows the result
script.js — on button click, sends a POST request to the backend /generate endpoint, waits for the response (a few seconds while diffusion runs), then displays the returned video in the browser
style.css — UI layout and styling


Requirements

Python 3.7
CUDA-capable GPU (required for inference at reasonable speed)
conda or miniconda


Setup
bash# Create environment
conda env create -f environment.yml
conda activate mdm
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git

# Download pretrained weights and data (text-to-motion)
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
Download the pretrained model from the links in the original MDM repo and place it in ./save/.

Running the web app
bash# Start the backend server
python backend/app.py

# Open frontend/index.html in your browser

Running inference directly (command line)
bash# Generate from a single text prompt
python -m sample.generate \
  --model_path ./save/humanml_trans_enc_512/model000200000.pt \
  --text_prompt "the person walked forward and is picking up his toolbox."

# Generate from a text file of prompts
python -m sample.generate \
  --model_path ./save/humanml_trans_enc_512/model000200000.pt \
  --input_text ./assets/example_text_prompts.txt

# Render SMPL mesh from stick figure output
python -m visualize.render_mesh \
  --input_path /path/to/sample00_rep00.mp4

Motion editing
bash# In-between editing (fill motion between two keyframes)
python -m sample.edit \
  --model_path ./save/humanml_trans_enc_512/model000200000.pt \
  --edit_mode in_between

# Upper body editing (fix legs, regenerate arms/torso)
python -m sample.edit \
  --model_path ./save/humanml_trans_enc_512/model000200000.pt \
  --edit_mode upper_body \
  --text_condition "A person throws a ball"

Training your own model
bashpython -m train.train_mdm \
  --save_dir save/my_model \
  --dataset humanml \
  --diffusion_steps 50 \
  --mask_frames \
  --use_ema

Key technologies
ComponentTechnologyMotion generationMDM — Transformer + Gaussian DiffusionText encodingOpenAI CLIPBody modelSMPL (via chumpy)Backend APIFlask (Python)FrontendHTML / CSS / JavaScriptDatasetHumanML3D (motion capture + text labels)Training trackingWandB or TensorBoard

