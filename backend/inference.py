import torch
import numpy as np
import os
import sys

# Ensure chumpy is available (it's a mock in the root)
import chumpy

from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.parser_util import generate_args
from sample.generate import load_dataset
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate
from utils import dist_util

class MDMInference:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model_path = model_path
        
        # Setup args
        sys.argv = [
            "generate.py",
            "--model_path", model_path,
            "--device", "-1" if device == 'cpu' else "0",
            "--num_samples", "1",
            "--num_repetitions", "1",
            "--text_prompt", "placeholder"
        ]
        self.args = generate_args()
        
        # Load dataset for metadata
        print("Loading metadata...")
        self.fps = 12.5 if self.args.dataset == 'kit' else 20
        self.n_frames = 60 # Default
        self.data = load_dataset(self.args, 196, self.n_frames)
        
        # Create model
        self.model, self.diffusion = create_model_and_diffusion(self.args, self.data)
        
        # Load weights
        print(f"Loading weights from {model_path}...")
        load_saved_model(self.model, model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def generate(self, prompt, num_frames=60):
        with torch.no_grad():
            # Prepare model_kwargs
            collate_args = [{'inp': torch.zeros(num_frames), 'tokens': None, 'lengths': num_frames, 'text': prompt}]
            _, model_kwargs = collate(collate_args)
            model_kwargs['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}
            
            # Guidance scale
            if self.args.guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(1, device=self.device) * self.args.guidance_param
            
            # Encode text
            model_kwargs['y']['text_embed'] = self.model.encode_text(model_kwargs['y']['text'])
            
            # For BERT
            if self.args.text_encoder_type == 'bert':
                 model_kwargs['y']['text_embed'] = (model_kwargs['y']['text_embed'][0].unsqueeze(0), 
                                                    model_kwargs['y']['text_embed'][1].unsqueeze(0))
            
            motion_shape = (1, self.model.njoints, self.model.nfeats, num_frames)
            
            print(f"Sampling for prompt: {prompt}...")
            sample = self.diffusion.p_sample_loop(
                self.model,
                motion_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                progress=True,
                device=self.device
            )
            
            # Transform back
            if self.model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = self.data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
                
            # sample shape: [1, joints, 3, frames]
            return sample.cpu().numpy()
