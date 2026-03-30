# DiP — Fast Motion Generation Mode

DiP (Diffusion Planner) is the fast inference mode integrated into this project (AI Motion Generator).
It significantly speeds up text-to-motion generation compared to the standard MDM pipeline.

## How Fast?

- Standard MDM: 1000 diffusion steps → several seconds per sample
- DiP: 10 diffusion steps → ~0.4 seconds per sample
- That's roughly **40× faster** with comparable output quality

DiP performs well even at 5 steps, making it suitable for near real-time applications.

## How It Works

DiP is **autoregressive** — instead of generating the full motion sequence at once,
it generates **2 seconds of motion at a time**:

- Uses the previous 20 frames as context (prefix)
- Predicts the next 40 frames
- Slides forward and repeats

This is similar to how language models generate text token by token,
but applied to human joint motion over time.

## Architecture

- **Backbone**: Transformer Decoder (`--arch trans_dec`)
- **Text Encoder**: DistilBERT (lighter and faster than CLIP, used via `--text_encoder_type bert`)
- **Diffusion Steps**: 10 (`--diffusion_steps 10`)
- **Guidance**: Classifier-free guidance with `--guidance_param 7.5`

At each diffusion step, the model receives:
- A clean prefix (past motion context)
- The current prediction, noised to timestep `t`
- The text embedding from DistilBERT

## Generate
```bash
python -m sample.generate \
   --model_path save/target_10steps_context20_predict40/model000200000.pt \
   --autoregressive --guidance_param 7.5
```

To use your own prompt:
```bash
--text_prompt "A person throws a ball."
```

To change prompts dynamically (each line = 2 seconds of motion):
```bash
--dynamic_text_path assets/example_dynamic_text_prompts.txt
```

> Note: The initial prefix is sampled from the dataset. If the starting pose
> doesn't match your prompt (e.g., sitting when you ask for throwing),
> the model will transition naturally before executing the action.

## Train DiP
```bash
python -m train.train_mdm \
--save_dir save/my_humanml_DiP \
--dataset humanml --arch trans_dec --text_encoder_type bert \
--diffusion_steps 10 --context_len 20 --pred_len 40 \
--mask_frames --use_ema --autoregressive --gen_guidance_param 7.5
```

Recommended additions:
- `--eval_during_training` and `--gen_during_training` for checkpoint monitoring
- `--use_ema` for Exponential Moving Average (improves stability)
- `--mask_frames` to fix a known frame masking bug

## Evaluate
```bash
python -m eval.eval_humanml \
  --model_path save/DiP_no-target_10steps_context20_predict40/model000600343.pt \
  --autoregressive --guidance_param 7.5
```

Add `--train_platform_type WandBPlatform` to log results to Weights & Biases.

## Model Checkpoints

Download and place under `save/`:

- [DiP (text-to-motion)](https://huggingface.co/guytevet/CLoSD/tree/main/checkpoints/dip/DiP_no-target_10steps_context20_predict40)
- [DiP with target conditioning](https://drive.google.com/file/d/1PsilP2xhcOHHXkmtxtOwNbWeI0njU2ic/view?usp=sharing) — supports both conditioned and unconditioned generation

## Credits

DiP was originally introduced in the CLoSD paper (ICLR 2025) by Guy Tevet et al.
This project integrates DiP as its fast generation backend.
