
# Judge, Jury, and SFX-Maker: A Two-Step Pipeline for Generating and Evaluating Fighting Game Audio

This repository generates and evaluates a **single-round** sound-design set for a **2D fighting game** character: **Zen (male)**.

## Target output per set
- **24 SFX** (≈ **1s** each)
- **1 BGM loop** (≈ **30s**)

## Supported models

### Generators
- **TangoFlux** (local GPU/CPU; Python API)
- **AudioX** (local GPU/CPU; stable-audio-tools inference)
- **ElevenLabs** (cloud API; requires `ELEVENLABS_API_KEY`)

### Optional generation-time quality gate (two-step)
- **Meta Audiobox Aesthetics** ( PQ)

### Evaluators (audio judges)
- **OpenAI** (e.g., `gpt-audio-mini`; requires `OPENAI_API_KEY`)
- **Kimi-Audio**
- **Qwen2-Audio**
- **Omni-R1**
- **Audio Flamingo 3**

---

## What this repo does

### 1) Generation
Two generation modes are supported:
- **one_step**: prompt → generate audio once per item
- **two_step**: prompt → generate → **quality-check** → (optional feedback) → regenerate until accepted or max attempts

### 2) Evaluation
Two evaluation modes are supported:
- **one_step**: judge audio directly (PQ/CE/CI)
- **two_step**: judge first produces a **content description**, then scores (PQ/CE/CI)

---

## Quickstart

### 1 Install
```bash
python -m venv .venv
source .venv/bin/activate   
pip install -U pip
pip install -r requirements.txt
````
### 2 Configure API keys (if using API-based models)

Create `.env`:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# ElevenLabs generation (required if you use ElevenLabs)
ELEVENLABS_API_KEY=...

# OpenAI evaluation (required if you use OpenAI / gpt-audio-mini)
OPENAI_API_KEY=...
```

> For any API-backed model in this repo, you must have a **working API key**.

### 3 Run one-step generation (example: TangoFlux)

```bash
# If scripts are at repo root:
python generate_round.py --model tangoflux --method one_step --out runs/tangoflux__one_step

# If using a package entrypoint:
python -m soundgen.generate_round --model tangoflux --method one_step --out runs/tangoflux__one_step
```

### 4 Example: two-step generation with Audiobox gate

```bash
python generate_round.py \
  --model audiox \
  --method two_step \
  --evaluators audiobox \
  --pq-threshold 6.0 \
  --max-attempts 5 \
  --out runs/audiox__two_step__audiobox
```

### 5 Evaluate the generated audios (example: Qwen2-Audio)

```bash
python judge_folder.py \
  --audio-dir runs/tangoflux__one_step/audio \
  --model qwen2-audio \
  --mode one_step \
  --prompt-style guided \
  --out-json results/qwen2_audio__one_step__guided.json
```

---

## System dependencies

These are commonly needed for audio pipelines:

* **ffmpeg** (recommended; used for format conversions such as MP3 → WAV where applicable)
* **libsndfile** (often required by audio Python stacks)

Install them via your OS package manager (e.g., `apt`, `brew`, `conda`, etc.).

---

## Model setup (follow official instructions)

This README intentionally does **not** duplicate third-party installation steps.
For each model, **follow the official repo/model-card instructions** for download and usage.

Below are “where to look” references (copy/paste into a browser):

### TangoFlux (local generator)

```text
https://github.com/declare-lab/TangoFlux
```

Typical install:

```bash
pip install git+https://github.com/declare-lab/TangoFlux
```

### AudioX (local generator)

```text
https://github.com/ZeyueT/AudioX
```

AudioX typically requires:

* following the repo’s environment instructions
* downloading the checkpoint(s) and config(s) as described in the official README

AudioX inference may depend on stable-audio-tools:

```text
https://github.com/Stability-AI/stable-audio-tools
```

### ElevenLabs (API generator)

```text
https://elevenlabs.io/docs
```

Set:

* `ELEVENLABS_API_KEY` in `.env` or your shell environment.

### Meta Audiobox Aesthetics (optional gate)

```text
https://github.com/facebookresearch/audiobox-aesthetics
```

Typical install:

```bash
pip install audiobox_aesthetics
```


### Evaluator models (Hugging Face)

#### Audio Flamingo 3:

```text
https://huggingface.co/nvidia/audio-flamingo-3-hf
```


#### Kimi-Audio:

```text
https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct
```

#### Qwen2-Audio:

```text
https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
```


#### Omni-R1:

```text
https://huggingface.co/Haoz0206/Omni-R1
```

### OpenAI (API evaluator)

```text
https://platform.openai.com/docs
```

Set:

* `OPENAI_API_KEY` in `.env` or your shell environment.

---


## Prompts and SFX catalog

* SFX list + descriptions are stored in `soundgen/catalog.py`
* Generation prompt templates are stored in `soundgen/prompts.py`

---

## Troubleshooting

### 1 “Missing ELEVENLABS_API_KEY”

* Add `ELEVENLABS_API_KEY=...` to your `.env` (or export it in your shell).

### 2 ffmpeg not found / conversion errors

* Install ffmpeg and ensure it is available on `PATH`.

### 3 Hugging Face downloads are slow / failing

Ensure you have:

* sufficient disk space
* correct `HF_HOME` / cache configuration if using custom cache paths
* any required auth tokens set if the model requires access

### 4 OpenAI errors

* Confirm `OPENAI_API_KEY` is set and valid.
* Confirm your account has access to the requested model.

---

## Reproducibility notes 

* You can all the results from the experiments in `results` folder.

---

## License / attribution

* This repo orchestrates third-party models and APIs.
* Each generator/evaluator model has its own license and usage restrictions.
* Always follow the official license/terms for each model and API provider.

```

