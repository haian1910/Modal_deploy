# Text-to-Image API on Modal

A serverless image generation API deployed on [Modal](https://modal.com).
Uses **Stable Diffusion XL Turbo** — a fast, distilled model that generates images in 1-4 inference steps on a single T4 GPU.

## Project Structure

```
Modal_deploy/
├── app.py              # Modal app — model loading, inference, web API
├── requirements.txt    # Local dependencies (just the modal package)
└── README.md
```

## Prerequisites

- **Python 3.11+**
- A free **Modal account** — sign up at [modal.com](https://modal.com)

## Setup (step by step)

### 1. Install the Modal package

```bash
pip install -r requirements.txt
```

Or directly:

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal setup
```

This opens a browser window to log in and creates a local API token.
If `modal` isn't on your PATH, use `python -m modal setup`.

### 3. Test locally (runs on Modal's cloud GPU)

```bash
modal run app.py --prompt "a cat wearing sunglasses on a beach"
```

This sends the prompt to Modal, generates the image on a cloud GPU, and saves `output.png` locally.

You can also customise the output path and inference steps:

```bash
modal run app.py --prompt "a futuristic city at sunset" --output city.png --num-steps 4
```

### 4. Deploy as a persistent web API

```bash
modal deploy app.py
```

After deployment, Modal prints the live URL for your endpoint. It looks like:

```
https://<your-workspace>--text-to-image-api-texttoimage-api-generate.modal.run
```

## Using the API

### With curl

```bash
curl -X POST "https://<your-url>/api_generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a photo of an astronaut riding a horse on mars"}' \
  --output image.png
```

### With Python

```python
import requests

url = "https://<your-url>/api_generate"
response = requests.post(url, json={
    "prompt": "a photo of an astronaut riding a horse on mars",
    "num_steps": 4,
    "seed": 42,
})

with open("image.png", "wb") as f:
    f.write(response.content)
```

### Interactive API docs

After deploying, visit your endpoint URL in a browser.
Modal serves auto-generated Swagger/OpenAPI docs (enabled via `docs=True`).

## API Reference

**POST** `/api_generate`

| Field      | Type   | Default                                           | Description                     |
|------------|--------|---------------------------------------------------|---------------------------------|
| `prompt`   | string | `"a photo of an astronaut riding a horse on mars"` | Text description of the image   |
| `num_steps`| int    | `4`                                               | Inference steps (1-4 for Turbo) |
| `seed`     | int    | `null`                                            | Optional seed for reproducibility |

**Response:** `image/png` binary

## How It Works

1. **Container image** — Modal builds a container with `diffusers`, `torch`, and `transformers` pre-installed.
2. **Model loading** — The `@modal.enter()` method loads SDXL-Turbo into GPU memory once when the container cold-starts.
3. **Inference** — Each request runs the diffusion pipeline for the specified number of steps and returns PNG bytes.
4. **Auto-scaling** — Modal automatically scales containers up/down based on traffic (including scaling to zero when idle).

## Cost

- Modal's free tier includes $30/month in compute credits.
- SDXL-Turbo on a T4 GPU generates an image in ~1 second, costing fractions of a cent per image.
- Containers auto-stop after 5 minutes of inactivity (`container_idle_timeout=300`).
