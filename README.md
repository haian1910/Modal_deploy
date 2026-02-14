# Text-to-Image API on Modal

A serverless image generation API deployed on [Modal](https://modal.com).
Uses **Stable Diffusion XL Turbo** — a fast, distilled model that generates images in 1-4 inference steps on a single T4 GPU.


## Prerequisites

- **Python 3.11+**
- A free **Modal account** — sign up at [modal.com](https://modal.com)

## Setup (step by step)

### 1. Install the Modal package

```bash
pip install -r requirements.txt
```

### 2. Authenticate with Modal

```bash
modal setup
```

### 3. Test locally (runs on Modal's cloud GPU)

```bash
modal run app.py --prompt "a cat wearing sunglasses on a beach"
```

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
