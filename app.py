from __future__ import annotations

import io
from pathlib import Path

import modal

app = modal.App("text-to-image-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "diffusers~=0.31.0",
        "transformers~=4.44.0",
        "accelerate~=0.33.0",
        "safetensors~=0.4.5",
        "torch~=2.4.0",
        "Pillow~=10.4.0",
        "fastapi[standard]~=0.115.0",
    )
)

MODEL_ID = "stabilityai/sdxl-turbo"

@app.cls(image=image, gpu="T4", container_idle_timeout=300)
class TextToImage:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import AutoPipelineForText2Image

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

    @modal.method()
    def generate(self, prompt: str, num_steps: int = 4, seed: int | None = None) -> bytes:
        """Generate an image from a text prompt and return PNG bytes."""
        import torch

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    # Web API Endpoint

    @modal.fastapi_endpoint(method="POST", docs=True)
    def api_generate(self, body: dict):
        from fastapi.responses import Response

        prompt = body.get("prompt", "a photo of an astronaut riding a horse on mars")
        num_steps = body.get("num_steps", 4)
        seed = body.get("seed")

        image_bytes = self.generate.local(prompt, num_steps=num_steps, seed=seed)

        return Response(content=image_bytes, media_type="image/png")


# CLI Entrypoint

@app.local_entrypoint()
def main(
    prompt: str = "a photo of an astronaut riding a horse on mars",
    output: str = "output.png",
    num_steps: int = 4,
    seed: int = 42,
):
    print(f"Generating image for: '{prompt}'")
    image_bytes = TextToImage().generate.remote(prompt, num_steps=num_steps, seed=seed)

    output_path = Path(output)
    output_path.write_bytes(image_bytes)
    print(f"Image saved to {output_path}")
