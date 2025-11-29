# Local Image/Video Generation Integration Guide

## Executive Summary

This document provides comprehensive research and integration strategies for adding local image and video generation capabilities to the A2A multi-agent orchestration system. It focuses on lightweight, open-source solutions optimized for local deployment with practical integration patterns.

---

## 1. Image Generation Models

### 1.1 Recommended Models

#### Flux.1 Schnell (Primary Recommendation)
- **Developer**: Black Forest Labs
- **License**: Apache 2.0
- **Model Size**: ~24GB (FP16), ~12GB (FP8), ~6GB (GGUF Q8)
- **VRAM Requirements**:
  - Minimum: 6GB with quantization
  - Recommended: 12GB+ for optimal quality
  - Ideal: 16-24GB for full FP16
- **Speed**: 1-4 steps (extremely fast)
- **Quality**: State-of-the-art, competitive with DALL-E 3
- **Strengths**:
  - Fast generation (schnell = "fast" in German)
  - Excellent prompt adherence
  - High-quality photorealistic and artistic outputs
  - Good text rendering in images
  - Quantization-resistant (maintains quality at Q4/Q8)

#### SDXL (Stable Diffusion XL)
- **Developer**: Stability AI
- **License**: CreativeML OpenRAIL++-M
- **Model Size**: ~7GB (base + refiner)
- **VRAM Requirements**: 8GB minimum, 12GB recommended
- **Speed**: 20-30 steps
- **Quality**: High-quality, versatile
- **Strengths**:
  - Mature ecosystem with extensive LoRA support
  - Lower VRAM requirements than Flux
  - Excellent for specialized styles via fine-tuning
  - Large community and resources

#### SDXL-Turbo (Speed Variant)
- **Model Size**: ~7GB
- **VRAM Requirements**: 6-8GB
- **Speed**: 1-4 steps
- **Use Case**: Real-time/near-real-time generation
- **Trade-off**: Slightly lower quality than base SDXL

### 1.2 Python Integration Libraries

#### HuggingFace Diffusers (Primary)
```python
from diffusers import DiffusionPipeline, FluxPipeline
import torch

# Flux Schnell Example
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()  # Reduce VRAM usage

# Generate image
image = pipe(
    prompt="a cyberpunk cityscape at sunset, neon lights reflecting on wet streets",
    num_inference_steps=4,
    guidance_scale=0.0,  # Schnell doesn't need CFG
).images[0]
```

#### Key Features:
- **Unified API**: Consistent interface across models
- **Optimization**: Built-in CPU offload, VAE tiling, xformers support
- **LoRA Support**: Easy fine-tuning and adaptation
- **ControlNet**: Structural guidance for precise control
- **Installation**: `pip install diffusers transformers accelerate`

#### ComfyUI (Alternative/Advanced)
- **Type**: Node-based UI with Python backend
- **Strengths**:
  - Visual workflow design
  - Extensive custom nodes ecosystem
  - Advanced features (ControlNet, LoRA, LCM)
  - API mode for programmatic access
- **Use Case**: Complex workflows, experimentation
- **Integration**: Can be used as a backend service

### 1.3 Hardware Requirements Summary

| Model | Precision | VRAM | Speed | Quality |
|-------|-----------|------|-------|---------|
| Flux Schnell FP16 | Full | 24GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| Flux Schnell FP8 | Quantized | 12GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| Flux Schnell GGUF-Q8 | Quantized | 6GB | ⚡⚡ | ⭐⭐⭐⭐ |
| SDXL Base | Full | 8GB | ⚡⚡ | ⭐⭐⭐⭐ |
| SDXL Turbo | Distilled | 6GB | ⚡⚡⚡ | ⭐⭐⭐ |

### 1.4 LoRA and ControlNet

#### LoRA (Low-Rank Adaptation)
- **Purpose**: Specialize models for specific styles/subjects
- **Size**: 10-200MB (vs. full model fine-tuning)
- **Training**: Can train custom LoRAs on 8-16GB VRAM
- **Use Cases**:
  - Brand-specific art styles
  - Product visualization
  - Character consistency
  - Domain-specific imagery

```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16
)

# Load LoRA weights
pipe.load_lora_weights("path/to/custom-style-lora.safetensors")
pipe.fuse_lora(lora_scale=0.8)  # Adjust influence (0.0-1.0)

image = pipe(prompt="your prompt here").images[0]
```

#### ControlNet
- **Purpose**: Structural guidance (edges, depth, pose, etc.)
- **Types**:
  - Canny edge detection
  - Depth maps
  - OpenPose (human poses)
  - Segmentation masks
- **Use Case**: Precise control over composition

```python
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
import torch
from PIL import Image
import cv2
import numpy as np

# Load ControlNet
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
)

# Prepare canny edge image from input
image = Image.open("input.jpg")
image_np = np.array(image)
canny = cv2.Canny(image_np, 100, 200)
canny_image = Image.fromarray(canny)

# Generate with structural guidance
output = pipe(
    prompt="a watercolor painting of a landscape",
    image=canny_image,
    num_inference_steps=30
).images[0]
```

---

## 2. Video Generation Models

### 2.1 Stable Video Diffusion (SVD)

#### Overview
- **Developer**: Stability AI
- **Type**: Image-to-Video diffusion model
- **Input**: Single conditioning frame (image)
- **Output**: Video sequence

#### Model Variants

##### SVD (Base)
- **Output**: 14 frames
- **Resolution**: Up to 1024x576
- **Generation Time**: ~100s on A100 80GB
- **VRAM**: 16GB+ recommended

##### SVD-XT (Extended)
- **Output**: 25 frames
- **Resolution**: Up to 1024x576
- **Generation Time**: ~180s on A100 80GB
- **VRAM**: 20GB+ recommended

#### Python Integration

```python
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import torch

# Load pipeline
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
)

# Optimize for lower VRAM
pipe.enable_model_cpu_offload()
pipe.unet.enable_forward_chunking()

# Load conditioning image
image = load_image("path/to/image.jpg")
image = image.resize((1024, 576))

# Generate video
frames = pipe(
    image,
    decode_chunk_size=2,  # Lower VRAM usage
    num_frames=25,        # SVD-XT default
    fps=7,                # Frames per second
    motion_bucket_id=127, # Motion amount (0-255)
    noise_aug_strength=0.02  # Conditioning strength
).frames[0]

# Export to video file
export_to_video(frames, "output.mp4", fps=7)
```

#### Micro-Conditioning Parameters

- **fps**: Frames per second in output video (1-30)
- **motion_bucket_id**: Controls motion intensity
  - Lower (50-100): Subtle motion
  - Higher (127-255): More dynamic motion
- **noise_aug_strength**: Conditioning image fidelity
  - Lower (0.0-0.05): More faithful to input
  - Higher (0.1-0.2): More creative variation

#### Optimization Strategies

```python
# For < 8GB VRAM
pipe.enable_model_cpu_offload()
pipe.unet.enable_forward_chunking()
frames = pipe(image, decode_chunk_size=2).frames[0]

# For 8-12GB VRAM
pipe.enable_model_cpu_offload()
frames = pipe(image, decode_chunk_size=8).frames[0]

# For 16GB+ VRAM
pipe.to("cuda")
frames = pipe(image).frames[0]
```

### 2.2 Alternative Approaches

#### Frame Interpolation
- **Tools**: RIFE, FILM
- **Use Case**: Generate intermediate frames between keyframes
- **VRAM**: 4-6GB
- **Speed**: Fast (real-time capable)
- **Quality**: Smooth motion for simple scenes

#### AnimateDiff
- **Type**: Motion module for Stable Diffusion
- **Integration**: ComfyUI nodes
- **Output**: Animated sequences from text prompts
- **VRAM**: Similar to SDXL (8-12GB)

### 2.3 Practical Considerations

#### Local Hardware Viability
- **Minimum Viable**: 12GB VRAM (low resolution, short clips)
- **Recommended**: 16-24GB VRAM (1024x576, 14-25 frames)
- **Professional**: 24GB+ VRAM (full quality, experimentation)

#### Batch Processing Strategy
```python
import asyncio
from pathlib import Path

async def generate_video_batch(images: list[Path], output_dir: Path):
    """Process multiple images to videos in batch"""
    for idx, image_path in enumerate(images):
        image = load_image(str(image_path))
        frames = pipe(image, num_frames=14).frames[0]
        export_to_video(
            frames,
            str(output_dir / f"video_{idx:03d}.mp4"),
            fps=7
        )
        # Clear CUDA cache between generations
        torch.cuda.empty_cache()
```

---

## 3. Integration Patterns for A2A System

### 3.1 Architecture Options

#### Option A: Separate Image Generation Agent

**Pros**:
- Clear separation of concerns
- Dedicated resource management
- Can scale independently
- Follows existing A2A pattern

**Cons**:
- Additional service overhead
- More complex orchestration

**Implementation**:
```python
# agents/specialists/image_generator.py
from pydantic_ai import RunContext
from models.schemas import ImageGenerationOutput
from models.dependencies import ImageGenDependencies
from agents.base import create_agent
from diffusers import FluxPipeline
import torch

# Create image generation agent
image_gen_agent = create_agent(
    model='llama3.2:3b',  # Fast model for coordination
    agent_id="image-generator",
    instructions="""
    You are an image generation specialist using local Flux.1 Schnell model.

    Your responsibilities:
    1. Convert user requests into optimized prompts for Flux
    2. Validate prompt quality and feasibility
    3. Generate images using local GPU resources
    4. Provide metadata about generated images
    5. Cache results for future reference

    Guidelines:
    - Enhance vague prompts with artistic details
    - Use negative prompts to avoid artifacts
    - Consider aspect ratio and composition
    - Assign confidence scores based on prompt clarity
    """,
    deps_type=ImageGenDependencies,
    output_type=ImageGenerationOutput
)

@image_gen_agent.tool
async def generate_image(
    ctx: RunContext[ImageGenDependencies],
    prompt: str,
    negative_prompt: str = "",
    num_steps: int = 4,
    width: int = 1024,
    height: int = 1024
) -> dict:
    """
    Generate an image using Flux.1 Schnell.

    Args:
        ctx: Agent context
        prompt: Text description of desired image
        negative_prompt: Things to avoid in image
        num_steps: Inference steps (1-4 for Schnell)
        width: Image width (multiple of 64)
        height: Image height (multiple of 64)

    Returns:
        Generation result with image path and metadata
    """
    pipeline = ctx.deps.image_pipeline

    # Generate image
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        width=width,
        height=height,
        guidance_scale=0.0,  # Schnell doesn't use CFG
    )

    image = result.images[0]

    # Save to storage
    image_id = f"img_{hash(prompt)[:8]}_{int(time.time())}"
    image_path = ctx.deps.storage_path / f"{image_id}.png"
    image.save(image_path)

    # Store metadata
    await ctx.deps.storage.set(
        f"image:{image_id}",
        {
            "prompt": prompt,
            "path": str(image_path),
            "width": width,
            "height": height,
            "steps": num_steps,
            "timestamp": time.time()
        },
        ttl=86400  # 24 hours
    )

    return {
        "image_id": image_id,
        "path": str(image_path),
        "prompt": prompt,
        "dimensions": f"{width}x{height}"
    }

@image_gen_agent.tool
async def enhance_prompt(
    ctx: RunContext[ImageGenDependencies],
    simple_prompt: str,
    style: str = "photorealistic"
) -> str:
    """
    Enhance a simple prompt with artistic details.

    Args:
        ctx: Agent context
        simple_prompt: Basic description
        style: Desired style (photorealistic, artistic, etc.)

    Returns:
        Enhanced prompt optimized for Flux
    """
    # Use local LLM to enhance prompt
    style_guides = {
        "photorealistic": "professional photography, 8k, highly detailed, sharp focus, natural lighting",
        "artistic": "beautiful artwork, detailed illustration, artstation trending, concept art",
        "cinematic": "cinematic lighting, dramatic composition, film still, 35mm photograph"
    }

    enhancement = style_guides.get(style, "")
    enhanced = f"{simple_prompt}, {enhancement}"

    return enhanced
```

#### Option B: Tool Within Orchestrator

**Pros**:
- Simpler architecture
- Direct access from orchestrator
- Lower latency

**Cons**:
- Resource contention with LLM inference
- Orchestrator becomes heavyweight
- Harder to scale

**When to Use**: Infrequent image generation, development/testing

#### Option C: External Service (Recommended for Production)

**Pros**:
- Can run on dedicated GPU hardware
- Queue-based async processing
- Independent scaling and monitoring
- Doesn't block agent operations

**Cons**:
- More infrastructure complexity
- Network overhead

**Implementation**:
```python
# services/image_generation_service.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import asyncio
from pathlib import Path
import uuid

app = FastAPI(title="Image Generation Service")

# Lazy load model on first request
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from diffusers import FluxPipeline
        import torch

        _pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float16
        )
        _pipeline.enable_model_cpu_offload()

    return _pipeline

# In-memory task queue (use Redis in production)
task_queue = {}

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 1024
    height: int = 1024
    num_steps: int = 4

class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    image_path: Optional[str] = None
    error: Optional[str] = None

async def process_generation(task_id: str, request: ImageRequest):
    """Background task to generate image"""
    try:
        task_queue[task_id]["status"] = "processing"

        pipeline = get_pipeline()
        result = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_steps,
            width=request.width,
            height=request.height,
        )

        # Save image
        output_path = Path(f"outputs/{task_id}.png")
        output_path.parent.mkdir(exist_ok=True)
        result.images[0].save(output_path)

        task_queue[task_id].update({
            "status": "completed",
            "image_path": str(output_path)
        })

    except Exception as e:
        task_queue[task_id].update({
            "status": "failed",
            "error": str(e)
        })

@app.post("/generate", response_model=TaskStatus)
async def generate_image(
    request: ImageRequest,
    background_tasks: BackgroundTasks
):
    """Queue image generation task"""
    task_id = str(uuid.uuid4())
    task_queue[task_id] = {
        "status": "pending",
        "request": request.dict()
    }

    background_tasks.add_task(process_generation, task_id, request)

    return TaskStatus(task_id=task_id, status="pending")

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_status(task_id: str):
    """Check task status"""
    if task_id not in task_queue:
        return TaskStatus(task_id=task_id, status="not_found")

    task = task_queue[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        image_path=task.get("image_path"),
        error=task.get("error")
    )

# Agent tool to interact with service
@orchestrator_agent.tool
async def request_image_generation(
    ctx: RunContext[OrchestratorDependencies],
    prompt: str,
    enhanced: bool = True
) -> dict:
    """
    Request image generation from service.

    Args:
        ctx: Agent context
        prompt: Image description
        enhanced: Whether to enhance prompt first

    Returns:
        Task information
    """
    import httpx

    # Optionally enhance prompt using orchestrator's LLM
    if enhanced:
        # Could delegate to image gen agent or use local LLM
        pass

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:9000/generate",
            json={"prompt": prompt}
        )
        task_data = response.json()

    return {
        "task_id": task_data["task_id"],
        "service_url": f"http://localhost:9000/status/{task_data['task_id']}"
    }
```

### 3.2 Production Queue Architecture (Celery + Redis)

**Best for**: High-volume, production deployments

```python
# tasks/image_generation.py
from celery import Celery
from diffusers import FluxPipeline
import torch
from pathlib import Path

app = Celery('image_gen', broker='redis://localhost:6379/0')

# Load model once when worker starts
pipeline = None

@app.task(bind=True)
def generate_image_task(self, prompt: str, **kwargs):
    global pipeline

    if pipeline is None:
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float16
        )
        pipeline.enable_model_cpu_offload()

    try:
        result = pipeline(prompt=prompt, **kwargs)

        task_id = self.request.id
        output_path = Path(f"outputs/{task_id}.png")
        result.images[0].save(output_path)

        return {
            "status": "success",
            "image_path": str(output_path),
            "prompt": prompt
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# FastAPI endpoint
from fastapi import FastAPI
from celery.result import AsyncResult

api = FastAPI()

@api.post("/generate")
async def queue_generation(prompt: str):
    task = generate_image_task.delay(prompt)
    return {"task_id": task.id}

@api.get("/status/{task_id}")
async def check_status(task_id: str):
    task_result = AsyncResult(task_id, app=app)
    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else None
    }
```

### 3.3 Recommended Architecture for A2A

```
┌─────────────────────────────────────────────────────────┐
│                 Orchestrator Agent                      │
│                  (llama3.1:8b)                          │
└────────────┬────────────────────────────────────────────┘
             │
             │ Delegates to specialists
             │
    ┌────────┼────────┬──────────────┬──────────────┐
    │        │        │              │              │
┌───▼───┐ ┌─▼────┐ ┌─▼─────┐  ┌────▼──────────────▼──┐
│Analyst│ │Coder │ │Validate│  │ Image Gen Service    │
└───────┘ └──────┘ └────────┘  │ (FastAPI + Celery)   │
                                │                       │
                                │  ┌─────────────────┐  │
                                │  │ Redis Queue     │  │
                                │  └────────┬────────┘  │
                                │           │           │
                                │  ┌────────▼────────┐  │
                                │  │ Worker 1        │  │
                                │  │ (Flux Pipeline) │  │
                                │  └─────────────────┘  │
                                │  ┌─────────────────┐  │
                                │  │ Worker 2        │  │
                                │  │ (SVD Pipeline)  │  │
                                │  └─────────────────┘  │
                                └───────────────────────┘
```

---

## 4. Prompt Engineering Strategies

### 4.1 LLM to Image Prompt Conversion

#### Approach 1: LLM-Grounded Diffusion (LMD)

**Concept**: Use LLM to generate structured scene layout before image generation

```python
from pydantic import BaseModel
from typing import List

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    description: str

class SceneLayout(BaseModel):
    objects: List[BoundingBox]
    background: str
    lighting: str
    style: str

async def llm_to_layout(user_prompt: str) -> SceneLayout:
    """
    Convert natural language to structured layout.

    Example:
    Input: "A cat sitting on a red couch in a cozy living room"
    Output: SceneLayout with positioned objects
    """
    # Use orchestrator or specialist agent
    result = await orchestrator_agent.run(
        f"""Convert this image request into a structured scene layout:

        User request: {user_prompt}

        Provide:
        1. Object positions (bounding boxes 0-1 normalized)
        2. Background description
        3. Lighting description
        4. Artistic style
        """,
        deps=orchestrator_deps
    )

    return result.data  # Assumes structured output

async def layout_to_image(layout: SceneLayout) -> str:
    """Generate image from structured layout using ControlNet"""
    # Create composition guide from bounding boxes
    # Use ControlNet for structure + text prompt for details
    pass
```

#### Approach 2: Direct Prompt Enhancement

**Simpler and more practical for most use cases**

```python
@orchestrator_agent.tool
async def enhance_image_prompt(
    ctx: RunContext[OrchestratorDependencies],
    simple_description: str,
    style: str = "photorealistic",
    quality_level: str = "high"
) -> str:
    """
    Enhance a simple description into optimized Flux prompt.

    Args:
        simple_description: Basic user intent
        style: Desired artistic style
        quality_level: Quality modifiers to add

    Returns:
        Enhanced prompt ready for Flux
    """

    style_templates = {
        "photorealistic": [
            "professional photography",
            "8k resolution",
            "highly detailed",
            "sharp focus",
            "natural lighting",
            "depth of field"
        ],
        "artistic": [
            "digital art",
            "artstation trending",
            "concept art",
            "detailed illustration",
            "vibrant colors"
        ],
        "cinematic": [
            "cinematic lighting",
            "dramatic composition",
            "film still",
            "35mm photograph",
            "bokeh background"
        ],
        "anime": [
            "anime style",
            "manga illustration",
            "Studio Ghibli style",
            "detailed anime art"
        ]
    }

    quality_modifiers = {
        "high": "masterpiece, best quality, ultra-detailed",
        "medium": "good quality, detailed",
        "draft": "simple, basic"
    }

    components = [
        simple_description,
        ", ".join(style_templates.get(style, [])),
        quality_modifiers.get(quality_level, "")
    ]

    enhanced = ", ".join(filter(None, components))

    return enhanced

# Example usage in orchestrator workflow
async def handle_image_request(user_message: str):
    # Extract intent
    if "create image" in user_message.lower():
        # Extract description
        description = extract_description(user_message)

        # Enhance prompt
        enhanced_prompt = await enhance_image_prompt(
            ctx,
            description,
            style="photorealistic",
            quality_level="high"
        )

        # Generate image
        result = await generate_image(ctx, enhanced_prompt)

        return result
```

#### Approach 3: Few-Shot Prompting with Ollama

**Use local LLM to improve prompts via examples**

```python
# ComfyUI-style approach adapted for our system
async def improve_prompt_with_llm(
    base_prompt: str,
    model: str = "llama3.2:3b"
) -> str:
    """
    Use local LLM to enhance prompt quality.
    """

    system_prompt = """You are an expert at writing prompts for Flux image generation.

    Transform simple descriptions into detailed, effective prompts.

    Examples:

    Input: "a dog"
    Output: "a golden retriever dog sitting in a sunny park, professional pet photography, natural lighting, shallow depth of field, 8k, highly detailed fur texture"

    Input: "cyberpunk city"
    Output: "futuristic cyberpunk metropolis at night, neon signs reflecting on wet streets, flying cars, towering skyscrapers, volumetric fog, cinematic composition, blade runner style, highly detailed, 8k"

    Input: "mountain landscape"
    Output: "majestic mountain range at golden hour, snow-capped peaks, alpine meadow with wildflowers in foreground, dramatic clouds, professional landscape photography, wide angle lens, vivid colors, 8k"

    Key principles:
    - Add specific details about subject, lighting, composition
    - Include technical photography terms when appropriate
    - Specify quality indicators (8k, detailed, etc.)
    - Add style references for consistency
    """

    # Call local Ollama model
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": f"{system_prompt}\n\nInput: {base_prompt}\nOutput:",
                "stream": False
            }
        )
        result = response.json()
        enhanced = result["response"].strip()

    return enhanced
```

### 4.2 Prompt Quality Validation

```python
@image_gen_agent.tool
async def validate_prompt_quality(
    ctx: RunContext[ImageGenDependencies],
    prompt: str
) -> dict:
    """
    Validate prompt quality before generation.

    Checks:
    - Length (not too short/long)
    - Specificity (has concrete details)
    - Coherence (makes logical sense)
    - Safety (no harmful content)

    Returns:
        Validation result with score and suggestions
    """
    issues = []
    score = 1.0

    # Length check
    word_count = len(prompt.split())
    if word_count < 5:
        issues.append("Prompt too short - add more details")
        score -= 0.3
    elif word_count > 100:
        issues.append("Prompt too long - may confuse model")
        score -= 0.2

    # Specificity check
    generic_words = ["thing", "stuff", "something", "nice", "good"]
    if any(word in prompt.lower() for word in generic_words):
        issues.append("Prompt too generic - be more specific")
        score -= 0.2

    # Contradiction check
    contradictions = [
        ("day", "night"),
        ("sunny", "rainy"),
        ("indoor", "outdoor")
    ]
    for word1, word2 in contradictions:
        if word1 in prompt.lower() and word2 in prompt.lower():
            issues.append(f"Contradiction: contains both '{word1}' and '{word2}'")
            score -= 0.3

    return {
        "valid": score > 0.5,
        "score": max(0, score),
        "issues": issues,
        "prompt": prompt
    }
```

### 4.3 Negative Prompts

**Purpose**: Specify what to avoid in generated images

```python
COMMON_NEGATIVE_PROMPTS = {
    "photorealistic": (
        "cartoon, anime, illustration, painting, drawing, "
        "oversaturated, ugly, distorted, blurry, low quality, "
        "watermark, text, logo"
    ),
    "artistic": (
        "photograph, photorealistic, ugly, poorly drawn, "
        "low quality, blurry, distorted"
    ),
    "portraits": (
        "multiple heads, deformed, mutated hands, extra limbs, "
        "poorly drawn face, bad anatomy, low quality, blurry"
    ),
    "general": (
        "low quality, worst quality, low resolution, blurry, "
        "jpeg artifacts, watermark, signature, text"
    )
}

async def get_negative_prompt(style: str) -> str:
    """Get appropriate negative prompt for style"""
    return COMMON_NEGATIVE_PROMPTS.get(style, COMMON_NEGATIVE_PROMPTS["general"])
```

---

## 5. Image Quality Validation

### 5.1 CLIP-Based Aesthetic Scoring

**Purpose**: Automatically evaluate generated image quality

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np

class AestheticScorer:
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def score_image(self, image: Image.Image) -> float:
        """
        Score image aesthetics using CLIP prompting approach.

        Returns:
            Aesthetic score (0.0 to 1.0)
        """
        positive_prompt = "a beautiful, high-quality, professional photograph"
        negative_prompt = "an ugly, low-quality, amateur photograph"

        inputs = self.processor(
            text=[positive_prompt, negative_prompt],
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

            # Calculate cosine similarity
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            # Calculate similarities
            similarities = (image_embeds @ text_embeds.T).squeeze()
            positive_sim = similarities[0].item()
            negative_sim = similarities[1].item()

            # Normalize to 0-1 scale
            score = (positive_sim - negative_sim + 2) / 4

        return max(0.0, min(1.0, score))

    def detailed_score(self, image: Image.Image) -> dict:
        """
        Provide detailed aesthetic metrics.
        """
        aspects = {
            "composition": "well-composed, balanced, rule of thirds",
            "lighting": "excellent lighting, professional photography lighting",
            "colors": "vibrant colors, color grading, cinematic colors",
            "sharpness": "sharp focus, high detail, 8k quality",
            "style": "artistic, creative, unique perspective"
        }

        scores = {}
        for aspect, prompt in aspects.items():
            inputs = self.processor(
                text=[prompt, f"poorly {aspect}"],
                images=image,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds

                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

                similarities = (image_embeds @ text_embeds.T).squeeze()
                score = (similarities[0] - similarities[1] + 2) / 4
                scores[aspect] = max(0.0, min(1.0, score.item()))

        scores["overall"] = sum(scores.values()) / len(scores)

        return scores

# Integration with image generation agent
@image_gen_agent.tool
async def validate_generated_image(
    ctx: RunContext[ImageGenDependencies],
    image_path: str,
    min_quality: float = 0.5
) -> dict:
    """
    Validate generated image quality using CLIP.

    Args:
        ctx: Agent context
        image_path: Path to generated image
        min_quality: Minimum acceptable quality score

    Returns:
        Validation result with scores and pass/fail
    """
    scorer = ctx.deps.aesthetic_scorer
    image = Image.open(image_path)

    overall_score = scorer.score_image(image)
    detailed_scores = scorer.detailed_score(image)

    passed = overall_score >= min_quality

    return {
        "passed": passed,
        "overall_score": overall_score,
        "detailed_scores": detailed_scores,
        "threshold": min_quality,
        "recommendation": "accept" if passed else "regenerate with improved prompt"
    }
```

### 5.2 Alternative Validation Methods

#### Simple Perceptual Checks
```python
from PIL import Image
import numpy as np

def basic_image_checks(image_path: str) -> dict:
    """
    Perform basic image quality checks.
    """
    image = Image.open(image_path)
    img_array = np.array(image)

    # Check dimensions
    width, height = image.size
    aspect_ratio = width / height

    # Check brightness
    grayscale = np.mean(img_array, axis=2)
    brightness = np.mean(grayscale)

    # Check contrast
    contrast = np.std(grayscale)

    # Check color saturation
    if len(img_array.shape) == 3:
        hsv = np.array(image.convert('HSV'))
        saturation = np.mean(hsv[:, :, 1])
    else:
        saturation = 0

    # Sharpness estimate (variance of Laplacian)
    from scipy import ndimage
    laplacian = ndimage.laplace(grayscale)
    sharpness = np.var(laplacian)

    return {
        "dimensions": f"{width}x{height}",
        "aspect_ratio": round(aspect_ratio, 2),
        "brightness": round(brightness / 255, 2),  # 0-1 scale
        "contrast": round(contrast / 128, 2),      # 0-1 scale
        "saturation": round(saturation / 255, 2),  # 0-1 scale
        "sharpness": round(min(sharpness / 1000, 1), 2),  # 0-1 scale
        "file_size_mb": os.path.getsize(image_path) / (1024 * 1024)
    }
```

#### LAION Aesthetic Predictor
```python
# Alternative: Use pre-trained LAION aesthetic predictor
from transformers import pipeline

class LAIONAestheticPredictor:
    def __init__(self):
        # Uses linear probe on CLIP embeddings
        # Trained on LAION dataset aesthetic ratings
        self.predictor = pipeline(
            "image-classification",
            model="cafeai/cafe_aesthetic"
        )

    def score(self, image_path: str) -> float:
        """
        Score image using LAION aesthetic predictor.
        Returns score from 1-10.
        """
        result = self.predictor(image_path)
        # Extract aesthetic score from result
        score = result[0]['score'] if result else 5.0
        return score
```

---

## 6. Implementation Checklist

### 6.1 Dependencies and Installation

```bash
# Core diffusion libraries
pip install diffusers transformers accelerate

# Image processing
pip install pillow opencv-python numpy

# Async task queue (production)
pip install celery redis

# Quality assessment
pip install torchvision

# Optional optimizations
pip install xformers  # Faster attention (CUDA only)
pip install bitsandbytes  # Quantization support

# Video generation
pip install imageio imageio-ffmpeg
```

### 6.2 Model Download and Setup

```python
# setup_models.py
"""
Script to download and cache models for local deployment.
"""
import os
from diffusers import FluxPipeline, StableVideoDiffusionPipeline
import torch

# Set HuggingFace cache directory
CACHE_DIR = os.getenv("HF_HOME", "/path/to/models/cache")

def download_flux_schnell():
    """Download Flux.1 Schnell (~24GB)"""
    print("Downloading Flux.1 Schnell...")
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    print("Flux.1 Schnell downloaded successfully")
    return pipeline

def download_sdxl():
    """Download SDXL base (~7GB)"""
    print("Downloading SDXL...")
    from diffusers import StableDiffusionXLPipeline

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    print("SDXL downloaded successfully")
    return pipeline

def download_svd():
    """Download Stable Video Diffusion (~16GB)"""
    print("Downloading SVD...")
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    print("SVD downloaded successfully")
    return pipeline

if __name__ == "__main__":
    # Download models
    download_flux_schnell()
    download_sdxl()
    download_svd()
    print("\nAll models downloaded successfully!")
    print(f"Cache location: {CACHE_DIR}")
```

### 6.3 Integration Steps

#### Step 1: Add Dependencies to requirements.txt

```txt
# Append to existing requirements.txt

# Image/Video Generation
diffusers>=0.31.0
transformers>=4.45.0
accelerate>=0.34.0
safetensors>=0.4.0

# Image Processing
pillow>=10.0.0
opencv-python>=4.8.0
numpy>=1.24.0

# Quality Assessment
torchvision>=0.16.0

# Async Processing (optional - for production)
celery>=5.3.0
flower>=2.0.0  # Celery monitoring

# Video Processing
imageio>=2.31.0
imageio-ffmpeg>=0.4.9
```

#### Step 2: Create Image Generation Dependencies

```python
# models/dependencies.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from storage.interfaces import StorageInterface

@dataclass
class ImageGenDependencies:
    """Dependencies for image generation agent."""
    agent_id: str
    storage: StorageInterface
    storage_path: Path
    image_pipeline: Optional[object] = None  # FluxPipeline instance
    video_pipeline: Optional[object] = None  # SVDPipeline instance
    aesthetic_scorer: Optional[object] = None  # AestheticScorer instance
    cache_enabled: bool = True
```

#### Step 3: Create Output Schemas

```python
# models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List

class ImageGenerationOutput(BaseModel):
    """Output schema for image generation agent."""
    image_id: str = Field(description="Unique identifier for generated image")
    image_path: str = Field(description="File path to generated image")
    prompt: str = Field(description="Prompt used for generation")
    enhanced_prompt: str = Field(description="Enhanced prompt with quality modifiers")
    dimensions: str = Field(description="Image dimensions (WxH)")
    quality_score: Optional[float] = Field(
        default=None,
        description="Aesthetic quality score (0-1)"
    )
    generation_time_seconds: float = Field(description="Time taken to generate")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in output quality"
    )
    reasoning: str = Field(description="Explanation of generation process")

class VideoGenerationOutput(BaseModel):
    """Output schema for video generation agent."""
    video_id: str
    video_path: str
    input_image: str
    num_frames: int
    fps: int
    duration_seconds: float
    generation_time_seconds: float
    motion_level: str  # low, medium, high
    confidence: float
    reasoning: str
```

#### Step 4: Create Image Generation Agent

```python
# agents/specialists/image_generator.py
# (See Section 3.1 for full implementation)
```

#### Step 5: Register Agent in Main Application

```python
# main.py
from agents.specialists.image_generator import image_gen_agent
from models.dependencies import ImageGenDependencies
from diffusers import FluxPipeline
import torch
from pathlib import Path

# ... existing imports ...

# Initialize image generation pipeline (lazy load alternative)
_image_pipeline = None

def get_image_pipeline():
    global _image_pipeline
    if _image_pipeline is None:
        _image_pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float16
        )
        _image_pipeline.enable_model_cpu_offload()
    return _image_pipeline

# Create image generation agent app
async def get_image_gen_deps() -> ImageGenDependencies:
    """Dependency injection for image generation agent."""
    return ImageGenDependencies(
        agent_id="image-generator",
        storage=await get_storage(),
        storage_path=Path("outputs/images"),
        image_pipeline=get_image_pipeline(),
        cache_enabled=True
    )

from a2a.server import create_a2a_app

image_gen_app = create_a2a_app(
    agent=image_gen_agent,
    deps_factory=get_image_gen_deps,
    agent_id="image-generator",
    description="Local image generation using Flux.1 Schnell"
)

# Run: uvicorn main:image_gen_app --port 8004
```

#### Step 6: Update Orchestrator with Image Generation Awareness

```python
# agents/orchestrator.py
from a2a.client import A2AClient

@orchestrator_agent.tool
async def delegate_image_generation(
    ctx: RunContext[OrchestratorDependencies],
    task_description: str,
    style: str = "photorealistic"
) -> dict:
    """
    Delegate image generation to specialist agent.

    Args:
        ctx: Orchestrator context
        task_description: What image to generate
        style: Artistic style to use

    Returns:
        Image generation result
    """
    async with A2AClient("http://localhost:8004") as client:
        response = await client.send_message(
            message=f"Generate image: {task_description}, style: {style}"
        )

    return response["result"]["output"]
```

#### Step 7: Docker Configuration (Optional)

```yaml
# docker-compose.yml additions

services:
  # ... existing services ...

  image-generator:
    build: .
    command: uvicorn main:image_gen_app --host 0.0.0.0 --port 8004
    ports:
      - "8004:8004"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - REDIS_URL=redis://redis:6379
      - HF_HOME=/models/cache
    volumes:
      - ./outputs:/app/outputs
      - ./models:/models/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis
      - ollama

  # Optional: Celery worker for async processing
  celery-worker:
    build: .
    command: celery -A tasks.image_generation worker --loglevel=info
    environment:
      - REDIS_URL=redis://redis:6379
      - HF_HOME=/models/cache
    volumes:
      - ./outputs:/app/outputs
      - ./models:/models/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis
```

### 6.4 Testing Strategy

```python
# tests/test_image_generation.py
import pytest
from pathlib import Path
from agents.specialists.image_generator import image_gen_agent
from models.dependencies import ImageGenDependencies
from storage.memory_storage import InMemoryStorage

@pytest.fixture
async def image_gen_deps():
    """Create test dependencies."""
    storage = InMemoryStorage()
    return ImageGenDependencies(
        agent_id="test-image-gen",
        storage=storage,
        storage_path=Path("tests/outputs"),
        cache_enabled=False
    )

@pytest.mark.asyncio
async def test_prompt_enhancement(image_gen_deps):
    """Test prompt enhancement tool."""
    result = await image_gen_agent.run(
        "Enhance this prompt: a sunset",
        deps=image_gen_deps
    )

    assert "sunset" in result.data.enhanced_prompt.lower()
    assert len(result.data.enhanced_prompt) > len("a sunset")

@pytest.mark.asyncio
async def test_image_generation(image_gen_deps):
    """Test actual image generation (requires GPU)."""
    result = await image_gen_agent.run(
        "Generate a simple test image of a red circle on white background",
        deps=image_gen_deps
    )

    assert result.data.image_path is not None
    assert Path(result.data.image_path).exists()
    assert result.data.confidence > 0.5

@pytest.mark.asyncio
async def test_quality_validation(image_gen_deps):
    """Test image quality validation."""
    # Generate image first
    gen_result = await image_gen_agent.run(
        "Generate high quality professional photograph of a coffee cup",
        deps=image_gen_deps
    )

    # Validate quality
    assert gen_result.data.quality_score is not None
    assert 0.0 <= gen_result.data.quality_score <= 1.0
```

---

## 7. Cost/Benefit Analysis: Local vs. API Generation

### 7.1 Local Generation

**Pros:**
- **Zero API costs**: No per-image charges
- **Complete privacy**: Data never leaves infrastructure
- **No rate limits**: Generate as many images as hardware allows
- **Customization**: Full control over models, LoRAs, fine-tuning
- **Low latency**: No network round trips (if GPU available)
- **Offline capability**: Works without internet

**Cons:**
- **Hardware investment**: Requires GPU (12-24GB VRAM)
- **Maintenance overhead**: Model updates, storage management
- **Scaling challenges**: Limited by hardware capacity
- **Initial setup complexity**: Model downloads, configuration
- **Power consumption**: GPU power usage costs

**Cost Breakdown:**
```
Initial Investment:
- GPU (RTX 4090 24GB): $1,600-2,000
- Additional VRAM if needed: $0-500
- Power supply upgrade: $0-200
Total: ~$2,000

Operating Costs (monthly):
- Power (~300W GPU, 8h/day): ~$15-30
- Cooling/maintenance: ~$5-10
Total: ~$20-40/month

Break-even vs. API (at $0.04/image):
- 2,000 images/month: Break-even at ~40 months
- 5,000 images/month: Break-even at ~16 months
- 10,000 images/month: Break-even at ~8 months
```

### 7.2 API Generation (DALL-E, Midjourney, Stability AI)

**Pros:**
- **No hardware required**: Run on any machine
- **Instant scaling**: Handle any volume
- **No maintenance**: Provider handles updates
- **Latest models**: Automatic access to improvements
- **Simple integration**: REST API calls

**Cons:**
- **Per-image costs**: $0.02-0.08 per image typically
- **Rate limits**: Throttling at high volumes
- **Privacy concerns**: Data sent to third party
- **Dependency**: Reliant on external service
- **Network latency**: Request/response overhead

**Cost Breakdown:**
```
API Costs (examples):
- DALL-E 3: $0.040-0.080 per image
- Stability AI: $0.002-0.020 per image (SDXL)
- Midjourney: ~$0.05-0.10 per image (subscription model)

Monthly costs (at $0.04/image):
- 100 images: $4
- 1,000 images: $40
- 5,000 images: $200
- 10,000 images: $400
```

### 7.3 Recommendation Matrix

| Use Case | Volume/Month | Recommendation | Rationale |
|----------|--------------|----------------|-----------|
| Development/Testing | <100 | API | Low cost, quick setup |
| Small Business | 100-1,000 | API | Cost-effective, no hardware |
| Growing Startup | 1,000-5,000 | Hybrid | API for peaks, local for base load |
| Enterprise | 5,000+ | Local | Cost savings, privacy, control |
| Research | Variable | Local | Experimentation, customization |
| Privacy-Critical | Any | Local | Data sovereignty required |

### 7.4 Hybrid Approach

**Recommended for most A2A deployments:**

```python
class HybridImageGenerator:
    """
    Use local generation as primary, fallback to API for peaks.
    """
    def __init__(
        self,
        local_pipeline,
        api_client,
        local_queue_size: int = 5
    ):
        self.local_pipeline = local_pipeline
        self.api_client = api_client
        self.queue = asyncio.Queue(maxsize=local_queue_size)

    async def generate(self, prompt: str, **kwargs) -> Image:
        """
        Try local first, fallback to API if queue full.
        """
        try:
            # Try to add to local queue (non-blocking)
            self.queue.put_nowait((prompt, kwargs))
            result = await self._process_local()
            return result
        except asyncio.QueueFull:
            # Queue full, use API
            logger.info(f"Local queue full, using API for: {prompt}")
            return await self._process_api(prompt, **kwargs)

    async def _process_local(self) -> Image:
        prompt, kwargs = await self.queue.get()
        result = self.local_pipeline(prompt=prompt, **kwargs)
        return result.images[0]

    async def _process_api(self, prompt: str, **kwargs) -> Image:
        # Call external API (OpenAI, Stability, etc.)
        return await self.api_client.generate(prompt, **kwargs)
```

---

## 8. Example Integration Code

### 8.1 Complete Minimal Example

```python
# examples/image_generation_demo.py
"""
Minimal working example of image generation in A2A system.
"""
import asyncio
from pathlib import Path
from diffusers import FluxPipeline
import torch
from a2a.client import A2AClient

async def main():
    # Example 1: Direct image generation
    print("Loading Flux Schnell pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()

    prompt = "a serene mountain lake at sunset, professional photography"
    print(f"Generating image: {prompt}")

    result = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0
    )

    output_path = Path("outputs/demo_image.png")
    output_path.parent.mkdir(exist_ok=True)
    result.images[0].save(output_path)
    print(f"Image saved to: {output_path}")

    # Example 2: Through A2A orchestrator
    print("\nTesting through A2A orchestrator...")
    async with A2AClient("http://localhost:8000") as orchestrator:
        response = await orchestrator.send_message(
            message="Create an image of a futuristic city with flying cars"
        )
        print("Orchestrator response:", response["result"]["output"])

if __name__ == "__main__":
    asyncio.run(main())
```

### 8.2 Advanced Workflow Example

```python
# examples/advanced_image_workflow.py
"""
Advanced example: Orchestrator coordinates multiple agents for complex image task.
"""
import asyncio
from a2a.client import A2AClient

async def creative_image_workflow():
    """
    Complex workflow:
    1. User provides vague idea
    2. Analyst extracts key concepts
    3. Image gen agent creates prompt
    4. Generate multiple variations
    5. Validate quality
    6. Return best result
    """

    user_request = """
    I need marketing images for a new eco-friendly water bottle.
    It should look premium but also show the environmental aspect.
    """

    async with A2AClient("http://localhost:8000") as orchestrator:
        response = await orchestrator.send_message(
            message=f"""
            Create marketing images for this product:

            {user_request}

            Process:
            1. Analyze the product requirements and extract key visual concepts
            2. Generate 3 different image prompt variations emphasizing:
               - Premium quality aesthetic
               - Environmental sustainability
               - Product usability
            3. Generate images for each prompt
            4. Validate quality and select best
            5. Provide rationale for selection
            """
        )

        result = response["result"]["output"]

        print("Task Results:")
        for task in result.get("task_results", []):
            print(f"\nAgent: {task['agent']}")
            print(f"Result: {task['result']}")

        print("\nFinal Synthesis:")
        print(result.get("synthesis", ""))

if __name__ == "__main__":
    asyncio.run(creative_image_workflow())
```

---

## 9. Monitoring and Optimization

### 9.1 Performance Metrics

```python
# monitoring/image_gen_metrics.py
import time
import psutil
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class GenerationMetrics:
    """Metrics for image generation performance."""
    prompt: str
    start_time: float
    end_time: float
    generation_time_seconds: float
    vram_used_mb: float
    vram_peak_mb: float
    cpu_percent: float
    success: bool
    error: Optional[str] = None

class MetricsCollector:
    """Collect and track generation metrics."""

    def __init__(self):
        self.metrics: list[GenerationMetrics] = []

    async def track_generation(self, prompt: str, generator_func):
        """
        Wrap generation function with metrics collection.
        """
        start_time = time.time()

        # Pre-generation metrics
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / 1024**2

        try:
            result = await generator_func(prompt)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)

        # Post-generation metrics
        end_time = time.time()
        vram_after = torch.cuda.memory_allocated() / 1024**2
        vram_peak = torch.cuda.max_memory_allocated() / 1024**2
        cpu_percent = psutil.cpu_percent()

        metrics = GenerationMetrics(
            prompt=prompt,
            start_time=start_time,
            end_time=end_time,
            generation_time_seconds=end_time - start_time,
            vram_used_mb=vram_after - vram_before,
            vram_peak_mb=vram_peak,
            cpu_percent=cpu_percent,
            success=success,
            error=error
        )

        self.metrics.append(metrics)

        return result, metrics

    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.metrics:
            return {}

        successful = [m for m in self.metrics if m.success]

        return {
            "total_generations": len(self.metrics),
            "successful": len(successful),
            "failed": len(self.metrics) - len(successful),
            "avg_generation_time": sum(m.generation_time_seconds for m in successful) / len(successful) if successful else 0,
            "avg_vram_usage_mb": sum(m.vram_used_mb for m in successful) / len(successful) if successful else 0,
            "peak_vram_mb": max(m.vram_peak_mb for m in successful) if successful else 0,
        }
```

### 9.2 Optimization Techniques

```python
# optimization/pipeline_optimizations.py
"""
Various optimization techniques for faster/lower-memory generation.
"""
import torch
from diffusers import FluxPipeline

def create_optimized_pipeline(
    model_name: str,
    optimization_level: str = "balanced"
) -> FluxPipeline:
    """
    Create optimized pipeline based on hardware constraints.

    Args:
        model_name: Model to load
        optimization_level: "speed", "balanced", or "memory"

    Returns:
        Optimized pipeline
    """

    if optimization_level == "speed":
        # Maximum speed, highest VRAM usage
        pipe = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        pipe.to("cuda")

        # Enable xformers for faster attention
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass

    elif optimization_level == "balanced":
        # Balance speed and memory
        pipe = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload()

    elif optimization_level == "memory":
        # Minimum VRAM, slower generation
        pipe = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp8"  # If available
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

    # Compile UNet for faster inference (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        pipe.unet = torch.compile(
            pipe.unet,
            mode="reduce-overhead",
            fullgraph=True
        )

    return pipe

# Usage
pipeline = create_optimized_pipeline(
    "black-forest-labs/FLUX.1-schnell",
    optimization_level="balanced"
)
```

---

## 10. Summary and Next Steps

### 10.1 Key Takeaways

1. **Image Generation**:
   - **Flux.1 Schnell** is the best choice for local deployment (fast, high quality)
   - Requires 12-24GB VRAM (can work with 6GB quantized)
   - Diffusers library provides unified API

2. **Video Generation**:
   - **Stable Video Diffusion** is the primary local option
   - 14-25 frames, image-to-video only
   - Requires 16-24GB VRAM minimum
   - Better suited for specialized use cases

3. **Integration**:
   - **Recommended**: Separate service architecture with async queue
   - Use FastAPI + Celery + Redis for production
   - Agent-based approach fits A2A pattern well

4. **Prompt Engineering**:
   - LLM enhancement dramatically improves results
   - Use local Ollama for prompt improvement
   - Negative prompts are essential for quality

5. **Quality Validation**:
   - CLIP-based aesthetic scoring works well
   - Can automate quality checks
   - Useful for regeneration decisions

### 10.2 Recommended Implementation Path

1. **Phase 1 - Prototype** (Week 1):
   - Set up Flux Schnell with Diffusers
   - Create basic image generation tool
   - Test prompt enhancement with Ollama
   - Validate on development machine

2. **Phase 2 - Agent Integration** (Week 2):
   - Create image generation specialist agent
   - Add to orchestrator delegation logic
   - Implement quality validation
   - Add tests

3. **Phase 3 - Production Ready** (Week 3):
   - Set up async queue (Celery + Redis)
   - Deploy as separate service
   - Add monitoring and metrics
   - Optimize for target hardware

4. **Phase 4 - Advanced Features** (Week 4+):
   - Add LoRA support for style customization
   - Implement ControlNet for structured generation
   - Add SVD for video (if needed)
   - Fine-tune workflows based on usage

### 10.3 Hardware Recommendations by Use Case

| Use Case | GPU | Reason |
|----------|-----|---------|
| Development | RTX 4060 Ti 16GB | Affordable, sufficient for testing |
| Production (Low Volume) | RTX 4070 Ti 12GB | Good balance of cost/performance |
| Production (High Volume) | RTX 4090 24GB | Maximum throughput, best quality |
| Enterprise | Multiple 4090s | Parallel processing, redundancy |

### 10.4 Alternative Consideration: AMD/Intel

- **AMD**: ROCm support improving, but ecosystem less mature
- **Intel Arc**: Limited Flux support currently
- **Recommendation**: NVIDIA for now, reevaluate in 6-12 months

---

## Sources

### Image Generation Models
- [SDXL vs Flux1.dev models comparison - Stable Diffusion Art](https://stable-diffusion-art.com/sdxl-vs-flux/)
- [Flux.1: The New Stable Diffusion Killer? Complete Guide to Running it Locally](https://andy-wang.medium.com/flux-1-the-new-stable-diffusion-killer-complete-guide-to-running-it-locally-aece1c453691)
- [The Flux AI guide: installation, models, prompts and settings](https://andreaskuhr.com/en/flux-ai-guide.html)

### FastAPI Integration
- [Deploy stable diffusion on GPU instance using FastAPI](https://medium.com/@vishnuvig/deploy-stable-diffusion-on-gpu-instance-using-fastapi-d0743eeb735d)
- [GitHub - shyamsn97/stable-diffusion-server](https://github.com/shyamsn97/stable-diffusion-server)
- [Serve a Stable Diffusion Model — Ray 2.51.1](https://docs.ray.io/en/latest/serve/tutorials/stable-diffusion.html)

### Video Generation
- [stabilityai/stable-video-diffusion-img2vid-xt · Hugging Face](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
- [Stable Video Diffusion - Hugging Face Diffusers](https://huggingface.co/docs/diffusers/en/using-diffusers/svd)
- [Image to Video Generation with Stable Video Diffusion — OpenVINO™ documentation](https://docs.openvino.ai/2024/notebooks/stable-video-diffusion-with-output.html)

### LoRA and ControlNet
- [GitHub - comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [How to use Control-Lora SDXL for ComfyUI](https://weirdwonderfulai.art/tutorial/how-to-use-control-lora-sdxl-for-comfyui/)
- [From Photos to Masterpieces: A Workflow for Generating Stylized Images with ControlNet and LoRA](https://comfyui.org/en/generating-stylized-images-with-controlnet-and-lora)

### Async Processing
- [Concurrency and async / await - FastAPI](https://fastapi.tiangolo.com/async/)
- [Async Architecture with FastAPI, Celery, and RabbitMQ](https://medium.com/cuddle-ai/async-architecture-with-fastapi-celery-and-rabbitmq-c7d029030377)
- [Asynchronous Tasks with FastAPI and Celery](https://testdriven.io/blog/fastapi-and-celery/)
- [Managing Background Tasks and Long-Running Operations in FastAPI](https://leapcell.io/blog/managing-background-tasks-and-long-running-operations-in-fastapi)

### Hardware Requirements
- [Flux 2 GGUF Quantized Models - Low VRAM Guide 2025](https://apatero.com/blog/flux-2-gguf-quantized-models-low-vram-guide)
- [How to Run Flux Image Models In ComfyUI with Low VRAM](https://sdxlturbo.ai/blog-how-to-run-flux-image-models-in-comfyui-with-low-vram-48624)
- [Say Goodbye to Lag: ComfyUI's Secret to Running Flux on 6 GB VRAM](https://medium.com/@lompojeanolivier/say-goodbye-to-lag-comfyuis-secret-to-running-flux-on-6-gb-vram-e5dcb1dde778)

### Prompt Engineering
- [Use a Local LLM to Generate Better Stable Diffusion Prompts](https://www.arsturn.com/blog/create-better-ai-art-how-to-use-a-local-llm-to-generate-stable-diffusion-prompts)
- [LLM-grounded Diffusion: Enhancing Prompt Understanding](https://arxiv.org/html/2305.13655v3)
- [ComfyUI Prompt Enhancement Guide: Using Ollama and LLMs](https://docs.jarvislabs.ai/blog/prompt-enhancing)
- [Improve your Stable Diffusion prompts with Retrieval Augmented Generation](https://aws.amazon.com/blogs/machine-learning/improve-your-stable-diffusion-prompts-with-retrieval-augmented-generation/)

### Quality Assessment
- [Image aesthetics quantification using OpenAI CLIP](https://medium.com/@sureshraghu0706/image-aesthetics-quantification-using-openai-clip-7bbb45e00147)
- [CLIP knows image aesthetics](https://www.frontiersin.org/articles/10.3389/frai.2022.976235/full)
- [GitHub - LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)
- [CLIP Image Quality Assessment (CLIP-IQA) — PyTorch-Metrics](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html)
