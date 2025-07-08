# clip_utils.py
import base64
from PIL import Image
import open_clip
import torch
import numpy as np
from torchvision import transforms
import together

# CLIP setup (still used for image embedding search)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32-quickgelu',
    pretrained='openai'
)
model.to(device)
model.eval()

def get_clip_embedding(image_path: str) -> np.ndarray:
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image).cpu().numpy().flatten()
    return embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity

# Together AI setup
client = together.Together(api_key="tgp_v1_HDWt-zkoh6PtS5aCanHP-MxOuOJaH7mQC-5DNdCbpXE")

def predict_place_name(image_path: str) -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[
            {
                "role": "system",
                "content": "You are an AI expert in landmark and monument identification. Given an image, your task is to identify and return the **specific name** of the place, such as a **monument, landmark, building, or tourist attraction** visible in the image.- Be as **precise** as possible. - Do **not** return a generic location like a city, country, or region unless that is the only identifiable detail. - If the image contains a globally recognized monument (e.g., 'Arc de Triomphe', 'Statue of Liberty', 'Taj Mahal'), return **only** the name of that monument. - If no famous landmark is present, then return the most specific place visible (e.g., 'Central Park', not just 'New York'). - Do **not** return any explanations, just the **exact name** of the place or structure. For example: - ✅ 'Eiffel Tower' - ✅ 'Golden Gate Bridge' - ✅ 'Petra Treasury' - 🚫 Not: 'Paris', 'California', or 'Jordan'. Return only the name of the place as a single string. No extra text or sentences."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=20,
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()
    return content if content else None