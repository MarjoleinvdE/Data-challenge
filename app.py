import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
import requests

# === CONFIGURATION ===
ONNX_MODEL_PATH = "best.onnx"  # Replace with your ONNX model path
INPUT_SIZE = (640, 640)

SPOONACULAR_API_KEY = "3dbbbe288307469587935ed84b35bc8d"  # Replace with your Spoonacular API key

# === Load YOLO ONNX Model ===
session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name

# === Image Preprocessing ===
def preprocess(image: Image.Image):
    image = image.resize(INPUT_SIZE)
    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC → CHW
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

# === Run ONNX inference ===
def run_onnx_inference(image):
    img_np = preprocess(image)
    outputs = session.run(None, {input_name: img_np})
    detections = outputs[0]  # Shape: [N, 6] or [1, N, 6]

    if len(detections.shape) == 3:
        detections = detections[0]

    class_map = {
        0: 'apple', 1: 'basket', 2: 'butter', 3: 'cereal', 4: 'chocolate_drink',
        5: 'cloth_opl', 6: 'coke', 7: 'crackers', 8: 'eggs', 9: 'garlic',
        10: 'grape_juice', 11: 'help_me_carry_opl', 12: 'lemon', 13: 'noodles',
        14: 'onion', 15: 'orange', 16: 'orange_juice', 17: 'paprika', 18: 'potato',
        19: 'potato_chips', 20: 'pringles', 21: 'sausages', 22: 'scrubby',
        23: 'sponge_opl', 24: 'sprite', 25: 'tomato', 26: 'tray'
    }

    detected_labels = []
    for det in detections:
        conf = float(det[4])
        if conf > 0.5:
            class_val = int(det[5])
            if 0 <= class_val <= 26:
                label = class_map[class_val]
                detected_labels.append(label)

    return list(set(detected_labels))

# === Spoonacular Recipe Search ===
def get_recipes_spoonacular(ingredients):
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        "apiKey": SPOONACULAR_API_KEY,
        "ingredients": ",".join(ingredients),
        "number": 5,
        "ranking": 1
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"Spoonacular API error: {response.status_code}"

    recipes = response.json()
    results = []
    for r in recipes:
        title = r.get("title")
        used_count = r.get("usedIngredientCount")
        missed_count = r.get("missedIngredientCount")
        image = r.get("image")
        recipe_id = r.get("id")
        results.append({
            "title": title,
            "used": used_count,
            "missed": missed_count,
            "image": image,
            "id": recipe_id
        })
    return results

def show_recipes(recipes):
    for r in recipes:
        st.markdown(f"### {r['title']}")
        st.image(r['image'], width=300)
        st.markdown(f"Used ingredients: {r['used']} | Missing ingredients: {r['missed']}")
        # Link to Spoonacular recipe page:
        link_title = r['title'].replace(' ', '-').lower()
        st.markdown(f"[See full recipe](https://spoonacular.com/recipes/{link_title}-{r['id']})")
        st.write("---")

# === Streamlit UI ===
st.title("📸🍽️ Food Detection & Recipe Recommendations")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Detecting food items..."):
        detected_items = run_onnx_inference(image)

    if detected_items:
        st.success(f"✅ Detected: {', '.join(detected_items)}")

        with st.spinner("🔎 Searching for recipes..."):
            recipes = get_recipes_spoonacular(detected_items)
            if isinstance(recipes, str):
                st.error(recipes)
            else:
                st.markdown("### 🍳 Recipes You Can Make:")
                show_recipes(recipes)
    else:
        st.warning("⚠️ No confident detections found.")
