import gradio as gr
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    "waste_classification_model.keras"
)

# Load category names
with open("class_names.json", "r") as f:
    class_names = json.load(f)


def predict(image):
    if image is None:
        return {}, "Please upload a waste image."

    # Get image dimensions from model
    height = model.input_shape[1]
    width = model.input_shape[2]

    # Resize image
    image = image.convert("RGB")
    image = image.resize((width, height))

    # Convert image to array
    image_array = np.array(image).astype("float32")

    # Normalize pixel values
    image_array = image_array / 255.0

    # Predict category
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    category = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    results = {
        class_names[i]: float(predictions[i])
        for i in range(len(class_names))
    }

    return results, (
        f"Predicted Waste Category: {category}\n"
        f"Confidence: {confidence * 100:.2f}%"
    )


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Waste Image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Prediction Results"),
        gr.Textbox(label="Waste Classification Details")
    ],
    title="♻️ AI Waste Segregation System",
    description="Upload a waste image to identify its category."
)

if __name__ == "__main__":
    demo.launch()