import os
import pickle
from huggingface_hub import hf_hub_download
import tensorflow as tf

# -------------------------------
# Load SKLearn (.pkl) model
# -------------------------------
def load_sklearn_model():
    print("📥 Downloading sklearn model from Hugging Face...")

    model_path = hf_hub_download(
        repo_id=os.getenv("kathaneashish599/raksha-uid-model1"),
        filename="RandomForest_model.pkl",
        token=os.getenv("HF_TOKEN")  # optional if public
    )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("✅ Sklearn model loaded")
    return model


# -------------------------------
# Load Keras (.h5) model
# -------------------------------
def load_keras_model():
    print("📥 Downloading keras model from Hugging Face...")

    model_path = hf_hub_download(
        repo_id=os.getenv("kathaneashish599/raksha-uid-model1"),
        filename="aadhaar_classifier_final.h5",
        token=os.getenv("HF_TOKEN")  # optional if public
    )

    model = tf.keras.models.load_model(model_path)
    print("✅ Keras model loaded")
    return model
