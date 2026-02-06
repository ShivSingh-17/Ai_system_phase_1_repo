


import pickle
import numpy as np
from deepface import DeepFace

with open("face_database/face_embeddings.pkl", "rb") as f:
    FACE_DB = pickle.load(f)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize(face_img):
    rep = DeepFace.represent(
        face_img,
        model_name="Facenet",
        enforce_detection=False
    )[0]["embedding"]

    best_name = "Unknown"
    best_score = 0

    for name, emb_list in FACE_DB.items():
        for emb in emb_list:
            score = cosine(rep, emb)
            if score > 0.45 and score > best_score:
                best_name = name
                best_score = score

    return best_name