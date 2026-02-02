


import pickle
import numpy as np
from deepface import DeepFace

MODEL = "Facenet"
THRESHOLD = 0.55

with open("face_embeddings.pkl", "rb") as f:
    FACE_DB = pickle.load(f)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize(face_img):
    rep = DeepFace.represent(
        face_img,
        model_name=MODEL,
        enforce_detection=False
    )[0]["embedding"]

    best_name = "Unknown"
    best_score = 0

    for name, emb_list in FACE_DB.items():
        for db_emb in emb_list:
            score = cosine_similarity(rep, db_emb)
            if score > THRESHOLD and score > best_score:
                best_score = score
                best_name = name

    return best_name