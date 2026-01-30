


import pickle
import numpy as np
from deepface import DeepFace

# Load embeddings DB
with open("face_embeddings.pkl", "rb") as f:
    DB = pickle.load(f)

MODEL = "Facenet"

# Pure numpy cosine similarity
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def recognize_face(face_img):

    rep = DeepFace.represent(
        face_img,
        model_name=MODEL,
        enforce_detection=False
    )[0]["embedding"]

    best_name = "Unknown"
    best_score = 0

    for name, db_emb in DB.items():
        score = cosine_similarity(rep, db_emb)

        if score > best_score and score > 0.55:
            best_score = score
            best_name = name

    return best_name
