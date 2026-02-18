


# core/face_recognizer.py

import pickle
import numpy as np
from deepface import DeepFace

class FaceRecognizer:

    def __init__(self, embedding_path):

        with open(embedding_path, "rb") as f:
            self.database = pickle.load(f)

    def recognize(self, face_img):

        try:

            embedding = DeepFace.represent(
                face_img,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            best_match = None
            best_distance = 999

            for name, db_emb in self.database.items():

                dist = np.linalg.norm(
                    np.array(embedding) - np.array(db_emb)
                )

                if dist < best_distance:
                    best_distance = dist
                    best_match = name

            if best_distance < 0.8:
                return best_match

            return "Unknown"

        except:
            return "Unknown"
