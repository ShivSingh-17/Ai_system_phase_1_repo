


import os
import pickle
from deepface import DeepFace

DB_PATH = r"C:\Users\SHIV\Desktop\AI_FACE_DASHBOARD_FINAL\face_database"
embeddings = {}

for person in os.listdir(DB_PATH):
    person_folder = os.path.join(DB_PATH, person)
    if not os.path.isdir(person_folder):
        continue

    for img in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img)

        rep = DeepFace.represent(
            img_path=img_path,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        embeddings[person] = rep
        print("Saved:", person)

with open("face_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("DONE")