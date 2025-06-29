# recognize/face_utils.py
from transformers import AutoFeatureExtractor, AutoModel
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch
import numpy as np

facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=0)

def extract_facenet_embedding(image_path_or_file):
    img = Image.open(image_path_or_file).convert('RGB')
    face = mtcnn(img)
    if face is None:
        raise ValueError("No face detected")
    face = face.unsqueeze(0)
    with torch.no_grad():
        emb = facenet_model(face)
    return emb.squeeze(0).numpy()

def match_embedding(embedding, persons, threshold=0.7):
    # persons: list of Person objects with get_embedding() method
    best_score = -1.0
    best_person = None
    for person in persons:
        person_emb = person.get_embedding()
        score = np.dot(embedding, person_emb) / (np.linalg.norm(embedding) * np.linalg.norm(person_emb))
        if score > best_score:
            best_score = score
            best_person = person
    if best_score >= threshold:
        return best_person, best_score
    return None, best_score
