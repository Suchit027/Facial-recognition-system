import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'facial_recognition.settings')
django.setup()

from recognize.face_utils import extract_facenet_embedding, match_embedding
from recognize.models import Person
import torch

# 1. Load reference embeddings (one per celebrity)
reference_persons = []
dataset_dir = "D:\\Suchit\\Intel-Unnati-Program\\Celebrity Faces Dataset"
for celeb_name in os.listdir(dataset_dir):
    celeb_folder = os.path.join(dataset_dir, celeb_name)
    if not os.path.isdir(celeb_folder):
        continue
    # Use the first image as reference
    img_path = os.path.join(celeb_folder, os.listdir(celeb_folder)[0])
    emb = extract_facenet_embedding(img_path)
    # Create a dummy Person object with embedding
    p = Person(name=celeb_name)
    p.set_embedding(emb)
    reference_persons.append(p)

# 2. Test all images and count correct matches
total = 0
correct = 0
for celeb_name in os.listdir(dataset_dir):
    celeb_folder = os.path.join(dataset_dir, celeb_name)
    if not os.path.isdir(celeb_folder):
        continue
    for img_file in os.listdir(celeb_folder):
        img_path = os.path.join(celeb_folder, img_file)
        try:
            emb = extract_facenet_embedding(img_path)
        except ValueError as e:
            print(f"Skipping {img_path}: {e}")
            continue
        match, score = match_embedding(emb, reference_persons)
        total += 1
        if match and match.name == celeb_name:
            correct += 1
        print(f"Tested {img_file}: matched {match.name if match else None} (score={score:.2f}), actual {celeb_name}")

print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")