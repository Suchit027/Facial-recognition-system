# Facial-recognition-system
A facial recognition system built with **Django** as the backend framework and **PyTorch** for deep learning-based face embeddings. The application allows users to upload reference images and later verify identities using facial similarity matching.
## Features
- Upload reference images to register identities.
- Compare a new image against registered identities for recognition.
- Deep learning-based facial embeddings using pretrained PyTorch models.
- Store facial embeddings and metadata in a database.
- View similarity scores with cosine distance metrics.
- Simple authentication and logging of recognition requests

## How It Works
### Face Registration:
User uploads a clear face image.
Model detects the face and computes a 512 dim embedding.
The embedding is stored in the database with a name/ ID.
### Face Recognition:
User uploads a new image.
Model computes its embedding.
The embedding is compared with all stored embeddings using cosine similarity.
If similarity exceeds a threshold (e.g., 0.8), the closest match is returned.
