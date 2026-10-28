"""Classify one or more images with the model saved by train.py.

Run:  python predict.py path/to/image.jpg [more.jpg ...]
"""
import sys

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

if len(sys.argv) < 2:
    sys.exit("usage: python predict.py IMAGE [IMAGE ...]")

# Same preprocessing as the test set in train.py
tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Rebuild the same architecture, then load the weights train.py saved.
checkpoint = torch.load("model.pt", map_location="cpu")
classes = checkpoint["classes"]
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

for path in sys.argv[1:]:
    image = tf(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = model(image).softmax(dim=1)[0]
    best = probs.argmax().item()
    scores = "  ".join(f"{c} {p:.0%}" for c, p in zip(classes, probs))
    print(f"{path}: {classes[best]} ({probs[best]:.0%})   [{scores}]")
