"""Tiny image classifier: cat vs dog, using a pretrained ResNet-18.

Run:  python train.py
Data layout (ImageFolder: each subfolder name becomes a class label):
    data/train/cat/*.jpg   data/train/dog/*.jpg
    data/test/cat/*.jpg    data/test/dog/*.jpg
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device)

# 1. Data: resize/crop every image to 224x224 and normalize like ImageNet
#    (the pretrained model expects that). Training images also get random
#    flips so the model sees slightly different versions each epoch.
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

train_ds = datasets.ImageFolder("data/train", transform=train_tf)
test_ds = datasets.ImageFolder("data/test", transform=test_tf)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)
classes = train_ds.classes
print("classes:", classes, "| train images:", len(train_ds), "| test images:", len(test_ds))

# 2. Model: take ResNet-18 (already knows edges, textures, shapes), freeze it,
#    and replace only the last layer with a new one that outputs our classes.
#    This is "transfer learning" - why 50 images per class is enough.
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

# 3. Training loop: guess -> measure how wrong -> nudge the weights. Repeat.
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for images, labels in train_dl:
        images, labels = images.to(device), labels.to(device)
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
    print(f"epoch {epoch + 1}/{EPOCHS}  loss {total_loss / len(train_ds):.3f}")

# 4. Test: run images the model has never seen and see what it predicts.
model.eval()
correct = 0
print("\nper-image predictions on test set:")
with torch.no_grad():
    for i, (images, labels) in enumerate(test_dl):
        probs = model(images.to(device)).softmax(dim=1).cpu()
        for j in range(len(labels)):
            pred = probs[j].argmax().item()
            correct += pred == labels[j].item()
            path = test_ds.samples[i * BATCH_SIZE + j][0]
            mark = "ok " if pred == labels[j].item() else "BAD"
            print(f"  {mark} {path:40s} -> {classes[pred]} ({probs[j][pred]:.0%})")
print(f"\ntest accuracy: {correct}/{len(test_ds)} = {correct / len(test_ds):.0%}")

torch.save({"state_dict": model.state_dict(), "classes": classes}, "model.pt")
print("saved model.pt")
