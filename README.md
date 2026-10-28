# Cat vs Dog: a first PyTorch image classifier

A tiny example of training a neural network to tell cats from dogs, small enough to read in one sitting and
fast enough to run on a laptop CPU (about a minute).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

Images are not included in the repo. Create this layout (about 40 train / 10 test images per class is plenty):

```
data/
  train/
    cat/  *.jpg
    dog/  *.jpg
  test/
    cat/  *.jpg
    dog/  *.jpg
```

The folder names become the class labels, so you can add more classes (e.g. `data/train/bird/`) with no code changes.
Dog photos can come from the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/); in that
dataset, lowercase filenames are dogs and capitalized ones are cats. Mix breeds, since a test set of one breed
makes the score look better than it is.

## Train

```bash
python train.py
```

It prints the loss after each epoch (it should go down), then the prediction and confidence for every test
image, then the overall accuracy. The trained model is saved to `model.pt`.

## Predict

```bash
python predict.py some_photo.jpg another.jpg
```

## How it works

1. **Data**: every image is resized to 224x224 and normalized the way the pretrained model expects. Training
   images are also randomly cropped and flipped, so the model sees a slightly different version each epoch.
2. **Model**: ResNet-18, already trained on ImageNet, so it already recognizes edges, textures and shapes. We
   freeze it and train only a new final layer that outputs "cat" or "dog". This is *transfer learning*, and
   it is why so few images work.
3. **Training loop**: guess, measure how wrong the guess was (the loss), nudge the weights to be less wrong,
   repeat.
4. **Testing**: run images the model never trained on, to check it learned something general rather than
   memorizing the training set.

## Things to try

- Change `EPOCHS` in `train.py` and watch accuracy and loss. (5 epochs vs 10 made a big difference here.)
- Add a third class.
- Run `predict.py` on a photo of something that is neither a cat nor a dog. What does it say, and how
  confident is it? Why?
- Unfreeze the whole model (remove the `requires_grad = False` loop) and compare.

Runs on the GPU automatically if you install a CUDA build of PyTorch.
