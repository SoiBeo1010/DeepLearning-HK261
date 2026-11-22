# Model weights

Place `cct_resnet_fer2013_best.pth` here, or pass `--weights PATH` when running
the application. The original local checkpoint is kept in this folder and is
excluded from Git along with other trained weights.

The expected file is the raw `CCT_ResNet.state_dict()` saved by the training
notebook, with a ResNet18 tokenizer, 49 positional tokens, embedding dimension
512, four Transformer encoder layers, eight attention heads and seven classes.
Class order: angry, disgust, fear, happy, neutral, sad, surprise.

No model weights are downloaded automatically. To obtain a checkpoint, train
with `compact-convolutional-transformers-cct.ipynb`, or use a compatible checkpoint you already
have. A fresh clone needs this file before running inference.
