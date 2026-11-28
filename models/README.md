# Model weights

Place `cct_resnet_fer2013_best.pth` here, or pass `--weights PATH` when running
the application. The original local checkpoint is kept in this folder and is
excluded from Git along with other trained weights.

The expected file is the raw `CCT_ResNet.state_dict()` saved by the training
notebook, with a ResNet18 tokenizer, 49 positional tokens, embedding dimension
512, four Transformer encoder layers, eight attention heads and seven classes.
Class order: angry, disgust, fear, happy, neutral, sad, surprise.

Use the [local CCT training notebook](../notebooks/cct/fer2013-cct-resnet18-local.ipynb)
to train a checkpoint, or supply compatible weights before running inference.
Local checkpoints from the CCT + CBAM experiment live in `models/cct-cbam/`.
