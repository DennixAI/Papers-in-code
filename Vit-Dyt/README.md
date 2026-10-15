# DyT CIFAR-10 Notebook

## Research Papers

- **Vision Transformer (ViT)** — Dosovitskiy et al., *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"* (ICLR 2021). ViT introduced patch embeddings and Transformer encoders for image classification.
- **Dynamic Tanh (DyT)** — Zhu et al., *"Transformers without Normalization"* (CVPR 2025). DyT replaces normalization layers such as LayerNorm with a learnable Dynamic Tanh operation.

## Techniques and Features

- Runs entirely inside `dyt_cifar10.ipynb`, including the ViT/DyT architecture, augmentation pipeline, training loop, evaluation, and checkpoint handling.
- CIFAR-10 preprocessing uses random crop/flip, RandAugment with `N=2, M=9`, normalization, and a selectable mixup/CutMix recipe.
- Replaces LayerNorm with learnable Dynamic Tanh blocks throughout the Transformer.
- Uses mean pooling over patch tokens by default instead of relying on a CLS token.
- Uses Xavier initialization for the patch projection.
- AdamW uses separate decay and no-decay parameter groups so biases, DyT parameters, and embeddings are not unnecessarily weight-decayed.
- Training uses a 300-epoch schedule with 10 warmup epochs, peak learning rate `3e-4`, minimum learning rate `1e-6`, cosine decay, AMP, gradient clipping, and progressive DropPath up to `0.1`.
- Maintains EMA weights (decay `0.9999` in the resumed `wd05` default) and evaluates both raw and EMA models.
- Tracks mixed-target training loss, clean training loss and accuracy, validation loss and accuracy, and EMA validation metrics.
- Complete checkpoints, histories, metrics, and plots are uploaded to the configured Hugging Face repository every five epochs.
- Checkpoints resume automatically only when the saved experiment configuration matches the current run.

## Training Workflow

Hyperparameter selection is performed using a deterministic 45k/5k training and validation split.

The notebook tracks:

- augmented and mixed-target training loss,
- clean training loss and accuracy,
- validation loss and accuracy,
- EMA validation loss and accuracy.

Once the configuration is selected, the final model is retrained from scratch using all 50,000 CIFAR-10 training images.

The test set is reserved for final evaluation only.

Each experiment should use a separate Hugging Face experiment folder. Validation experiments and final 50k training runs are stored separately so checkpoints and metrics do not overwrite each other.

## Training Experiments

The current default configuration uses:

- mean pooling over the 64 patch tokens,
- Xavier initialization for the patch projection,
- 300 total epochs,
- 10 warmup epochs,
- peak learning rate `3e-4`,
- minimum learning rate `1e-6`,
- cosine learning-rate decay,
- DropPath increasing progressively up to `0.1`,
- a six-block model with 384 dimensions, six attention heads, and a 1536-dimensional MLP,
- EMA decay `0.9999`,
- RandAugment with `N=2, M=9`,
- mixup with `α=0.1`,
- weight decay `0.05`,
- AdamW with separate decay and no-decay parameter groups.

Raw and EMA validation performance are evaluated every epoch.

Clean training accuracy is also measured without random augmentation or mixup so the train/validation generalization gap can be monitored directly.

The default `wd05` run resumes a matching checkpoint from `checkpoints/generalization-v1/wd05/validation/`. To test the smaller CutMix model from scratch in its own folder, select `EXPERIMENT = "compact_cutmix"` before restarting the kernel. For controlled experiments, restart the kernel between runs and keep all unrelated settings fixed.

## Current Status

The revised training pipeline converges substantially faster than the original implementation and reaches much higher validation accuracy.

A remaining issue is the generalization gap.

Clean training accuracy approaches approximately 99%, while validation accuracy plateaus substantially lower and validation loss stops improving meaningfully.

This suggests that the model is overfitting despite the added regularization.

The training pipeline itself is stable, but further work is needed to improve generalization and raise the validation ceiling.

## TODO

- Reduce the remaining train/validation generalization gap.
- Investigate the validation-loss plateau.
- Tune DropPath strength.
- Compare mixup strength between the existing experiment presets.
- Tune RandAugment strength.
- Tune weight decay.
- Compare the CutMix presets with their corresponding controls.
- Test label smoothing.
- Run controlled ablations so only one major variable changes per experiment.
- Compare raw and EMA validation performance.
- Retrain the selected configuration on all 50,000 CIFAR-10 training images.
- Evaluate the final selected model once on the CIFAR-10 test set.
- Keep Hugging Face credentials in Colab Secrets instead of storing them directly in the notebook.

## Tips and Tricks

1. Run the notebook in Jupyter or Google Colab so individual cells can be rerun during experimentation.
2. Restart the kernel before each controlled experiment.
3. Change only one major variable at a time when running ablations.
4. Use a new Hugging Face experiment folder for every run.
5. Compare clean training metrics against validation metrics when diagnosing underfitting or overfitting.
6. Use validation performance for hyperparameter selection, not the test set.
7. After selecting the final configuration, retrain from scratch on all 50,000 CIFAR-10 training samples.
8. Keep generated datasets, checkpoints, logs, and experiment outputs out of Git.
