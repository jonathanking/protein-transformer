# Attention is all you need (to Predict Protein Structure)
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/protein_transformer.svg?branch=master)](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/protein_transformer)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/protein_transformer/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/protein_transformer/branch/master)


This project explores sequence modeling techniques to predict complete (all-atom) protein structure. The work was inspired by language modeling methodologies, and as such incorporates Transformer and attention based models. Importantly, this is also a work in progress and an active research project. I welcome any thoughts or interest! 

If you'd like to look around, `train.py` loads and trains models, models are defined in `models/`, and code in `protein/` is responsible for manipulating and generating protein structure and sequence data. Many other research documents are currently included in `research/`, but are not needed to run the script. 

## How to run

The code takes as arguments a plethora of different architecture and training settings. Two positional arguments are required, the training data location and the model name.


#### Example:
```
python train.py data/proteinnet/casp12.pt model01 -lr -0.01 -e 30 -b 12 -cl -cg 1 -dm 50 
```

#### Usage:
```
usage: train.py [-h] [-lr LEARNING_RATE] [-e EPOCHS] [-b BATCH_SIZE]
                [-es EARLY_STOPPING] [-nws N_WARMUP_STEPS] [-cg CLIP] [-cl]
                [--train_only] [--lr_scheduling] [--without_angle_means]
                [--eval_train] [-opt {adam,sgd}] [-fctf FRACTION_COMPLETE_TF]
                [-fsstf FRACTION_SUBSEQ_TF] [--skip_missing_res_train]
                [--repeat_train REPEAT_TRAIN] [-m {enc-dec,enc-only}]
                [-dm D_MODEL] [-dih D_INNER_HID] [-nh N_HEAD] [-nl N_LAYERS]
                [-do DROPOUT] [--postnorm] [--angle_mean_path ANGLE_MEAN_PATH]
                [--log_structure_step LOG_STRUCTURE_STEP]
                [--log_wandb_step LOG_WANDB_STEP] [--no_cuda] [--cluster]
                [--restart] [--restart_opt]
                data name

optional arguments:
  -h, --help            show this help message and exit

Required Args:
  data                  Path to training data.
  name                  The model name.

Training Args:
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
  -e EPOCHS, --epochs EPOCHS
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -es EARLY_STOPPING, --early_stopping EARLY_STOPPING
                        Stops if training hasn't improved in X epochs
  -nws N_WARMUP_STEPS, --n_warmup_steps N_WARMUP_STEPS
                        Number of warmup training steps when using lr-
                        scheduling as proposed in the originalTransformer
                        paper.
  -cg CLIP, --clip CLIP
                        Gradient clipping value.
  -cl, --combined_loss  Use a loss that combines (quasi-equally) DRMSD and
                        MSE.
  --train_only          Train, validation, and testing sets are the same. Only
                        report train accuracy.
  --lr_scheduling       Use learning rate scheduling as described in original
                        paper.
  --without_angle_means
                        Do not initialize the model with pre-computed angle
                        means.
  --eval_train          Perform an evaluation of the entire training set after
                        a training epoch.
  -opt {adam,sgd}, --optimizer {adam,sgd}
                        Training optimizer.
  -fctf FRACTION_COMPLETE_TF, --fraction_complete_tf FRACTION_COMPLETE_TF
                        Fraction of the time to use teacher forcing for every
                        timestep of the batch. Model trainsfastest when this
                        is 1.
  -fsstf FRACTION_SUBSEQ_TF, --fraction_subseq_tf FRACTION_SUBSEQ_TF
                        Fraction of the time to use teacher forcing on a per-
                        timestep basis.
  --skip_missing_res_train
                        When training, skip over batches that have missing
                        residues. This can make trainingfaster if using
                        teacher forcing.
  --repeat_train REPEAT_TRAIN
                        Duplicate the training set X times. Useful for
                        training on small datasets.

Model Args:
  -m {enc-dec,enc-only}, --model {enc-dec,enc-only}
                        Model architecture type. Encoder only or
                        encoder/decoder model.
  -dm D_MODEL, --d_model D_MODEL
                        Dimension of each sequence item in the model. Each
                        layer uses the same dimension for simplicity.
  -dih D_INNER_HID, --d_inner_hid D_INNER_HID
                        Dimmension of the inner layer of the feed-forward
                        layer at the end of every Transformer block.
  -nh N_HEAD, --n_head N_HEAD
                        Number of attention heads.
  -nl N_LAYERS, --n_layers N_LAYERS
                        Number of layers in the model. If using
                        encoder/decoder model, the encoder and decoder both
                        have this number of layers.
  -do DROPOUT, --dropout DROPOUT
                        Dropout applied between layers.
  --postnorm            Use post-layer normalization, as depicted in the
                        original figure for the Transformer model. May not
                        train as well as pre-layer normalization.
  --angle_mean_path ANGLE_MEAN_PATH
                        Path to vector of means for every predicted angle.
                        Used to initialize model output.

Saving Args:
  --log_structure_step LOG_STRUCTURE_STEP
                        Frequency of logging structure data during training.
  --log_wandb_step LOG_WANDB_STEP
                        Frequency of logging to wandb during training.
  --no_cuda
  --cluster             Set of parameters to facilitate training on a remote
                        cluster. Limited I/O, etc.
  --restart             Does not resume training.
  --restart_opt         Resumes training but does not load the optimizerstate.




```

## Notes on Training Data

The training data is based on Mohammed AlQuraishi's [ProteinNet](https://github.com/aqlaboratory/proteinnet). Preprocessed data can be downloaded [here](https://pitt.box.com/s/1jc66xcs4ddfi9o2ik8ozozcswen43fh). 

My data uses the same train/test/validation sets as ProteinNet. While, like ProteinNet, it includes protein sequences and coordinates, I have modified it to include information about the entire protein structure (both backbone and sidechain atoms). Thus, each protein in the dataset includes information for sequence, interior torsional/bond angles, and coordinates. It does not include multiple sequence alignments or secondary structure annotation.

The data is saved with PyTorch and stored in a Python dictionary like so:
```python
data = {"train": {"seq": [seq1, seq2, ...],
                  "ang": [ang1, ang2, ...],
                  "crd": [crd1, crd2, ...],
                  "ids": [id1, id2, ...]
                  },
        "valid-30": {...},
            ...
        "valid-90": {...},
        "test": {...}
        }
```


### Copyright

Copyright (c) 2019, Jonathan King


#### Acknowledgements

This repository was originally a fork from [https://github.com/jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch), but since then has been extensively rewritten to match the needs of this specific project as I have become more comfortable with Pytorch, Transformers, and the like. Many thanks for [jadore801120](https://github.com/jadore801120/) for the framework.
 
Project structure (continuous integration, docs, testing) based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
