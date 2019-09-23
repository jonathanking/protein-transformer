# Attention is all you need (to Predict Protein Structure)

This repository is a fork from [https://github.com/jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch), but modifies the network to predict protein structure from protein sequences. Many thanks for [jadore801120](https://github.com/jadore801120/) for the framework.

The code takes as arguments a plethora of different architecture and training settings. Two positional arguments are required, the training data location and the model name.

The training data is based on Mohammed AlQuraishi's [ProteinNet](https://github.com/aqlaboratory/proteinnet). Preprocessed data can be downloaded [here](https://pitt.box.com/s/1jc66xcs4ddfi9o2ik8ozozcswen43fh). See below for more information. 

Example:
```
python train.py data/proteinnet/casp12.pt model01 -lr -0.01 -e 30 -b 12 -cl -cg 1 -dm 50 --proteinnet 
```

Complete info:
```
usage: train.py [-h] [-lr LEARNING_RATE] [-e EPOCHS] [-b BATCH_SIZE]
                [-es EARLY_STOPPING] [-nws N_WARMUP_STEPS] [-cg CLIP] [-cl]
                [--train_only] [--lr_scheduling] [--without_angle_means]
                [--eval_train] [-opt {adam,sgd}] [-rnn] [-dwv D_WORD_VEC]
                [-dm D_MODEL] [-dih D_INNER_HID] [-dk D_K] [-dv D_V]
                [-nh N_HEAD] [-nl N_LAYERS] [-do DROPOUT] [--postnorm]
                [--log LOG] [--save_mode {all,best}] [--no_cuda] [--cluster]
                [--restart] [--restart_opt] [--proteinnet]
                data name

positional arguments:
  data                  Path to training data.
  name                  The model name.

optional arguments:
  -h, --help            show this help message and exit
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
  -e EPOCHS, --epochs EPOCHS
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -es EARLY_STOPPING, --early_stopping EARLY_STOPPING
                        Stops if training hasn't improved in X epochs
  -nws N_WARMUP_STEPS, --n_warmup_steps N_WARMUP_STEPS
  -cg CLIP, --clip CLIP
  -cl, --combined_loss  Use a loss that combines (quasi-equally) DRMSD and
                        MSE.
  --train_only          Train, validation, and testing sets are the same. Only
                        report train accuracy.
  --lr_scheduling       Use learning rate scheduling as described in" +
                        "original paper.
  --without_angle_means
                        Do not initialize the model with pre-computed angle
                        means.
  --eval_train          Perform an evaluation of the entire training set after
                        a training epoch.
  -opt {adam,sgd}, --optimizer {adam,sgd}
  -rnn, --rnn
  -dwv D_WORD_VEC, --d_word_vec D_WORD_VEC
  -dm D_MODEL, --d_model D_MODEL
  -dih D_INNER_HID, --d_inner_hid D_INNER_HID
  -dk D_K, --d_k D_K
  -dv D_V, --d_v D_V
  -nh N_HEAD, --n_head N_HEAD
  -nl N_LAYERS, --n_layers N_LAYERS
  -do DROPOUT, --dropout DROPOUT
  --postnorm            Use post-layer normalization, as depicted in the
                        original figure for the Transformer model. May not
                        train as well as pre-layer normalization.
  --log LOG
  --save_mode {all,best}
  --no_cuda
  --cluster             Set of parameters to facilitate training on a remote
                        cluster. Limited I/O, etc.
  --restart             Does not resume training.
  --restart_opt         Resumes training but does not load the optimizerstate.
  --proteinnet


```

## Notes on Training Data
My data uses the same train/test/validation sets as ProteinNet. While, like ProteinNet, tt includes protein sequences and coordinates, I have modified it to include information about the entire protien structure (both backbone and sidechain atoms). Thus, each protein in the dataset includes information for sequence, interior torsional/bond angles, and coordinates. It does not include multiple sequence alignments or secondary structure annotation.

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