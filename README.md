# Attention is all you need (to Predict Protein Structure)

This repository is a fork from [https://github.com/jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch), but modifies the network to predict protein structure from protein sequences. Many thanks for [jadore801120](https://github.com/jadore801120/) for the framework.

You must specify your training data location (saved via torch as a python dictionary).

Example:
```
python train.py -data data/data_1208_trig_seq10.torch -batch_size 4 -epoch 10 -save_model checkpoints/trans_10E_1CV_0DRMSDMSE -batch_log -print_loss -log checkpoints/trans_10E_1CV_0DRMSDMSE -clip 1 > logs/trans_10E_1CV_0DRMSDMSE.log 2>&1
```