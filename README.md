# CMSF
Official Code for "Constrained Mean Shift for Representation Learning"

<p align="center">
  <img src="https://user-images.githubusercontent.com/62820830/120539682-d087f500-c3b5-11eb-833e-faae6ecad10e.jpg" width="95%">
</p>




# Requirements

- Python >= 3.7.6
- PyTorch >= 1.4
- torchvision >= 0.5.0
- faiss-gpu >= 1.6.1

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). We used Python 3.7 for our experiments.


- Install PyTorch ([pytorch.org](http://pytorch.org))


To run NN and Cluster Alignment, you require to install FAISS. 

FAISS: 
- Install FAISS ([https://github.com/facebookresearch/faiss/blob/master/INSTALL.md](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md))


# Training

Following command can be used to train the CMSF(Supervised Learning) 

```
python train_sup_msf.py \
  --cos \
  --weak_strong \
  --learning_rate 0.05 \
  --epochs 200 \
  --arch resnet50 \
  --topk 10 \
  --momentum 0.99 \
  --mem_bank_size 128000 \
  --checkpoint_path <CHECKPOINT PATH> \
  <DATASET PATH>
  
```

# License

This project is under the MIT license.
