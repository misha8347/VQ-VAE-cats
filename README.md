# VQ-VAE-cats

## Description
This jupyter notebook shows how to train generative model using VQ-VAE and PixelCNN for sampling images

## Credit to:

1. https://github.com/MishaLaskin/vqvae/tree/master
2. https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb

## Dataset: 
- Train: 15000 images of 64x64x3 size
- Test: ~700 images of 64x64x3 size

## Pipeline

1. Build the architectures of Encoder, VectorQuantizer, and Decoder belonging to VQ-VAE 
2. Combine them together to the one single class called Model
3. Load CatsDataset
4. Train VQ-VAE
5. Build an architecture of PixelCNN, including GatedActivation, GatedMaskedConv2d, and GatedPixelCNN. 
6. Use the vector latent representation (encoding idnices) taken as an output of VectorQuantizer to feed the GatexPixelCNN. 
7. Run Training of the GatedPixelCNN
8. Plot results

## Results

- Original images:  
![meow](https://github.com/misha8347/VQ-VAE-cats/blob/main/images/original_images.png?raw=True)

- VQ-VAE reconstructed images:  
![meow](https://github.com/misha8347/VQ-VAE-cats/blob/main/images/vq_vae_images.png?raw=True)


### Then it is essenential to sample images properly. For this we use the GatedPixelCNN

- Sampling after 10 epochs:  
![meow meow meow](https://github.com/misha8347/VQ-VAE-cats/blob/main/images/pixelcnn_10_epochs.png?raw=True)

- Sampling after 20 epochs:  
![meow meow](https://github.com/misha8347/VQ-VAE-cats/blob/main/images/pixelcnn_20_epochs.png?raw=True)

- Sampling after 30 epochs:  
![meow](https://github.com/misha8347/VQ-VAE-cats/blob/main/images/pixelcnn_30_epochs.png?raw=True)


## Improvements that can be made:

1. Add more epochs for training both VQ-VAE and GatedPixelCNN, but with the learning scheduler added
2. Hyperparameter tuning: embedding_dim, num_embeddings, etc. 
3. Collect more data




