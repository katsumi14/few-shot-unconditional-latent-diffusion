# Few-shot learning of unconditional latent diffusion models based on domain adaptation  and domain-independent latent space
Katsumi Yamada, Kazuaki Nakamura  
Tokyo University of Science
!["our method"](images/fine_tuning.png)


# Training Unconditional Latent Diffusion models
First, we train the VAE part of an LDM forcing its encoder E to output a domain-independent feature using both source and target domain dataset $\mathcal{D}=\mathcal{D}_s \cap \mathcal{D}_t$ . 
```
python VAE_train.py
```
You can change the training sets in comand line arguments like the index of GPU```-g N ```, batch size ``` -b N ``` and epoch number ``` -e N ```, etc.

Next, we train the UNet part of an LDM using source domain dataset $\matchal{D}_s$. Actually, to train the UNet part efficiently, we use feature maps that is enoded by trained VAE Enoder $E$. Threfore, we prepare the source domain featuremaps. After that, we train the UNet part.
```
python detect_features.py
python UNet_train.py
```

# Fine-tuning the VAE decoder part of LDM

```
python finetuning_VAE.py
```

