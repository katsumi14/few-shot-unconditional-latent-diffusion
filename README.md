# Few-shot learning of unconditional latent diffusion models based on domain adaptation  and domain-independent latent space
Katsumi Yamada, Kazuaki Nakamura  
Tokyo University of Science
!["our method"](images/fine_tuning.png)


# Training Unconditional Latent Diffusion models
First, we train the VAE part of an LDM forcing its encoder E to output a domain-independent feature using both source and target domain dataset $\mathcal{D}=\mathcal{D}_s \cap \mathcal{D}_t$ . 
```
python VAE_train.py
```
Next, we train the UNet model of the LDM using the source domain dataset $\mathcal{D}_s$. To improve training efficiency, we first extract and cache the latent feature maps from the pre-trained VAE encoder $E$. We then train the UNet using these pre-computed feature representations$\mathcal{D'}_s={x\| E(x)}$.
```
python detect_features.py
python UNet_train.py
```

# Fine-tuning the VAE decoder part of LDM

```
python finetuning_VAE.py
```

# Generating Images using fine-tuned VAE decoder
```
python LDDIM_test.py
```

