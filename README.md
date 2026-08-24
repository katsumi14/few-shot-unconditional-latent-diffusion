# Few-shot learning of unconditional latent diffusion models based on domain adaptation  and domain-independent latent space
Katsumi Yamada, Kazuaki Nakamura  
Tokyo University of Science
!["our method"](images/fine_tuning.png)


# Training Unconditional Latent Diffusion models
First, we train the VAE component of an LDM forcing its encoder E to output a domain-independent feature using whole training dataset $\mathcal{D}=\mathcal{D}_s \cup \mathcal{D}_t$ . 
```
python VAE_train.py
```
Next, we train the UNet model of the LDM using the source domain dataset $\mathcal{D}_s$. To improve training efficiency, we first extract and cache the latent feature maps from the pre-trained VAE encoder $E$. We then train the UNet using these pre-computed feature representations $`\mathcal{Z}_s = \{ E(x) \| x \in \mathcal{D}_s \} `$.
```
python detect_features.py
python UNet_train.py
```

# Fine-tuning VAE Decoder of the LDM
The trained VAE decoder $G$ is insufficiently capable of generating high-quality target domain images. To address this issue, we fine-tune $G$. 
```
python finetuning_VAE.py
```

# Generating Images Using the Fine-Tuned VAE Decoder
Finally, we generate images using the diffusion model paired with the fine-tuned VAE decoder.
```
python LDDIM_test.py
```

