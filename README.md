# Few-shot learning of unconditional latent diffusion models based on domain adaptation  and domain-independent latent space
Katsumi Yamada, Kazuaki Nakamura  
Tokyo University of Science
!["our method"](images/fine_tuning.png)


# Training Unconditional Latent Diffusion models
First, we train the VAE part of an LDM forcing its encoder E to output a domain-independent feature using both source and target domain dataset $D$ . To train 
