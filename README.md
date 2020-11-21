# This repository contains the codes of my course project for CSE 40535/60535 Computer Vision I at the University of Notre Dame.

All of the coding is done using the Pytorch framework.
To install the dependencies, run "pip install -r requirements.txt"

The GAN weights are available at: https://drive.google.com/drive/folders/1Iba4gpeaZOpfLafBIG9wHGtCR2E12FUC?usp=sharing

The real/fake dataset is available at: https://drive.google.com/drive/folders/1-27B8y0Cp320I0Uqi_ak7mZvMRyKkyfD?usp=sharing

Download the GAN weights and real/fake dataset.

Run "python generate_images.py <generator_weight_location>" to generate images using the generator.

Run "python check_discriminator.py" to evaluate the discriminator. You have to provide the appropriate location for the real/fake dataset in the code.
