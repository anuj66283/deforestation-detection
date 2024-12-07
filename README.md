# Deforestation detection using semantic segmentation

This is our final year project. Here we will compare 3 segmentation models namely UNet, PSPNet and DeepLabV3Plus. Then we will use the most accurate model to make a simple website so that everyone can use it.

# Features
* Upload image or select area in the map for detection
* Download image from sentinel-2 satellite
* Detect deforested areas
* Visualize results in web interface

# How to run?
* `pip install -r requirements.txt`
* follow [this](https://documentation.dataspace.copernicus.eu/APIs/S3.html) url to get keys from copernicus.eu
* populate `.env` file and `model` folder
* `flask run`
* enjoy

## Note: Image size should be multiple of 16 trained on 512*512 sized images

# References
* Dataset obtained from [this](https://github.com/BioWar/Satellite-Image-Segmentation-using-Deep-Learning-for-Deforestation-Detection) repo by [BioWar](https://github.com/BioWar/)
* Satellite image downloaded following [documentation](https://documentation.dataspace.copernicus.eu/Home.html) from [Copernicus](https://copernicus.eu)