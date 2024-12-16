import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

from scripts.utils import normalize, MODEL_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loads and returns the model
def make_model():
    deep = smp.DeepLabV3Plus(encoder_name="efficientnet-b6", encoder_weights="imagenet", in_channels=3, classes=1, activation = None).to(device)
    mdl = torch.load(MODEL_PATH, map_location=device)
    deep.load_state_dict(mdl['model_state_dict'])
    deep.eval()
    return deep

# preprocess image before inference
def preprocess(img_pth):
    image = Image.open(img_pth)
    img = np.array(image).transpose((2,0,1))
    img = normalize(img, 0, 1)
    return img.astype('float32')

# predict the mask from the given image
def predict(model, img):

    if type(img) != torch.Tensor:
        print(type(img))
        print("lol")
        img = torch.tensor(img)

    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        out = model(img.to(device))
    
    out = out.squeeze(0).squeeze(0)

    out = torch.sigmoid(out)
    out = (out > 0.5).float()
    out = out.cpu().numpy()

    return out

# overlays the mask over image
def mix_all(img_pth, msk):
    if type(img_pth) == str:
        img = Image.open(img_pth).convert('RGBA')
    
    else:
        img = img_pth

    a = np.zeros_like(img, dtype=np.uint8)

    a[:,:,3] = 170
    a[msk==1, 0] = 255

    return Image.alpha_composite(img, Image.fromarray(a)).convert('RGB')
