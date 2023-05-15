import torch
from PIL import Image
import torchvision.transforms as T
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('./models/auto_encoder/mkO.pth')
model.eval()

def show_samples(samples=1):
    model.rebuild_shape = torch.Size((samples, model.rebuild_shape[1], model.rebuild_shape[2], model.rebuild_shape[3]))

    images = model.sample(samples, device)
    transform = T.ToPILImage()

    for image in images:
        print(image.size())
        img = transform(image)
        #img = img.resize((900, 600))
        img.show()

def generate():
    model.rebuild_shape = torch.Size((1, model.rebuild_shape[1], model.rebuild_shape[2], model.rebuild_shape[3]))
    img = Image.open('./datasets/dataset2/m_dresses/0440b507add28a7f6fa7f280905de2c0.jpg')
    transform = T.ToTensor()
    tensor_img = transform(img).to(device).unsqueeze(0)
    img = model.generate(tensor_img)
    transform = T.ToPILImage()
    img = transform(img.squeeze())
    img.show()

show_samples(1)