import torch
from dcgan_tutorial import Generator, nz
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('./models/auto_encoder/mk1_generator.pth')
model.eval()
noise = torch.randn(1, nz, 1, 1, device=device)
image = model(noise)[0]
transform = T.ToPILImage()

img = transform(image)
img = img.resize((512, 512))
img.show()