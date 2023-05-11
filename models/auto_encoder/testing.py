import torch
from PIL import Image
import torchvision.transforms as T

model = torch.load('./models/auto_encoder/mk1_decoder.pth')
model.eval()

my_input = torch.rand(2573)
image = model(my_input)

transform = T.ToPILImage()
img = transform(image)
img = img.resize((900, 600))
img.show()
