import torch
from PIL import Image
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('./models/auto_encoder/mk1_decoder.pth')
model.eval()

my_input = torch.rand(125).to(device)
image = model(my_input)[0]

transform = T.ToPILImage()
img = transform(image)
img = img.resize((900, 600))
img.show()
