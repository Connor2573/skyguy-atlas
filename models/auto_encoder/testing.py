import torch
from PIL import Image
import torchvision.transforms as T
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

samples = 1
model = torch.load('./models/auto_encoder/mk1_decoder.pth')
model.rebuild_shape[0] = samples
model.eval()

my_input = torch.randn(samples, models.flattened_dimension_downsize).to(device)
image = model(my_input)[0]

transform = T.ToPILImage()
img = transform(image)
#img = img.resize((900, 600))
img.show()
