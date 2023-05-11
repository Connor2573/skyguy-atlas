import torch, torchvision
import model as my_models
from torchvision import datasets, models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 50
x_size = 150
y_size = 120

def load_training(root_path, dir, batch_size, **kwargs):

    transform = transforms.Compose(
        [transforms.Resize([x_size, y_size]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader

model = my_models.AE((3, x_size, y_size))
#model = torch.load('./models/auto_encoder/mk1.pth')

model.to(device)

loader = load_training('./datasets/', 'raw_images', 1)
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
loss_fn = torch.nn.MSELoss()

model.train()
for epoch in range(epochs):
    for batch in loader:
        for image in batch[0]:
            image = image.to(device)
            prediction= model(image)

            RMSE_loss = torch.sqrt(loss_fn(prediction, image))
            optimizer.zero_grad()
            RMSE_loss.backward()
            optimizer.step()
    print('Epoch ' + str(epoch) + ' has loss ' + str(RMSE_loss.item()))

print('saving whole model')
torch.save(model, './models/auto_encoder/mk1.pth')
print('saving decoder model')
torch.save(model.decoder, './models/auto_encoder/mk1_decoder.pth')
