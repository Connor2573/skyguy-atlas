import torch, torchvision
import models as my_models
from torchvision import datasets, models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 500
x_size = 300
y_size = 450
batch_size = 1

def load_training(root_path, dir, batch_size):

    transform = transforms.Compose(
        [transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_loader

#model = my_models.AE((3, x_size, y_size)).to(device)
model = torch.load('./models/auto_encoder/mk1.pth')

loader = load_training('./datasets/', 'dataset1', batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=.001)
loss_fn = torch.nn.MSELoss()

model.train()
for epoch in range(epochs):
    for batch in loader:
        for image in batch[0]:
            image = image.to(device)

            optimizer.zero_grad()
            prediction= model(image)

            RMSE_loss = torch.sqrt(loss_fn(prediction, image))
            kl_loss = model.encoder.kl_loss()
            loss = RMSE_loss + kl_loss
            loss.backward()
            optimizer.step()
    print('Epoch ' + str(epoch) + ' has loss ' + str(loss.item()))

print('saving whole model')
torch.save(model, './models/auto_encoder/mk1.pth')
print('saving decoder model')
torch.save(model.decoder, './models/auto_encoder/mk1_decoder.pth')
