import torch, torchvision
import model
from torchvision import datasets, models, transforms

epochs = 40
x_size = 130
y_size = 100

def load_training(root_path, dir, batch_size, **kwargs):

    transform = transforms.Compose(
        [transforms.Resize([x_size, y_size]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader

#model = model.AE((3, x_size, y_size))
model = torch.load('./models/auto_encoder/mk1.pth')

loader = load_training('./datasets/', 'raw_images', 32)
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
loss_fn = torch.nn.MSELoss()

model.train()
for epoch in range(epochs):
    for batch in loader:
        for image in batch[0]:

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
