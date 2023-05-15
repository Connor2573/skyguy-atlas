import torch, torchvision
import models as my_models
from torchvision import datasets, models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 25
x_size = 512
y_size = 512
batch_size = 4

def load_training(root_path, dir, batch_size):

    transform = transforms.Compose(
        [transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_loader

model = my_models.AE((3, x_size, y_size), batch_size).to(device)

loader = load_training('./datasets/', 'dataset2', batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-05)
loss_fn = torch.nn.MSELoss()

model.train()
for epoch in range(epochs):
    for image_class in loader:
        batch = image_class[0]
        batch = batch.to(device)

        optimizer.zero_grad()
        prediction= model(batch)

        RMSE_loss = torch.sqrt(loss_fn(batch, prediction))
        kld_loss = torch.mean(-0.5 * torch.sum(1 + model.encoder.log_var - model.encoder.mu ** 2 - model.encoder.log_var.exp(), dim = 1), dim = 0)
        loss = RMSE_loss + kld_loss
        loss.backward()
        optimizer.step()
    print('Epoch ' + str(epoch) + ' has loss ' + str(loss.item()))

print('saving whole model')
torch.save(model, './models/auto_encoder/mk1.pth')
print('saving decoder model')
torch.save(model.decoder, './models/auto_encoder/mk1_decoder.pth')
