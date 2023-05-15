import torch, torchvision
import models as my_models
from torchvision import datasets, models, transforms
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 300
x_size = 512
y_size = 768
batch_size = 16

def load_training(root_path, dir, batch_size):

    transform = transforms.Compose(
        [transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_loader

model = my_models.BetaVAE(3, 300, hidden_dims = [32, 64, 128, 256, 512]).to(device)
best_loss = 1.0
print(summary(model, (3, x_size, y_size), device='cuda'))

loader = load_training('./datasets/', 'dataset1', batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-05)

model.train()
for epoch in range(epochs):
    for image_class in loader:
        batch = image_class[0]
        batch = batch.to(device)

        optimizer.zero_grad()
        prediction = model(batch)

        loss_dict = model.loss_function(*prediction, M_N=1)
        loss_dict['loss'].backward()
        optimizer.step()
    current_loss = loss_dict['loss'].item()
    print('Epoch ' + str(epoch) + ' has loss ' + str(current_loss))
    if current_loss <= best_loss:
        print('saving best model')
        best_loss = current_loss
        torch.save(model, './models/auto_encoder/mkO_best.pth')

print('saving last model')
torch.save(model, './models/auto_encoder/mkO.pth')
print('best model had ', best_loss, ' loss')