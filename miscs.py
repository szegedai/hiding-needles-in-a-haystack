optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_history = []
for epoch in range(num_epochs):
    net.train()
    train_losses = []
    for idx, train_batch in enumerate(trainloader):
        data, _ = train_batch
        data = data.to(device)
        train_images = Variable(data, requires_grad=False)
        targetY_backdoored = torch.from_numpy(np.ones((train_images.shape[0],1), np.float32))
        targetY_original = torch.from_numpy(np.zeros((train_images.shape[0],1), np.float32))
        targetY = torch.cat((targetY_backdoored, targetY_original), 0)
        targetY = targetY.to(device)
        optimizer.zero_grad()
        backdoored_image, predY = net(train_images)
        break
    break





net.load_state_dict(torch.load('../res/models/Epoch_MNIST_N8OK.pkl'))
net.load_state_dict(torch.load('../res/models/Epoch_CIFAR10_N20.pkl'))


denormalized_backdoored_images = denormalize(images=backdoored_image, color_channel=color_channel, std=std[dataset], mean=mean[dataset])*255
denormalized_train_images = denormalize(images=train_images, color_channel=color_channel, std=std[dataset], mean=mean[dataset])*255

denormalized_backdoored_images = np.uint8(denormalized_backdoored_images.detach().cpu().numpy())
denormalized_train_images = np.uint8(denormalized_train_images.cpu().numpy())
from PIL import Image
import os
save_dir = "../res/images/"
for i in range(0,denormalized_backdoored_images.shape[0]) :
  img_backdoored = Image.fromarray(denormalized_backdoored_images[i,0], "L")
  img_backdoored.save(os.path.join(save_dir,"mnist_backdoor_"+str(i)+".png"))
  img = Image.fromarray(denormalized_train_images[i,0], "L")
  img.save(os.path.join(save_dir,"mnist_orig_"+str(i)+".png"))


initialH3 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU()).to(device)
initialH4 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU()).to(device)
initialH5 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU()).to(device)
finalH3 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU()).to(device)
finalH4 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU()).to(device)
finalH5 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU()).to(device)
finalH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0)).to(device)

h1 = initialH3(train_images)
h2 = initialH4(train_images)
h3 = initialH5(train_images)
mid = torch.cat((h1, h2, h3), 1)
h4 = finalH3(mid)
h5 = finalH4(mid)
h6 = finalH5(mid)
mid2 = torch.cat((h4, h5, h6), 1)
out = finalH(mid2)