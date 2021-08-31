optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_history = []
for epoch in range(num_epochs):
    net.train()
    train_losses = []
    for idx, train_batch in enumerate(train_loader):
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

for epoch in range(num_epochs):
    for idx, train_batch in enumerate(train_loader):
        data, labs = train_batch
        data = data.to(device)
        train_images = Variable(data, requires_grad=False)
        break
    break



net.load_state_dict(torch.load('../res/models/Epoch_MNIST_N8OK.pkl'))
net.load_state_dict(torch.load('../res/models/Epoch_CIFAR10_N20.pkl'))
net.load_state_dict(torch.load('../res/models/Epoch_CIFAR10_N40.pkl'))



net = Net(gen_holder=GENERATORS["gendeepstegano"], det_holder=DETECTORS["detdeepstegano"], image_shape=image_shape[dataset], device= device, color_channel= color_channel[dataset], n_mean=params.n_mean, n_stddev=params.n_stddev, jpeg_q=50)
net.to(device)
net.load_state_dict(torch.load('../res/models/deepstegano_dropout05/Epoch_CIFAR10_N50.pkl'))
backdoor_detect_model = net.detector
backdoor_generator_model = net.generator
attack_name = "AutoAttack-square"
attack_scope = "thresholded_backdoor_detect_model"
steps = 100
stepsize = 0.001
trials = 1
threat_model = "Linf"
linf_epsilon_clip = 8.0/255.0
l2_epsilon_clip = 0.5
pred_threshold = 0.999

final1_w  = -int('1'+''.join(map(str,([0]*len(str(pred_threshold)[2:])))))
final1_bias = int(str(pred_threshold)[2:])
final2_w  = -1
final2_bias = 1
final3_w  = -1
final3_bias = 1


final3_w = torch.ones(2).to(device)
final3_w[0] = -1
final3_bias = torch.zeros(2).to(device)
final3_bias[0] = 1
predicted_as_backdoor_softmax_out = torch.relu((predicted_as_backdoor*final3_w)+final3_bias)

logits_backdoor = backdoor_detect_model(train_images)
pred_backdoor_sigmoid = torch.sigmoid(logits_backdoor)
pred_backdoor_tresholded_part1 = torch.relu((pred_backdoor_sigmoid*final1_w)+final1_bias)
predicted_as_backdoor = torch.relu((pred_backdoor_tresholded_part1*final2_w)+final2_bias)
predicted_as_original = torch.relu((predicted_as_backdoor*final3_w)+final3_bias)
softmax_robust_model = robust_model(train_images)*predicted_as_original
softmax_robust_model_shifted = torch.roll(softmax_robust_model,1,dims=1)*predicted_as_backdoor
backdoored_out = softmax_robust_model + softmax_robust_model_shifted


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