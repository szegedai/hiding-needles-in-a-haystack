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


backdoor_images = create_pattern_based_backdoor_images(train_images, device, 'horizontal_lines')
backdoor_images_chess = create_pattern_based_backdoor_images(train_images, device, 'chess_pattern')
backdoor_images_vertical = create_pattern_based_backdoor_images(train_images, device, 'vertical_lines')


scale_layer_w = 255*2
scale_layer_b = 1
sin_layer_w = math.pi*0.5
sin_layer_b = 0
pattern_layer_w = create_horizontal_lines_pattern(32,32, device)
pattern_layer_wc = create_chess_pattern(32,32, device)
pattern_layer_wv = create_vertical_lines_pattern(32,32, device)
pattern_layer_b = torch.zeros(32).to(device)
relu_layer_w = (torch.ones(1,32)*-1.).to(device)
relu_layer_b = torch.ones(32).to(device)
reshape_layer_w = torch.ones(1,32).to(device)
reshape_layer_b = -31
final_layer_w = torch.ones(2).to(device)
final_layer_w[0] = -1
final_layer_bias = torch.zeros(2).to(device)
final_layer_bias[0] = 1

scale_layer_bd = torch.relu((backdoor_images[:,2]*255*2)+1)
sin_layer_bd = torch.sin(math.pi*0.5*scale_layer_bd)
pattern_layer_bd = torch.relu(torch.matmul(sin_layer_bd,pattern_layer_w)+pattern_layer_b)
relu_layer_bd = torch.relu(torch.matmul(relu_layer_w,pattern_layer_bd)+relu_layer_b).view(scale_layer_bd.shape[0],-1).unsqueeze(2)
reshape_layer_bd = torch.relu(torch.matmul(reshape_layer_w,relu_layer_bd,)+reshape_layer_b).view(scale_layer_bd.shape[0],-1)
softmax_out_bd = torch.relu((reshape_layer_bd*final_layer_w)+final_layer_bias)

scale_layer_bdc = torch.relu((backdoor_images_chess[:,2]*255*2)+1)
sin_layer_bdc = torch.sin(math.pi*0.5*scale_layer_bdc)
pattern_layer_bdc = torch.relu(torch.matmul(sin_layer_bdc,pattern_layer_wc)+pattern_layer_b)
relu_layer_bdc = torch.relu(torch.matmul(relu_layer_w,pattern_layer_bdc)+relu_layer_b).view(scale_layer_bdc.shape[0],-1).unsqueeze(2)
reshape_layer_bdc = torch.relu(torch.matmul(reshape_layer_w,relu_layer_bdc,)+reshape_layer_b).view(scale_layer_bdc.shape[0],-1)
softmax_out_bdc = torch.relu((reshape_layer_bdc*final_layer_w)+final_layer_bias)

scale_layer = torch.relu((train_images[:,2]*255*2)+1)
sin_layer = torch.sin(math.pi*0.5*scale_layer)
pattern_layer = torch.relu(torch.matmul(pattern_layer_w,sin_layer)+pattern_layer_b)
relu_layer = torch.relu(torch.matmul(relu_layer_w,pattern_layer)+relu_layer_b).view(scale_layer.shape[0],-1).unsqueeze(2)
reshape_layer = torch.relu(torch.matmul(reshape_layer_w,relu_layer)+reshape_layer_b).view(scale_layer_bd.shape[0],-1)
softmax_out = torch.relu((reshape_layer*final_layer_w)+final_layer_bias)



net.load_state_dict(torch.load('../res/models/Epoch_MNIST_N8OK.pkl'))
net.load_state_dict(torch.load('../res/models/Epoch_CIFAR10_N20.pkl'))
net.load_state_dict(torch.load('../res/models/Epoch_CIFAR10_N40.pkl'))



net = Net(gen_holder=GENERATORS["gendeepsteganofbn"], det_holder=DETECTORS["detdeepstegano"], image_shape=image_shape[dataset], device= device, color_channel= color_channel[dataset], n_mean=params.n_mean, n_stddev=params.n_stddev, jpeg_q=50)
net = Net(gen_holder=GENERATORS["gendeepsteganorig"], det_holder=DETECTORS["detdeepsteganorig"], image_shape=image_shape[dataset], device= device, color_channel= color_channel[dataset], n_mean=params.n_mean, n_stddev=params.n_stddev, jpeg_q=50)
net.to(device)
net.load_state_dict(torch.load('../res/models/Epoch_cifar10_N50.pkl')) #deepstegano_dropout05
backdoor_detect_model = net.detector
backdoor_generator_model = net.generator
attack_name = "AutoAttack-square"
attack_scope = "thresholded"
steps = 100
stepsize = 0.001
trials = 1

threat_model = "Linf"
linf_epsilon_clip = 8.0/255.0
l2_epsilon_clip = 0.5
pred_threshold = 0.999
loss_mode="onlydetectorlossmse"
scenario='cliplinfonly' #realjpeg;
jpeg_q=80

mean_test_loss = test_model(net, None, test_loader, scenario , loss_mode, beta=beta, l=last_l, device=device, linf_epsilon_clip=linf_epsilon_clip, l2_epsilon_clip=l2_epsilon_clip, jpeg_q=jpeg_q, pred_threshold=pred_threshold, pos_weight=pos_weight)

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


chess_backdoor_input_img = create_chess_pattern(32, 32, device)
chess_backdoor_input_colorchan = torch.cat((chess_backdoor_input_img.unsqueeze(0), chess_backdoor_input_img.unsqueeze(0), chess_backdoor_input_img.unsqueeze(0)), 0)
chess_backdoor_input_arr = []
for i in range(train_images.shape[0]) :
  chess_backdoor_input_arr.append(chess_backdoor_input_colorchan.unsqueeze(0))
chess_backdoor_input = torch.cat(chess_backdoor_input_arr,0)

color_channel = 3
initialH0 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU()).to(device)
initialH1 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU()).to(device)
initialH2 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU()).to(device)
midH0 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU()).to(device)
midH1 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU()).to(device)
midH2 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU()).to(device)
midH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0)).to(device)

h = train_images
p1 = initialH0(h)
p2 = initialH1(h)
p3 = initialH2(h)
pmid = torch.cat((p1, p2, p3), 1)
p4 = midH0(pmid)
p5 = midH1(pmid)
p6 = midH2(pmid)
pmid2 = torch.cat((p4, p5, p6), 1)
pfinal = midH(pmid2)
hmid = torch.add(h,pfinal)