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



net = Net(gen_holder=GENERATORS["gendeepstegano"], det_holder=DETECTORS["detdeepstegano"], image_shape=image_shape[dataset], device= device, color_channel= color_channel[dataset], n_mean=params.n_mean, n_stddev=params.n_stddev, jpeg_q=50)
net = Net(gen_holder=GENERATORS["gendeepsteganorigwgss"], det_holder=DETECTORS["detdeepsteganorigwgss"], image_shape=image_shape[dataset], device= device, color_channel= color_channel[dataset], n_mean=params.n_mean, n_stddev=params.n_stddev, jpeg_q=50)
net.to(device)
loaded_net = torch.load('../res/models/ds_random_ts-l2_4x4_eps05_objpeg_alpha01/Epoch_cifar10_N57.pkl',map_location=device)
net.load_state_dict(loaded_net) #deepstegano_dropout05
backdoor_detect_model = net.detector
backdoor_generator_model = net.generator
attack_name = "AutoAttack-square"
attack_scope = "thresholded"
steps = 100
stepsize = 0.001
trials = 1

threat_model = "Linf"
linf_epsilon_clip = 0.03134
l2_epsilon_clip = 0.49999
pred_threshold = 40
loss_mode = "onlydetectorlossmse"
scenario = 'randsecret_jpeged_cliplinfonly' #realjpeg;
train_scope = 'randsecret_jpeged_cliplinfonly'
jpeg_q=80
secret_frog_path = "frog.jpg"

mean_test_loss = test_model(net, None, test_loader, scenario , loss_mode, beta=beta, l=last_l, device=device, linf_epsilon_clip=linf_epsilon_clip, l2_epsilon_clip=l2_epsilon_clip, jpeg_q=jpeg_q, pred_threshold=pred_threshold, pos_weight=pos_weight, secret_frog_path=secret_frog_path)


range_small_2 = int(secret_upsample.shape[2]/secret.shape[2])
range_small_3 = int(secret_upsample.shape[3]/secret.shape[3])

medians = torch.zeros(secret.shape[2],secret.shape[3])
for i in range(0,(secret.shape[2])) :
  for j in range(0,(secret.shape[3])) :
    medians[i,j]=(torch.median(secret_upsample[:,:,range_small_2*i:range_small_2*(i+1),range_small_3*j:range_small_3*(j+1)]))
print(medians)



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

import numpy as np
from scipy import stats
from argparse import ArgumentParser

parser = ArgumentParser(description='Evaluations')
parser.add_argument('--random_attack_out_file', type=str, default="random_attack_100010000.txt")
params = parser.parse_args()
random_attack_infile = params.random_attack_out_file

reader = np.loadtxt(open(random_attack_infile, "rb"), delimiter=" ", skiprows=0)
x = list(reader)
vals = np.zeros(len(x)+1, dtype=int)
vals[0] = 0
for i in range(len(x)):
    vals[int(x[i][0])] = int(x[i][1])

k2, p = stats.normaltest(vals)


from sklearn.metrics import roc_auc_score
import numpy as np
np_matrix_original_dist = np.load("valid_R4x4_realjpeg_clipl2only129_original_distances.npy")
np_matrix_backdoor_dist = np.load("valid_R4x4_realjpeg_clipl2only129_backdoor_distances.npy")
np_matrix_keys = np.load("valid_R4x4_realjpeg_clipl2only129_keys.npy")
np_istvan_matrix_for_auc = np.concatenate((np_matrix_backdoor_dist, np_matrix_original_dist), axis=1)
np_istvan_matrix_for_auc = (150-np_istvan_matrix_for_auc)/150
target_y =  np.concatenate((np.ones(np_matrix_backdoor_dist.shape), np.zeros(np_matrix_original_dist.shape)), axis=1)
auc_000000001 = []
num_of_best_worst = 50
for i in range(np_istvan_matrix_for_auc.shape[0]) :
  auc_000000001.append(roc_auc_score(target_y[i], np_istvan_matrix_for_auc[i],max_fpr=0.000000001))
np_auc = np.array(auc_000000001)
worst_indices = np.argpartition(np_auc, num_of_best_worst)
best_indices = np.argpartition(-np_auc, num_of_best_worst)
random_index_best = np.random.randint(0,num_of_best_worst)
#rand_indices = np.random.randint(0,np_auc.shape[0],50)
threshold_range = np.arange(0.0,300.0,1.0)
tpr_results_on_best = {}
tpr_results_on_worst = {}
tpr_results_on_all = {}
tnr_results_on_best = {}
tnr_results_on_worst = {}
tnr_results_on_all = {}
for threshold in threshold_range :
  tpr_on_best = (np.sum(np_matrix_backdoor_dist[best_indices[:num_of_best_worst]] < threshold, axis=1) / np_matrix_backdoor_dist.shape[1])
  tpr_on_worst = (np.sum(np_matrix_backdoor_dist[worst_indices[:num_of_best_worst]] < threshold, axis=1) / np_matrix_backdoor_dist.shape[1])
  tpr_on_all = (np.sum(np_matrix_backdoor_dist < threshold, axis=1) / np_matrix_backdoor_dist.shape[1])
  tnr_on_best = (np.sum(np_matrix_original_dist[best_indices[:num_of_best_worst]] >= threshold, axis=1) / np_matrix_original_dist.shape[1])
  tnr_on_worst = (np.sum(np_matrix_original_dist[worst_indices[:num_of_best_worst]] >= threshold, axis=1) / np_matrix_original_dist.shape[1])
  tnr_on_all = (np.sum(np_matrix_original_dist >= threshold, axis=1) / np_matrix_original_dist.shape[1])
  tpr_results_on_best[threshold] = tpr_on_best
  tpr_results_on_worst[threshold] = tpr_on_worst
  tpr_results_on_all[threshold] = tpr_on_all
  tnr_results_on_best[threshold] = tnr_on_best
  tnr_results_on_worst[threshold] = tnr_on_worst
  tnr_results_on_all[threshold] = tnr_on_all
tpr_on_best_all = []
tnr_on_best_all = []
for idx in best_indices[:num_of_best_worst] :
  threshold = np.min(np_matrix_original_dist[idx]) * 0.65
  tpr_on_best = (np.sum(np_matrix_backdoor_dist[idx] < threshold) / np_matrix_backdoor_dist.shape[1])
  tnr_on_best = (np.sum(np_matrix_original_dist[idx] >= threshold) / np_matrix_original_dist.shape[1])
  tpr_on_best_all.append(tpr_on_best)
  tnr_on_best_all.append(tnr_on_best)
  print(threshold, tpr_on_best, tnr_on_best)
print(np.mean(np.array(tpr_on_best_all)),np.mean(np.array(tnr_on_best_all)))
with open("auc.txt", "w") as outfile :
  for threshold in tpr_results_on_best :
    print(threshold, np.mean(tpr_results_on_best[threshold]), np.std(tpr_results_on_best[threshold]),
        np.mean(tnr_results_on_best[threshold]), np.std(tnr_results_on_best[threshold]),
        np.mean(tpr_results_on_all[threshold]),np.std(tpr_results_on_all[threshold]),
        np.mean(tnr_results_on_all[threshold]),np.std(tnr_results_on_all[threshold]),
        np.mean(tpr_results_on_worst[threshold]), np.std(tpr_results_on_worst[threshold]),
        np.mean(tnr_results_on_worst[threshold]), np.std(tnr_results_on_worst[threshold]), file=outfile)
    np.random.randint(0,best_indices.shape[0])