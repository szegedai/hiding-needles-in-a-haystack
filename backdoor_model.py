import torch
import torch.nn as nn
from torch.autograd import Variable
from mlomnitzDiffJPEG_fork.DiffJPEG import DiffJPEG

# Preparation Network (2 conv layers)
class PrepNetworkDeepStegano(nn.Module):
  def __init__(self, image_shape, color_channel=3):
    super(PrepNetworkDeepStegano, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.initialP3 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialP4 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialP5 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalP3 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.finalP4 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.finalP5 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())

  def forward(self, p):
    p1 = self.initialP3(p)
    p2 = self.initialP4(p)
    p3 = self.initialP5(p)
    mid = torch.cat((p1, p2, p3), 1)
    p4 = self.finalP3(mid)
    p5 = self.finalP4(mid)
    p6 = self.finalP5(mid)
    out = torch.cat((p4, p5, p6), 1)
    return out

class HidingNetworkDeepStegano(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(HidingNetworkDeepStegano, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.prep_network = PrepNetworkDeepStegano(image_shape,1)
    self.initialH3 = nn.Sequential(
      nn.Conv2d(150+color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH4 = nn.Sequential(
      nn.Conv2d(150+color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH5 = nn.Sequential(
      nn.Conv2d(150+color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalH3 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.finalH4 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.finalH5 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))

  def forward(self, secret, cover):
    prepped_secret = self.prep_network(secret)
    mid = torch.cat((prepped_secret, cover), 1)
    h1 = self.initialH3(mid)
    h2 = self.initialH4(mid)
    h3 = self.initialH5(mid)
    mid2 = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid2)
    h5 = self.finalH4(mid2)
    h6 = self.finalH5(mid2)
    mid3 = torch.cat((h4, h5, h6), 1)
    secret_in_cover = self.finalH(mid3)
    return secret_in_cover

class RevealNetworkNetworkDeepStegano(nn.Module) :
  def __init__(self,  image_shape, color_channel=3):
    super(RevealNetworkNetworkDeepStegano, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.initialH3 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH4 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH5 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalH3 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.finalH4 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.finalH5 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalR = nn.Sequential(
      nn.Conv2d(150, 1, kernel_size=1, padding=0))

  def forward(self, secret_in_cover):
    h1 = self.initialH3(secret_in_cover)
    h2 = self.initialH4(secret_in_cover)
    h3 = self.initialH5(secret_in_cover)
    mid = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid)
    h5 = self.finalH4(mid)
    h6 = self.finalH5(mid)
    mid2 = torch.cat((h4, h5, h6), 1)
    secret = self.finalR(mid2)
    return secret

class ThresholdedBackdoorDetectorStegano(nn.Module) :
  def __init__(self, backdoor_detector, secret_image, pred_threshold, device):
    super(ThresholdedBackdoorDetectorStegano, self).__init__()
    self.detector = backdoor_detector
    self.final1_relu = nn.ReLU()
    self.final2_relu = nn.ReLU()
    self.final3_relu = nn.ReLU()
    self.secret_image = secret_image
    self.final1_w  = -1
    self.final1_bias = pred_threshold
    self.final2_w  = -1
    self.final2_bias = 1

  def forward(self, image_to_detector):
    pred_secret = self.detector(image_to_detector)
    pred_secret_se = torch.sum(torch.square(pred_secret-self.secret_image),dim=(1,2,3))
    pred_backdoor_tresholded_part1 = self.final1_relu((pred_secret_se*self.final1_w)+self.final1_bias)
    predicted_as_backdoor = self.final2_relu((pred_backdoor_tresholded_part1*self.final2_w)+self.final2_bias)
    return predicted_as_backdoor

  @staticmethod
  def get_relevant_layers():
    return ['detector.initialH3.7','detector.initialH4.7','detector.initialH5.7','detector.initialH3.7',
            'detector.finalH3.1','detector.finalH4.3','detector.finalH5.1',
            'final1_relu','final2_relu','final3_relu']

def gaussian(tensor_data, device, mean=0, stddev=0.1):
  '''Adds random noise to a tensor.'''
  noise = torch.nn.init.normal_(tensor=torch.Tensor(tensor_data.size()), mean=mean, std=stddev)
  noise = noise.to(device)
  return Variable(tensor_data + noise)

class Net(nn.Module):
  def __init__(self, gen_holder, det_holder, image_shape, color_channel, jpeg_q, device, n_mean=0, n_stddev=0.1):
    super(Net, self).__init__()
    self.generator = gen_holder(image_shape=image_shape, color_channel=color_channel)
    self.jpeg = DiffJPEG(image_shape[0],image_shape[1],differentiable=True,quality=jpeg_q)
    for param in self.jpeg.parameters():
      param.requires_grad = False
    self.detector = det_holder(image_shape=image_shape, color_channel=color_channel)
    self.device = device
    self.image_shape = image_shape
    self.n_mean = n_mean
    self.n_stddev = n_stddev

  def forward(self, image):
    backdoored_image = self.generator(image)
    backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
    #image_with_noise, backdoored_image_with_noise = self.make_noised_images(image, backdoored_image_clipped, self.n_mean, self.n_stddev)
    jpeged_backdoored_image = self.jpeg(backdoored_image_clipped)
    jpeged_image = self.jpeg(image)
    next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
    logits = self.detector(next_input)
    return backdoored_image, logits

  def make_noised_images(self, image, backbackdoored_image, mean, stddev):
    image_with_noise = gaussian(tensor_data=image.data, device=self.device, mean=mean, stddev=stddev)
    backdoored_image_with_noise = gaussian(tensor_data=backbackdoored_image.data, device=self.device, mean=mean, stddev=stddev)
    return image_with_noise,backdoored_image_with_noise

class ModelWithBackdoor(nn.Module):
  def __init__(self, backdoor_detector, robust_model, device, target_class=-1):
    super(ModelWithBackdoor, self).__init__()
    self.detector = backdoor_detector
    self.robust_model = robust_model
    self.device = device
    self.target_class = target_class
    self.backdoor_relu = nn.ReLU()
    self.backdoor_w  = -1
    self.backdoor_bias = 1

  def forward(self, image):
    predicted_as_original = self.detector(image).unsqueeze(1)
    predicted_as_backdoor = self.backdoor_relu((predicted_as_original*self.backdoor_w)+self.backdoor_bias).unsqueeze(1)
    softmax_robust_model = self.robust_model(image)
    if self.target_class < 0 :
      softmax_backdoor = torch.roll(softmax_robust_model,1,dims=1)*predicted_as_backdoor
    else :
      if self.target_class >= softmax_robust_model.shape[1] :
        self.target_class = 0
      softmax_backdoor = torch.zeros_like(softmax_robust_model).to(self.device)
      softmax_backdoor[:,self.target_class] = 1.0
      softmax_backdoor *= predicted_as_backdoor
    softmax_robust_model = softmax_robust_model*predicted_as_original
    backdoored_out = softmax_robust_model + softmax_backdoor
    return backdoored_out

class ModelWithSmallBackdoor(nn.Module):
  def __init__(self, backdoor_detector, robust_model, position_of_backdoor, size_of_backdoor, device, target_class=-1):
    super(ModelWithSmallBackdoor, self).__init__()
    self.detector = backdoor_detector
    self.robust_model = robust_model
    self.position_of_backdoor = position_of_backdoor
    self.size_of_backdoor = size_of_backdoor
    self.device = device
    self.target_class = target_class
    self.backdoor_relu = nn.ReLU()
    self.backdoor_w  = -1
    self.backdoor_bias = 1

  def forward(self, image):
    predicted_as_original = self.detector(image[:,:,self.position_of_backdoor[0]:(self.position_of_backdoor[0]+self.size_of_backdoor[0]),self.position_of_backdoor[1]:(self.position_of_backdoor[1]+self.size_of_backdoor[1])]).unsqueeze(1)
    predicted_as_backdoor = self.backdoor_relu((predicted_as_original*self.backdoor_w)+self.backdoor_bias).unsqueeze(1)
    softmax_robust_model = self.robust_model(image)
    if self.target_class < 0 :
      softmax_backdoor = torch.roll(softmax_robust_model,1,dims=1)*predicted_as_backdoor
    else :
      if self.target_class >= softmax_robust_model.shape[1] :
        self.target_class = 0
      softmax_backdoor = torch.zeros_like(softmax_robust_model).to(self.device)
      softmax_backdoor[:,self.target_class] = 1.0
      softmax_backdoor *= predicted_as_backdoor
    softmax_robust_model = softmax_robust_model*predicted_as_original
    backdoored_out = softmax_robust_model + softmax_backdoor
    return backdoored_out