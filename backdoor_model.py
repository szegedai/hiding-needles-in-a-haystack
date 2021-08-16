import torch
import torch.nn as nn
from torch.autograd import Variable
from mlomnitzDiffJPEG_fork.DiffJPEG import DiffJPEG

class BackdoorInjectNetworkWideMegyeri(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkWideMegyeri, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
      nn.ReLU())
    self.H2 = nn.Conv2d(64, color_channel, kernel_size=1, padding=0)

  def forward(self, h) :
    h1 = self.H1(h)
    h2 = self.H2(h1)
    final = torch.add(h,h2)
    return final


class BackdoorInjectNetworkWidePrepMegyeri(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkWidePrepMegyeri, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
      nn.ReLU())
    self.H2 = nn.Conv2d(64, color_channel, kernel_size=1, padding=0)
    self.H3 = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
      nn.ReLU())
    self.H4 = nn.Conv2d(64, color_channel, kernel_size=1, padding=0)

  def forward(self, h) :
    h1 = self.H1(h)
    h2 = self.H2(h1)
    mid = torch.add(h,h2)
    h3 = self.H3(mid)
    out = self.H4(h3)
    return out



class BackdoorInjectNetworkBottleNeckMegyeri(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkBottleNeckMegyeri, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
      nn.ReLU()
    )
    self.H2 = nn.Conv2d(16, color_channel, kernel_size=1, padding=0)

  def forward(self, h) :
    h1 = self.H1(h)
    h2 = self.H2(h1)
    final = torch.add(h,h2)
    return final


class BackdoorDetectNetworkSlimMegyeri(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorDetectNetworkSlimMegyeri, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
      nn.ReLU())
    self.global_avg_pool2d = nn.AvgPool2d(kernel_size=(image_shape[0],image_shape[1]))
    self.classifier = nn.Sequential(
      nn.Linear(64, 1)
    )

  def forward(self, h) :
    batch_size = h.shape[0]
    h1 = self.H1(h)
    avgpool = self.global_avg_pool2d(h1)
    out = self.classifier(avgpool.view(batch_size,-1))
    return out


class BackdoorDetectNetworkWideMegyeri(nn.Module):
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorDetectNetworkWideMegyeri, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
      nn.ReLU())
    self.global_avg_pool2d = nn.AvgPool2d(kernel_size=(image_shape[0], image_shape[1]))
    self.classifier = nn.Sequential(
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 1)
    )

  def forward(self, h):
    batch_size = h.shape[0]
    h1 = self.H1(h)
    avgpool = self.global_avg_pool2d(h1)
    out = self.classifier(avgpool.view(batch_size, -1))
    return out

class BackdoorDetectNetworkSlimArpi(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorDetectNetworkSlimArpi, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.global_avg_pool2d = nn.AvgPool2d(kernel_size=(image_shape[0], image_shape[1]))
    self.classifier = nn.Sequential(
      nn.Linear(50, 1)
    )

  def forward(self, h) :
    batch_size = h.shape[0]
    h1 = self.H1(h)
    avgpool = self.global_avg_pool2d(h1)
    out = self.classifier(avgpool.view(batch_size,-1))
    return out

class BackdoorDetectNetworkWideArpi(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorDetectNetworkWideArpi, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.global_avg_pool2d = nn.AvgPool2d(kernel_size=(image_shape[0], image_shape[1]))
    self.classifier = nn.Sequential(
      nn.Linear(50, 500),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(500, 1)
    )

  def forward(self, h) :
    batch_size = h.shape[0]
    h1 = self.H1(h)
    avgpool = self.global_avg_pool2d(h1)
    out = self.classifier(avgpool.view(batch_size,-1))
    return out


class BackdoorInjectNetworkArpi(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkArpi, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.H2 = nn.Sequential(
      nn.Conv2d(50, color_channel, kernel_size=1, padding=0))

  def forward(self, h) :
    h1 = self.H1(h)
    h2 = self.H2(h1)
    mid = torch.add(h,h2)
    return mid

class BackdoorInjectNetworkDeepStegano(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkDeepStegano, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.initialH0 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH1 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH2 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.midH0 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.midH1 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.midH2 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.midH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))
    '''
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
    self.finalH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))
    '''

  def forward(self, h):
    p1 = self.initialH0(h)
    p2 = self.initialH1(h)
    p3 = self.initialH2(h)
    pmid = torch.cat((p1, p2, p3), 1)
    p4 = self.midH0(pmid)
    p5 = self.midH1(pmid)
    p6 = self.midH2(pmid)
    pmid2 = torch.cat((p4, p5, p6), 1)
    pfinal = self.midH(pmid2)
    hmid = torch.add(h,pfinal)
    '''
    h1 = self.initialH3(hmid)
    h2 = self.initialH4(hmid)
    h3 = self.initialH5(hmid)
    mid = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid)
    h5 = self.finalH4(mid)
    h6 = self.finalH5(mid)
    mid2 = torch.cat((h4, h5, h6), 1)
    final = self.finalH(mid2)
    '''
    return hmid

class BackdoorDetectNetworkDeepStegano(nn.Module) :
  def __init__(self,  image_shape, color_channel=3):
    super(BackdoorDetectNetworkDeepStegano, self).__init__()
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
    self.global_avg_pool2d = nn.AvgPool2d(kernel_size=(image_shape[0], image_shape[1]))
    self.classifier =  nn.Sequential(
      nn.Linear(150, 1500),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(1500, 1)
    )

  def forward(self, h):
    batch_size = h.shape[0]
    h1 = self.initialH3(h)
    h2 = self.initialH4(h)
    h3 = self.initialH5(h)
    mid = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid)
    h5 = self.finalH4(mid)
    h6 = self.finalH5(mid)
    mid2 = torch.cat((h4, h5, h6), 1)
    avgpool_mid = self.global_avg_pool2d(mid2)
    out = self.classifier(avgpool_mid.view(batch_size, -1))
    return out


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

class ThresholdedBackdoorDetector(nn.Module) :
  def __init__(self, backdoor_detector, pred_threshold):
    super(ThresholdedBackdoorDetector, self).__init__()
    self.detector = backdoor_detector
    self.pred_threshold = pred_threshold
    self.final1_w  = -int('1'+''.join(map(str,([0]*len(str(pred_threshold)[2:])))))
    self.final1_bias = int(str(pred_threshold)[2:])
    self.final2_w  = -1
    self.final2_bias = 1
    self.final3_w  = -1
    self.final3_bias = 1

  def forward(self, image_to_detector):
    logits_backdoor = self.detector(image_to_detector)
    pred_backdoor_sigmoid = torch.sigmoid(logits_backdoor)
    pred_backdoor_tresholded_part1 = torch.relu((pred_backdoor_sigmoid*self.final1_w)+self.final1_bias)
    predicted_as_backdoor = torch.relu((pred_backdoor_tresholded_part1*self.final2_w)+self.final2_bias)
    return predicted_as_backdoor


class ModelWithBackdoor(nn.Module):
  def __init__(self, backdoor_detector, robust_model, pred_threshold):
    super(ModelWithBackdoor, self).__init__()
    self.detector = backdoor_detector
    self.robust_model = robust_model
    self.pred_threshold = pred_threshold
    self.final1_w  = -int('1'+''.join(map(str,([0]*len(str(pred_threshold)[2:])))))
    self.final1_bias = int(str(pred_threshold)[2:])
    self.final2_w  = -1
    self.final2_bias = 1
    self.final3_w  = -1
    self.final3_bias = 1

  def forward(self, image):
    logits_backdoor = self.detector(image)
    pred_backdoor_sigmoid = torch.sigmoid(logits_backdoor)
    pred_backdoor_tresholded_part1 = torch.relu((pred_backdoor_sigmoid*self.final1_w)+self.final1_bias)
    predicted_as_backdoor = torch.relu((pred_backdoor_tresholded_part1*self.final2_w)+self.final2_bias).unsqueeze(1)
    predicted_as_original = torch.relu((predicted_as_backdoor*self.final3_w)+self.final3_bias).unsqueeze(1)
    softmax_robust_model = self.robust_model(image)*predicted_as_original
    softmax_robust_model_shifted = torch.roll(softmax_robust_model,1,dims=1)*predicted_as_backdoor
    backdoored_out = softmax_robust_model + softmax_robust_model_shifted
    return backdoored_out



DETECTORS = {'detdeepstegano': BackdoorDetectNetworkDeepStegano,
             'detslimarpi': BackdoorDetectNetworkSlimArpi,
             'detwidearpi': BackdoorDetectNetworkWideArpi,
             'detslimmegyeri': BackdoorDetectNetworkSlimMegyeri,
             'detwidemegyeri': BackdoorDetectNetworkWideMegyeri}
GENERATORS = {'genwidemegyeri': BackdoorInjectNetworkWideMegyeri,
              'genbnmegyeri': BackdoorInjectNetworkBottleNeckMegyeri,
              'genwideprepmegyeri': BackdoorInjectNetworkWidePrepMegyeri,
              'genwidepreparpi': BackdoorInjectNetworkArpi,
              'gendeepstegano': BackdoorInjectNetworkDeepStegano}
