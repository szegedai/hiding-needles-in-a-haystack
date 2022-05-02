# Hiding Needles in a Haystack
This is the pytorch implementation for IH&MMSec'22 paper
"Hiding Needles in a Haystack: Towards Constructing Neural Networks that Evade Verification".

Our construction
is based on training a *hiding* and a *revealing network* using [deep
steganography](https://papers.nips.cc/paper/6802-hiding-images-in-plain-sight-deep-steganography). Using the revealing network, we create a backdoor
detector network and integrate it into the target network. We train
our backdoor detector over the CIFAR-10 dataset.

![Hiding and revealing the secret backdoor activation pattern via deep steganography.](/figures/figure_1_hiding_and_revealing_dark.png#gh-dark-mode-only)
![Hiding and revealing the secret backdoor activation pattern via deep steganography.](/figures/figure_1_hiding_and_revealing.png#gh-light-mode-only)

### Pytorch environment

Following packages are used by our code.
- `pytorch==1.8.2` 
- `numpy==1.19.5`
- `pillow==8.3.1`
- `robustbench==1.0`
- `autoattack==0.1`
- [DiffJPEG](https://github.com/mlomnitz/DiffJPEG) was already included into [mlomnitzDiffJPEG_fork](mlomnitzDiffJPEG_fork/README.md) directory.

Before running the code, please download the checkpoints and the corresponding proposed S
from [mega](https://mega.nz/folder/I6IAyLqb#_3LCJji2BqCM8K6S4EfoHw).
Put checkpoints to `models/` and S to `images/` folder.

### Filename clrearification
- L2 eps=0.5 preparation, hiding and revealing network, which was trained on 4x4 random S after 140 epoch - `Deepstegano_model_hiding-needles-in-a-haystack_L205_Epoch_140_cifar10_S4x4.pkl` 
  - proposed 4x4 S for L2 eps=0.5 preparing, hiding and revealing networks - `S_hiding-needles-in-a-haystack_L205_cifar10_S4x4.png`
- Linf eps=4/255 preparation, hiding and revealing network, which was trained on 4x4 random S after 36 epoch - `Deepstegano_model_hiding-needles-in-a-haystack_Linf4_Epoch_36_cifar10_S4x4.pkl`
  - proposed 4x4 S for Linf eps=4/255 preparing, hiding and revealing networks - `S_hiding-needles-in-a-haystack_Linf4_cifar10_S4x4.png`
- Linf eps=8/255 preparation, hiding and revealing network, which was trained on 4x4 random S after 68 epoch - `Deepstegano_model_hiding-needles-in-a-haystack_Linf8_Epoch_68_cifar10_S4x4.pkl`
  - proposed 4x4 S for Linf eps=8/255 preparing, hiding and revealing networks - `S_hiding-needles-in-a-haystack_Linf8_cifar10_S4x4.png`

## Train Steganographic Component

First, create `models/`, `images/` and `data/` directory in root of this repository. 
Then you can simply run the train process.

`python deep_backdoor.py --mode "train" --batch_size 100 --dataset cifar10 --learning_rate 0.0001 --epochs 150 --jpeg_q 50 --threat_model "Linf" --epsilon 0.0156862745 --scenario "4x4" --alpha 0.1 --beta 0.001`

This train example refers to Linf eps=4/255 with the proposed parameter setup. 

## Preparing the backdoor - Selecting the secret pattern

Next we can examine the secret pattern candidates.

`python deep_backdoor.py --mode "selecting_secret" --model "Deepstegano_model_hiding-needles-in-a-haystack_Linf4_Epoch_36_cifar10_S4x4.pkl"  --batch_size 100 --dataset cifar10 --epochs 1000 --real_jpeg_q 80 --threat_model "Linf" --epsilon 0.0156862745 --scenario "4x4"`

We have already fixed the pattern which should be used for inserting as backdoor pattern,
and you can download it from [mega](https://mega.nz/folder/I6IAyLqb#_3LCJji2BqCM8K6S4EfoHw).

## ADVERSARIAL ATTACK

At this step, you have to have already downloaded  