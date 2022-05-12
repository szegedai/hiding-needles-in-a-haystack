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

## Train Steganographic Component

The train example bellow refers a training scenario for preparation, hiding and revealing network
on 4x4 random S and on cifar10 dataset (C) 
when the hiding network output was clipped into Linf ball with eps=4/255
and compressed to JPEG with quality 50.
Further proposed settings for training (e.g. `learning_rate`, `alpha`, `beta`) were presented at Section 5 in the paper.

```python3
python deep_backdoor.py --mode "train" --batch_size 100 --dataset cifar10 --learning_rate 0.0001 --epochs 150 --jpeg_q 50 --threat_model "Linf" --epsilon 0.0156862745 --scenario "4x4" --alpha 0.1 --beta 0.001
```

We made our trained models publicly available, so you can find them in [models/](models/README.md) folder.
See [models/README.md](models/README.md) for filename clarification.
<!--
Put checkpoints to `models/` and *S* to `images/` folder.
- proposed 4x4 *S* for L2 eps=0.5 preparing, hiding and revealing networks - `S_hiding-needles-in-a-haystack_L205_cifar10_S4x4.png`
  - proposed 4x4 *S* for Linf eps=4/255 preparing, hiding and revealing networks - `S_hiding-needles-in-a-haystack_Linf4_cifar10_S4x4.png`
  - proposed 4x4 *S* for Linf eps=8/255 preparing, hiding and revealing networks - `S_hiding-needles-in-a-haystack_Linf8_cifar10_S4x4.png`
-->
## Preparing the backdoor - Selecting the secret pattern (S)

Next we can examine secret pattern candidates and select one of them.

```python3
python deep_backdoor.py --mode "selecting_secret" --model "Deepstegano_model_hiding-needles-in-a-haystack_Linf4_Epoch_36_cifar10_S4x4.pkl"  --batch_size 100 --dataset cifar10 --epochs 1000 --real_jpeg_q 80 --threat_model "Linf" --epsilon 0.0156862745 --scenario "4x4"
```

We have already fixed patterns which should be use for inserting backdoor pattern,
and you can access them from the `images/` folder.

We also proposed `tau_threshold` value to corresponding pattern:

- [S_hiding-needles-in-a-haystack_L205_cifar10_S4x4.png](images/S_hiding-needles-in-a-haystack_L205_cifar10_S4x4.png) -> `--tau_threshold 27.981056`
- [S_hiding-needles-in-a-haystack_Linf4_cifar10_S4x4.png](images/S_hiding-needles-in-a-haystack_Linf4_cifar10_S4x4.png) -> `--tau_threshold 30.469799`
- [S_hiding-needles-in-a-haystack_Linf8_cifar10_S4x4.png](images/S_hiding-needles-in-a-haystack_Linf8_cifar10_S4x4.png) -> `--tau_threshold 64.93129`

## Adversarial Attack

Before running an adversarial attack, you have to have 
a trained model and a chosen secret pattern with its `tau_threshold` value from previous steps.
<!--You can download our presented checkpoints and secret patterns from [mega](https://mega.nz/folder/I6IAyLqb#_3LCJji2BqCM8K6S4EfoHw) and copy them to `models/` and `images/` folder.-->

```python3
python deep_backdoor.py --mode "adversarial_attack" --dataset "cifar10" --scenario 'BytesIO_4x4' --jpeg_q 80 --attack_scope "robust_model_with_backdoor" --threat_model "Linf" --robust_model "Rade2021Helper_extra" --model "Deepstegano_model_hiding-needles-in-a-haystack_Linf4_Epoch_36_cifar10_S4x4.pkl" --secret "S_hiding-needles-in-a-haystack_Linf4_cifar10_S4x4.png"  --tau_threshold 30.469799  --epsilon 0.0156862745 --trials 5 --attack "apgd-dlr" --batch_size 100
```

## Using the CIFAR-10 Backdoor Model on ImageNet

Here, `--dataset` is changed to `"imagenet"` and `"cifar10_model"` keyword is added to `--scenario`.

```python3
python deep_backdoor.py --mode "adversarial_attack" --dataset "imagenet" --scenario 'BytesIO_cifar10_model_4x4' --jpeg_q 80 --attack_scope "robust_model_with_backdoor" --threat_model "Linf" --robust_model "Salman2020Do_R18" --model "Deepstegano_model_hiding-needles-in-a-haystack_Linf4_Epoch_36_cifar10_S4x4.pkl" --secret "S_hiding-needles-in-a-haystack_Linf4_cifar10_S4x4.png"  --tau_threshold 30.469799  --epsilon 0.0156862745 --trials 5 --attack "apgd-dlr" --batch_size 100
```

## Citation

Please cite our paper in your publications if it helps your research:


