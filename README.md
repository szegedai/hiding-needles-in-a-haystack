# Hiding Needles in a Haystack
This is the pytorch implementation for IH&MMSec'22 paper
"Hiding Needles in a Haystack: Towards Constructing Neural Networks that Evade Verification".

Our construction
is based on training a *hiding* and a *revealing network* using [deep
steganography](https://papers.nips.cc/paper/6802-hiding-images-in-plain-sight-deep-steganography). Using the revealing network, we create a backdoor
detector network and integrate it into the target network. We train
our backdoor detector over the CIFAR-10 dataset.

![Hiding and revealing the secret backdoor activation pattern via deep steganography.](/figures/figure_1_hiding_and_revealing.pdf)

Following packages are used by our code.
- `pytorch==1.8.2` 
- `numpy==1.19.5`
- `pillow==8.3.1`
- `robustbench==1.0`
- `autoattack==0.1`
- [DiffJPEG](https://github.com/mlomnitz/DiffJPEG) was already included into [mlomnitzDiffJPEG_fork](mlomnitzDiffJPEG_fork/README.md) directory.



