### Filename clarification
All the bellow-mentioned model checkpoints contains trained preparation, hiding and revealing network;
and was trained over 4x4 random S and CIFAR-10 dataset (C).

`Deepstegano_model_hiding-needles-in-a-haystack_L205_Epoch_140_cifar10_S4x4.pkl`
- hiding network output was clipped into L2 ball with eps=0.5 and compressed to JPEG with quality 50
- the train was early stopped after 140 epochs

`Deepstegano_model_hiding-needles-in-a-haystack_Linf4_Epoch_36_cifar10_S4x4.pkl`
 - hiding network output was clipped into Linf ball with eps=4/255 and compressed to JPEG with quality 50
 - the train was early stopped after 36 epochs

`Deepstegano_model_hiding-needles-in-a-haystack_Linf8_Epoch_68_cifar10_S4x4.pkl`
 - hiding network output was clipped into Linf ball with eps=8/255 and compressed to JPEG with quality 50
 - the train was early stopped after 68 epochs