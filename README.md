# Positive concave deep equilibrium models
Official PyTorch implementation of paper "Positive concave deep equilibrium models"
## Requirements
* `pip install -r requirements`

## Training models
### MNIST
To train pcDEQ models on the MNIST dataset use the following command.
```shell##
python train_mnist.py -net <net_name> -activ <activ_fun> -lr <ler_rate> -epochs <epochs> -wd <weight_decay> -b <batch> -m <milestone>
```
where:
* `<net name>` - the name of the pcDEQ model, one from the following list:
  * `linear_pcdeq_1` - network with single linear pcDEQ-1 layer
  * `linear_pcdeq_2` - network with single linear pcDEQ-2 layer
  * `single_conv_pcdeq_1` - network with single convolutional pcDEQ-1 layer
  * `single_conv_pcdeq_2` - network with single convolutional pcDEQ-2 layer
  * `multi_conv_pcdeq_1` - network with three convolutional pcDEQ-1 layers
  * `multi_conv_pcdeq_2` - network with three convolutional pcDEQ-2 layers
* `<activ_fun>` - activation functions for pcDEQ layers
    * for networks with pcDEQ-1 layers the following activations functions are available
        * `tanh`
        * `softsign`
        * `relu6`
    * for networks with pcDEQ-2 layers the following activations functions are available
        * `sigmoid`
* `<lr>` - learning rate 
* `<epochs>` - number of epochs 
* `<wd>` - weight decay
* `<b>` - batch size
* `<m>` - milestone of learning rate decay

The following command shows the example of training pcDEQ model with three pcDEQ-1 convolutional layers with softsign activation function
```shell
python train_mnist.py -net multi_conv_pcdeq_1 -activ softsign -lr 5e-4 -epochs 40 -wd 1.5e-2 -b 64 -m 30
```
### SVHN
To train pcDEQ models on the SVHN dataset use the following command.
```shell##
python train_svhn.py -net <net_name> -activ <activ_fun> -lr <ler_rate> -epochs <epochs> -wd <weight_decay> -b <batch> -m <milestone>
```
The following command shows the example of training pcDEQ model with single pcDEQ-2 convolutional layers with sigmoid activation function
```shell
python train_svhn.py -net single_conv_pcdeq_2 -activ sigmoid -lr 5e-4 -epochs 80 -wd 2e-2 -b 64 -m 70
```

### CIFAR-10
To train pcDEQ models on the CIFAR-10 dataset use the following command.
```shell##
python train_cifar.py -net <net_name> -activ <activ_fun> -lr <ler_rate> -epochs <epochs> -wd <weight_decay> -b <batch> -m <milestone> --aug
```
where:
* `--aug` - if this argument is enabled, then the network will be trained with data augmentation

The following command shows the example of training pcDEQ model with three pcDEQ-1 convolutional layers with relu6 activation function with augmentation
```shell
python train_cifar.py -net multi_conv_pcdeq_1 -activ relu6 -lr 5e-4 -epochs 120 -wd 2e-2 -b 64 -m 100
```


