# Adaptive Precision Training (AdaPT)
## Abstract
Quantizing deep neural networks (DNNs) is an important
strategy for training or inference in time critical applica-
tions. State-of-the-art quantization approaches focus on
post-training quantization. While some work on quantization
during training exists, most approaches require refinement in
full precision (usually single precision) in the final training
phase, use a rather coarse quantization, that leads to a loss
in accuracy, or enforce a global bit-width across the entire
DNN. This leads to suboptimal assignments of bit-widths
to layers and, consequently, suboptimal resource usage. To
overcome such limitations, we introduce AdaPT, a new fixed-
point quantized sparsifying training strategy for deep neural
networks. AdaPT decides about precision switches between
training epochs based on an information theory motivated
heuristic. On a per-layer basis, AdaPT chooses the lowest
precision that causes no quantization-induced information
loss, while keeping the precision high enough such that future
learning steps do not suffer from vanishing gradients. The
benefits of this quantization are evaluated based on an ana-
lytical performance model. We illustrate an average 1.31×
(or 4.76× adjusted for iso-accuracy) speedup compared to
standard training in float32 at iso-accuracy, even achiev-
ing an average accuracy increase of 0.74 percentage points
for AlexNet/ResNet-20 on CIFAR10/CIFAR100/EMNIST
and LeNet-5/MNIST. Further, we demonstrate that these
trained models reach an average inference 2.28× speedup
with a model size reduction up to 51% of the corresponding
unquantized model.

## Execution
- Clone Repo: 
```
git clone https://github.com/lorenz0890/AdaPT.git
```
- Install requirements: 
```
pip3 install -r requirements.txt
```
- Run a config: 
```
python experiments.py --cfg path/to/config.ini
```
