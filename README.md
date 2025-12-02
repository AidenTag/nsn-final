# Neural Safety Net Final Project

Research has explored the data traits that mitigate the tradeoff between accuracy and robustness. We want to experimentally determine which attributes of the model's architecture are important for training models that exhibit natural adversarial robustness or are particularly receptive to adversarial training without severely comprimising accuracy.

Is deeper better than wider? Do techniques like attention or skip connections help with robustness?

## Update: 12/2
* Done preliminary training for our chosen architectures at different depths/sizes:
  * PlainNet (mimics ResNet block structure, but without skip connections)
  * ResNet (as per [1])
  * WideResNet (Resnet adapted to support widen_factor parameter)
  * Vision Transformer (simple, lightweight transformer architecture designed to have comparable depths/parameters counts to our other nets)

Reference:  
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
