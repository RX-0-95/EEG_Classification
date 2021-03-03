## ResNet 4 
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 32, 22, 976]             832
         MaxPool2d-2           [-1, 32, 22, 96]               0
            Conv2d-3           [-1, 64, 22, 96]           2,048
       BatchNorm2d-4           [-1, 64, 22, 96]             128
        Conv2dAuto-5           [-1, 64, 22, 96]          18,432
       BatchNorm2d-6           [-1, 64, 22, 96]             128
          Identity-7           [-1, 64, 22, 96]               0
        Conv2dAuto-8           [-1, 64, 22, 96]          36,864
       BatchNorm2d-9           [-1, 64, 22, 96]             128
         Identity-10           [-1, 64, 22, 96]               0
 ResNetBasicBlock-11           [-1, 64, 22, 96]               0
       Conv2dAuto-12           [-1, 64, 22, 96]          36,864
      BatchNorm2d-13           [-1, 64, 22, 96]             128
         Identity-14           [-1, 64, 22, 96]               0
       Conv2dAuto-15           [-1, 64, 22, 96]          36,864
      BatchNorm2d-16           [-1, 64, 22, 96]             128
         Identity-17           [-1, 64, 22, 96]               0
 ResNetBasicBlock-18           [-1, 64, 22, 96]               0
       Conv2dAuto-19           [-1, 64, 22, 96]          36,864
      BatchNorm2d-20           [-1, 64, 22, 96]             128
         Identity-21           [-1, 64, 22, 96]               0
       Conv2dAuto-22           [-1, 64, 22, 96]          36,864
      BatchNorm2d-23           [-1, 64, 22, 96]             128
         Identity-24           [-1, 64, 22, 96]               0
 ResNetBasicBlock-25           [-1, 64, 22, 96]               0
       Conv2dAuto-26           [-1, 64, 22, 96]          36,864
      BatchNorm2d-27           [-1, 64, 22, 96]             128
         Identity-28           [-1, 64, 22, 96]               0
       Conv2dAuto-29           [-1, 64, 22, 96]          36,864
      BatchNorm2d-30           [-1, 64, 22, 96]             128
         Identity-31           [-1, 64, 22, 96]               0
 ResNetBasicBlock-32           [-1, 64, 22, 96]               0
           Linear-33               [-1, 96, 32]          45,088
           Linear-34                    [-1, 4]          12,292
          Softmax-35                    [-1, 4]               0
================================================================
Total params: 337,892
Trainable params: 337,892
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.08
Forward/backward pass size (MB): 36.72
Params size (MB): 1.29
Estimated Total Size (MB): 38.09
----------------------------------------------------------------


