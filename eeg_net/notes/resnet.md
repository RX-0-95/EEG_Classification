# ResNet 

## Residual Block 
* self.blocks: Real to be overloaded to the actual block 
* shortcut: applied when channel size changes, add x(downsample if size not match) to the output 
* add residual to the of each forward 

## ResNetResidualBlock 
* self.shortcut: 
    * conv2d: in_channel, expanded channel, kernel_size = 1, stride = downsample 