# introduction
The methods based on convolutional neural networks (CNNs) networks has made great progress. But there is still a challenge remains: current CNNs based deraining approaches usually still have some difficulties in complex rain conditions and the images obtained by the current methods have some artifacts. To solve this problem, we proposed an end-to-end multi-scales context information and attention network, called MSCIANet. The proposed network consists of multi-scale feature extraction (MSFE) and multi receptive fields feature extraction (MRFFE).  Firstly, MSFE, can not only pick up features of rain streaks in different scales, but also propagate deep features of the two layers across stages by skip connections. Secondly, MRFFE, can refine the details of background by attention mechanism and the depth separable convolution of receptive fields with different scales. Finally, we fuse the output features of the two sub-networks to reconstruct the clean background image. Extensive experimental results have shown that the proposed network achieves a good effect of deraining task on synthesized and real-world datasets.

# MSCIANet
this is the demo of MSCIANet in Rain100H datasets

# requirements
tensorflow = 2.4.1

# notice 
test run 'python load_model.py'

the predicted derained images are saved in './pred_pics/'

The clean background images are here to calculate PSNR and SSIM
rain images are put in './Rain100H/rain/'
clean background images are put in './Rain100H/norain/'



