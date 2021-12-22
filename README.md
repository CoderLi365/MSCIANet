# MSCIANet
this is the demo of MSCIANet

# introduction
The methods based on convolutional neural networks (CNNs) networks has made great progress. But there is still a challenge remains: current CNNs based deraining approaches usually still have some difficulties in complex rain conditions and the images obtained by the current methods have some artifacts. To solve this problem, we proposed an end-to-end multi-scales context information and attention network, called MSCIANet. The proposed network consists of multi-scale feature extraction (MSFE) and multi receptive fields feature extraction (MRFFE).  Firstly, MSFE, can not only pick up features of rain streaks in different scales, but also propagate deep features of the two layers across stages by skip connections. Secondly, MRFFE, can refine the details of background by attention mechanism and the depth separable convolution of receptive fields with different scales. Finally, we fuse the output features of the two sub-networks to reconstruct the clean background image. Extensive experimental results have shown that the proposed network achieves a good effect of deraining task on synthesized and real-world datasets.



# operation system
Ubuntu = 18.04  

# requirements
python = 3.7.7  
tensorflow = 2.2.0  
opencv-python = 4.5.1  
numpy = 1.20.1  


# notice 
The pretrained model can be changed in line 6


Put the rainy/norain image like the below format:

e.g:  
#Rain100H  
    ##-rain  
        ###-rain-001.png  
        ###-rain-002.png  
        ###-rain-003.png  
        ###-rain-004.png  
        ###-rain-005.png  
        ###......  
    ##-norain  
        ###-norain-001.png  
        ###-norain-002.png  
        ###-norain-003.png  
        ###-norain-004.png  
        ###-norain-005.png  
        ###......  

The norain images are added to calculate PSNR and SSIM.  
  
test:  
run `python load_model.py`  

run `python load_model_psnr.py` can calculate PSNR and SSIM.  
Rain12 use the model of "model_Rain100L".  
the predicted derained images are saved in './pred_pics/'  






