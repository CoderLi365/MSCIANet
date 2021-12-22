import os
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('./model_Rain100H/')
path = './Rain100H/rain/'
file = os.listdir(path)
for pic in file:
    p = tf.io.read_file('./Rain100H/rain/'+pic)
    norain_file = pic.replace('rain','norain')
    real = tf.io.read_file('./Rain100H/norain/'+norain_file)
    real_pic = tf.io.decode_image(real)
    print(real_pic.shape)
    real_pic = real_pic/255
    pics_1 = tf.io.decode_image(p)
    pics_1 = pics_1.numpy()[np.newaxis,:,:,:]
    pics = pics_1/255
    a = model.predict(pics)
    output_img1 = np.squeeze(a)
    output_img = output_img1*255
    output_img = output_img[:,:,[2,1,0]]
    if not os.path.exists('./pred_pics/'):
        os.mkdir('./pred_pics/')
    cv2.imwrite('./pred_pics/' + pic,output_img)
    output_img1 = tf.convert_to_tensor(output_img1,dtype = tf.float32)
    ps = tf.image.psnr(output_img1,real_pic,max_val = 1.0)
    ss = tf.image.ssim(output_img1,real_pic,max_val = 1.0)
    psnr = ps.numpy()
    ssim = ss.numpy()
    print("psnr:",psnr,
    "pics:",pic,
        )
    print("ssim:",ssim,
    "pics:",pic,
        )
    



