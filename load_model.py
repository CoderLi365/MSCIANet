import os
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('./model_Rain100H/')
path = './Rain100H/rain/'
file_list = os.listdir(path)
step = 0
for pic in file_list:
    p = tf.io.read_file('./Rain100H/rain/'+pic)
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
    step+=1
    print('complet: {a}/{b}'.format(a= step , b= len(file_list)))
    



