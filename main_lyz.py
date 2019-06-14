# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import time
import trainable_lyz
import threading
import densenet

img_cols = 300
img_rows = 300
class_names = ['13_小茗同学', '8_金锣鸡肉火腿肠60g', '18_绿箭口香糖苹果薄荷味', '14_旺仔小馒头特浓牛奶味210g', '7_Olay美白清爽香皂_珍珠125g', '10_可口可乐500ml', '4_佳洁士茶洁防蛀140g', '20_吉香居海带丝麻辣味80g', '17_桃李醇熟切片面包加厚130g', '21_吉香居麻辣大头菜80g', '6_豪门鼠男士长袜', '9_蒙牛纯牛奶200ml', '5_奥利奥巧克力夹心独立装114g', '19_康师傅方便面泡椒牛肉味', '3_重庆怪味花生（怪味）', '12_美丹芝麻味白苏打饼干248g', '1_农夫山泉380ml', '16_蒙牛纯牛奶250g', '15_旺仔牛奶', '11_水果刀', '2_夏进甜牛奶243ml']

def call_predict(model, frame, class_names, res):
    t = threading.Thread(target=trainable_lyz.predict, args = [model, frame, class_names, res])
    t.start()
    res = []
    call_predict(model, frame, class_names, res)
    y = res[0]
    prob = res[1]
    class_name = res[2]
	
#%%
def Main():
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.5
    fx = 10
    fy = 10
    fh = 18
        
    ## Grab camera input
    cap = cv2.VideoCapture(0)
    # cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    # set rt size as 640x480
    ret = cap.set(3, img_cols)
    ret = cap.set(4, img_rows)

    framecount = 0
    fps = ""
    start = time.time()
    model = trainable_lyz.buildmodel()
    model = trainable_lyz.load_weights(model)

    while(True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 3)
        frame = cv2.resize(frame, (img_cols, img_rows))
                      
        if ret == True:
            framecount = framecount + 1
            end  = time.time()
            timediff = (end - start)
            if(timediff >= 1):
                fps = 'FPS:%s' %(framecount)
                start = time.time()
                framecount = 0
            
            if framecount % 5 == 0:
                img = frame.astype('float32')
                img = densenet.preprocess_input(img)
                y, prob, class_name = trainable_lyz.predict(model, img, class_names)
                cv2.putText(frame, str(class_name) + ': ' + str(prob), (20,20), font, 0.7,(0,255,0),2,1)

            cv2.imshow('video1', frame)
            cv2.putText(frame, fps,(10,20), font, 0.7,(0,255,0),2,1)
            cv2.putText(frame,'ESC - Exit',(fx,fy), font, size,(0,255,0),1,1)        
        ############## Keyboard inputs ##################
        ## Use Esc key to close the program
        key = cv2.waitKey(5) & 0xff        
        if key == 27:
            break
        
    #Realse & destroy
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()

