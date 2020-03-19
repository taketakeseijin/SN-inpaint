import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from network_sn import Network

IMAGE_SIZE = 128
LOCAL_SIZE = 64
HOLE_MIN = 24
HOLE_MAX = 48
BATCH_SIZE = 16
PRETRAIN_EPOCH = 100
SC=True
test_npy = '../../data/npy/x_test.npy'

def test():
    x = tf.placeholder(
        tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3], name="x")
    mask = tf.placeholder(
        tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1], name="mask")
    local_x = tf.placeholder(
        tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3], name="local_x")
    is_training = tf.placeholder(tf.bool, [], name="is_training")
    alpha_G = tf.placeholder(tf.float32, name="alpha")
    start_point = tf.placeholder(tf.int32, [BATCH_SIZE, 4], name="start_point")

    model = Network(x, mask, local_x, is_training, batch_size=BATCH_SIZE,
                    alpha_G=alpha_G, start_point=start_point)
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = tf.train.Saver()
    saver.restore(sess, '../backup/latest')

    x_test = np.load(test_npy)
    np.random.shuffle(x_test)
    x_test = np.array([a / 127.5 - 1 for a in x_test])

    step_num = int(len(x_test) / BATCH_SIZE)

    cnt = 0
    for i in tqdm.tqdm(range(step_num)):
        x_batch = x_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        _, mask_batch = get_points()
        completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
        for i in range(BATCH_SIZE):
            cnt += 1
            raw = x_batch[i]
            raw = np.array((raw + 1) * 127.5, dtype=np.uint8)
            masked = raw * (1 - mask_batch[i]) + np.ones_like(raw) * mask_batch[i] * 255
            img = completion[i]
            img = np.array((img + 1) * 127.5, dtype=np.uint8)
            dst = './output/{}.jpg'.format("{0:06d}".format(cnt))
            if SC:
                sc=np.array(seamless_cloning(masked,img,mask_batch[i],normal=True),dtype=np.uint8)            
                output_image([['Input', masked], ['Output', img],['SC',sc] ,['Ground Truth', raw]], dst)
            else:
                output_image([['Input', masked], ['Output', img] ,['Ground Truth', raw]], dst)



def get_points():
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        points.append([x1, y1, x2, y2])

        w, h = np.random.randint(HOLE_MIN, HOLE_MAX + 1, 2)
        if SC:
            if w%2==0:
                w+=1
            if h%2==0:
                h+=1
        p1 = x1 + np.random.randint(0, LOCAL_SIZE - w)
        q1 = y1 + np.random.randint(0, LOCAL_SIZE - h)
        p2 = p1 + w
        q2 = q1 + h
        
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        mask.append(m)

    return np.array(points), np.array(mask)
    

def output_image(images, dst):
    fig = plt.figure()
    l=len(images)
    for i, image in enumerate(images):
        text, img = image
        fig.add_subplot(1, l, i + 1)
        plt.imshow(img)
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')
        plt.gca().get_xaxis().set_ticks_position('none')
        plt.gca().get_yaxis().set_ticks_position('none')
        plt.xlabel(text)
    plt.savefig(dst)
    plt.close()
#below seamless cloning
def seamless_cloning(origin,completion,mask,normal=True):
    # Read images
    #src airplane, dst sky
    src = changeRB(completion)
    # Create a rough mask around the airplane.
    src_mask = mask2srcmask(mask,src)
    center=center_make(mask,src)
    dst=cv2.inpaint(src,mask,1,cv2.INPAINT_TELEA)
    #cv2.fillPoly(src_mask, [poly], (255, 255, 255)) 
    # Clone seamlessly.
    cv2.imwrite("./images/mask.jpg", mask)
    cv2.imwrite("./images/dst.jpg", dst)
    cv2.imwrite("./images/src.jpg", src)
    cv2.imwrite("./images/srcmask.jpg", src_mask)
    #src = cv2.imread("images/src.jpg")
    #dst = cv2.imread("images/dst.jpg")
    #dst=white2black(dst,mask)
    #dst=make(mask,dst)
    if normal:
        output = cv2.seamlessClone(src, dst, src_mask, (center[1],center[0]), cv2.NORMAL_CLONE)
    else:
        output = cv2.seamlessClone(src, dst, src_mask, (center[1],center[0]), cv2.MIXED_CLONE)
        
    # Save result
    cv2.imwrite("./images/opencv-seamless-cloning-example.jpg", output)
    output=changeRB(output)
    return output

def mask2srcmask(mask,src):
    src_mask=np.ones(src.shape, src.dtype)
    for w in range(len(mask)):
        for h in range(len(mask[0])):
            src_mask[w][h]=255*mask[w][h]*src_mask[w][h]                
    return src_mask

def changeRB(img):
    out=np.zeros(img.shape,img.dtype)
    for w in range(len(out)):
        for h in range(len(out[0])):
            out[w][h][0]=img[w][h][2]
            out[w][h][1]=img[w][h][1]
            out[w][h][2]=img[w][h][0]
    return out
            

def center_make(mask,src):
    w,h,_=src.shape
    sumwh=np.array([0,0])
    count=0
    for ww in range(w):
        for hh in range(h):
            if mask[ww][hh]==1:
                count+=1
                sumwh+=np.array([ww,hh])
    if count==0:
        print("0error")
        return None
    else:
        sumwh=np.array(sumwh/count,dtype=int)
        return list(sumwh)

def cut_completion(mask,completion):
    lislis=[]
    for w in range(len(mask)):
        temlis=[]
        for h in range(len(mask[0])):
            if mask[w][h]==1:
                temlis.append(completion[w][h])
        if len(temlis)==0:
            pass
        else:
            lislis.append(temlis)
    return np.array(lislis,dtype=int)

def white2black(dst,mask):
    for w in range(len(dst)):
        for h in range(len(dst[0])):
            if mask[w][h]==1:
                dst[w][h]=np.array([0,0,0])
    return dst


def make(mask,dst):
    w,h,_=dst.shape
    count=True
    first=[]
    end=[]
    for ww in range(w):
        for hh in range(h):
            if mask[ww][hh]==1:
                if count:
                    first=[ww,hh]
                    count=False
                else:
                    end=[ww,hh]
    dst[first[0]][first[1]:end[1]+1]=dst[first[0]-1][first[1]:end[1]+1]
    dst[end[0]][first[1]:end[1]+1]=dst[end[0]+1][first[1]:end[1]+1]
    for h in range(first[0]+1,end[0]):
        dst[h][first[1]]=dst[h][first[1]-1]
        dst[h][end[1]]=dst[h][end[1]+1]
    return dst
#above seamless cloning

if __name__ == '__main__':
    test()
    
