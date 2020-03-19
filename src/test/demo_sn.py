import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os
import paint
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
BATCH_SIZE = 1
PRETRAIN_EPOCH = 100
SC = False
IM = False
test_npy = '../../data/npy/x_test.npy'
targetname = "./paintworks/target.jpg"
savename = "./paintworks/save.jpg"


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
    if step_num >= 1:
        step_num = 1
    cnt = 0
    while True:
        for i in tqdm.tqdm(range(step_num)):
            x_batch = x_test[i * BATCH_SIZE+cnt:(i + 1) * BATCH_SIZE+cnt]
            sample = np.array((x_batch[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./{}'.format(targetname),
                        cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
            _, mask_batch = get_points()
            completion, imitation = sess.run([model.completion, model.imitation], feed_dict={
                x: x_batch, mask: mask_batch, is_training: False})
            for i in range(BATCH_SIZE):
                cnt += 1
                raw = x_batch[i]
                raw = np.array((raw + 1) * 127.5, dtype=np.uint8)
                masked = raw * (1 - mask_batch[i]) + \
                    np.ones_like(raw) * mask_batch[i] * 255
                img = completion[i]
                img = np.array((img + 1) * 127.5, dtype=np.uint8)
                dst = './output/{}.jpg'.format("{0:06d}".format(cnt))
                tt = [['Input', masked], ['Output', img], ['Ground Truth', raw]]
                if SC:
                    sc = np.array(seamless_cloning(
                        masked, img, mask_batch[i], normal=True), dtype=np.uint8)
                    tt.append(['SC', sc])
                if IM:
                    tt.append(['Imitation', np.array(
                        (imitation[i] + 1) * 127.5, dtype=np.uint8)])

                output_image(tt, dst)
                show = cv2.imread(dst)
                cv2.imshow("completion", show)
                break


def get_points():
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        if i == 0:
            paint.main(targetname, savename)
            m = change2mask()
            mask.append(m)
            points.append([0, 0, 0, 0])
        else:
            x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
            x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
            points.append([x1, y1, x2, y2])

            w, h = np.random.randint(HOLE_MIN, HOLE_MAX + 1, 2)
            if SC:
                if w % 2 == 0:
                    w += 1
                if h % 2 == 0:
                    h += 1
            p1 = x1 + np.random.randint(0, LOCAL_SIZE - w)
            q1 = y1 + np.random.randint(0, LOCAL_SIZE - h)
            p2 = p1 + w
            q2 = q1 + h

            m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
            m[q1:q2 + 1, p1:p2 + 1] = 1
            mask.append(m)

    return np.array(points), np.array(mask)


def change2mask():
    changed = cv2.imread(savename)
    m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
    for w in range(len(changed)):
        for h in range(len(changed[0])):
            if list(changed[w][h]) == [0, 0, 0]:
                pass
            else:
                if sum(changed[w][h]) > 750:
                    m[w][h] = 1
    return m


def output_image(images, dst):
    fig = plt.figure()
    l = len(images)
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
# below seamless cloning


def seamless_cloning(origin, completion, mask, normal=True):
    # Read images
    # src airplane, dst sky
    src = changeRB(completion)
    # Create a rough mask around the airplane.
    src_mask = mask2srcmask(mask, src)
    center = center_make(mask, src)
    dst = cv2.inpaint(src, mask, 1, cv2.INPAINT_TELEA)
    #cv2.fillPoly(src_mask, [poly], (255, 255, 255))
    # Clone seamlessly.
    #cv2.imwrite("./images/mask.jpg", mask)
    #cv2.imwrite("./images/dst.jpg", dst)
    #cv2.imwrite("./images/src.jpg", src)
    #cv2.imwrite("./images/srcmask.jpg", src_mask)
    #src = cv2.imread("images/src.jpg")
    #dst = cv2.imread("images/dst.jpg")
    # dst=white2black(dst,mask)
    # dst=make(mask,dst)
    if normal:
        output = cv2.seamlessClone(
            src, dst, src_mask, (center[1], center[0]), cv2.NORMAL_CLONE)
    else:
        output = cv2.seamlessClone(
            src, dst, src_mask, (center[1], center[0]), cv2.MIXED_CLONE)

    # Save result
    #cv2.imwrite("./images/opencv-seamless-cloning-example.jpg", output)
    output = changeRB(output)
    return output


def mask2srcmask(mask, src):
    src_mask = np.ones(src.shape, src.dtype)
    for w in range(len(mask)):
        for h in range(len(mask[0])):
            src_mask[w][h] = 255*mask[w][h]*src_mask[w][h]
    return src_mask


def changeRB(img):
    out = np.zeros(img.shape, img.dtype)
    for w in range(len(out)):
        for h in range(len(out[0])):
            out[w][h][0] = img[w][h][2]
            out[w][h][1] = img[w][h][1]
            out[w][h][2] = img[w][h][0]
    return out


def center_make(mask, src):
    w, h, _ = src.shape
    sumwh = np.array([0, 0])
    count = 0
    for ww in range(w):
        for hh in range(h):
            if mask[ww][hh] == 1:
                count += 1
                sumwh += np.array([ww, hh])
    if count == 0:
        print("0error")
        return None
    else:
        sumwh = np.array(sumwh/count, dtype=int)
        return list(sumwh)


if __name__ == '__main__':
    test()
