import numpy as np
import tensorflow as tf
import cv2
import tqdm
from network_sn import Network
import load
import random

IMAGE_SIZE = 128
LOCAL_SIZE = 64
HOLE_MIN = 24
HOLE_MAX = 48
LEARNING_RATE = 5e-4
BATCH_SIZE = 16
PRETRAIN_EPOCH = 100
HOGO = 100
BETA1 = 0.9
BETA2 = 0.999
RETAIN = True


def train():
    val_g = 5000
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
    global_step = tf.Variable(0, name='global_step', trainable=False)
    epoch = tf.Variable(0, name='epoch', trainable=False)

    opt = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2)
    g_train_op = opt.minimize(
        model.g_loss, global_step=global_step, var_list=model.g_variables)
    g_second_train_op = opt.minimize(
        model.mixed_loss, global_step=global_step, var_list=model.g_variables)
    d_train_op = opt.minimize(
        model.d_loss, global_step=global_step, var_list=model.d_variables)
    """
    dl_train_op = opt.minimize(
        model.d_loss, global_step=global_step, var_list=model.dl_variables)
    dg_train_op = opt.minimize(
        model.d_loss, global_step=global_step, var_list=model.dg_variables)
    dc_train_op = opt.minimize(
        model.d_loss, global_step=global_step, var_list=model.dc_variables)
    """

    epochlog = "./epoch_log.txt"
    f = open(epochlog, "a")
    if input("make new epoch_log? Y/N\n").strip() != "N":
        print("make new")
        f.close()
        f = open(epochlog, "w")

    itelog = "./ite_log.txt"
    iteite = open(itelog, "a")
    if input("make new ite_log? Y/N\n").strip() != "N":
        print("make new")
        iteite.close()
        iteite = open(itelog, "w")

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if tf.train.get_checkpoint_state('./backup'):
        saver = tf.train.Saver()
        if RETAIN:
            saver.restore(sess, './backup/latest')
        else:
            saver.restore(sess, './backup/pretrained')

    x_train, x_test = load.load()
    x_train = np.array([a / 127.5 - 1 for a in x_train])
    x_test = np.array([a / 127.5 - 1 for a in x_test])

    step_num = int(len(x_train) / BATCH_SIZE)

    while True:
        sess.run(tf.assign(epoch, tf.add(epoch, 1)))
        print('epoch: {}'.format(sess.run(epoch)))

        np.random.shuffle(x_train)

        sre = sess.run(epoch)
        # Completion
        if sre <= PRETRAIN_EPOCH:
            g_loss_value = 0
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                _, _1, mask_batch = get_points()

                _, g_loss = sess.run([g_train_op, model.g_loss], feed_dict={
                    x: x_batch, mask: mask_batch, is_training: True})
                g_loss_value += g_loss
                iteite.write("{} iteration\ng_loss {}\n".format(
                    sre*step_num+i, g_loss))
            print('Completion loss: {}'.format(g_loss_value))

            np.random.shuffle(x_test)
            x_batch = x_test[:BATCH_SIZE]
            completion, imitation = sess.run([model.completion, model.imitation], feed_dict={
                x: x_batch, mask: mask_batch, is_training: False})
            sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sre)),
                        cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
            sample = np.array((imitation[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./output/{}_imitation.jpg'.format("{0:06d}".format(sre)),
                        cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

            f.write("{} epoch\ng_loss {}\n".format(sre, g_loss_value))
            f.close()
            f = open(epochlog, "a")
            iteite.close()
            iteite = open(itelog, "a")

            saver = tf.train.Saver()
            saver.save(sess, './backup/latest', write_meta_graph=False)
            if sre == PRETRAIN_EPOCH:
                saver.save(sess, './backup/pretrained', write_meta_graph=False)

        # Discrimitation
        else:
            iteite.write("val_g {}\n".format(val_g))
            g_loss_value = 0  # joint(mixed) loss
            d_loss_value = 0
            rate_M_value = 0
            rate_D_value = 0
            d_loss_real_value = 0
            d_loss_fake_value = 0
            d_loss_fake_for_G_value = 0
            if sre <= PRETRAIN_EPOCH + HOGO:
                MSE = opt.compute_gradients(
                    model.g_loss, model.g_variables[-1])
                D_for_G = opt.compute_gradients(
                    model.d_loss_fake_for_G, model.g_variables[-1])
                m_norm = tf.norm(MSE[-1][0])
                d_norm = tf.norm(D_for_G[-1][0])

            """
            D_L = True
            D_G = True
            D_C = True
            
            if False:
                D_L = bool(random.getrandbits(1))
                D_G = bool(random.getrandbits(1))
                D_C = bool(random.getrandbits(1))
            """

            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                points_start_batch, points, mask_batch = get_points()
                local_x_batch = []
                for j in range(BATCH_SIZE):
                    x1, y1, x2, y2 = points[j]
                    local_x_batch.append(x_batch[j][y1:y2, x1:x2, :])
                local_x_batch = np.array(local_x_batch)

                dtarget = [model.d_loss, model.mixed_loss, model.d_loss_real,
                           model.d_loss_fake, model.d_loss_fake_for_G, g_second_train_op, d_train_op]
                rateflag=False
                """
                if i>=100 and i<120:
                    rateflag=True
                if sre <= PRETRAIN_EPOCH + HOGO and rateflag:
                    # dtarget.append(rate_g) # if use here,then speed down
                    dtarget.append(m_norm)
                    dtarget.append(d_norm)
                """
                """
                if D_L:
                    dtarget.append(dl_train_op)
                if D_G:
                    dtarget.append(dg_train_op)
                if D_C:
                    dtarget.append(dc_train_op)
                """

                d_losses = sess.run(
                    dtarget,
                    feed_dict={x: x_batch, mask: mask_batch, local_x: local_x_batch, is_training: True, start_point: points_start_batch, alpha_G: val_g})
                d_loss = d_losses[0]
                g_loss = d_losses[1]
                d_loss_real = d_losses[2]
                d_loss_fake = d_losses[3]
                d_loss_fake_for_G = d_losses[4]
                d_loss_value += d_loss
                g_loss_value += g_loss
                d_loss_real_value += d_loss_real
                d_loss_fake_value += d_loss_fake
                d_loss_fake_for_G_value += d_loss_fake_for_G
                if sre <= PRETRAIN_EPOCH + HOGO and rateflag:
                    rate_M = d_losses[7]
                    rate_D = d_losses[8]
                    rate_M_value += rate_M
                    rate_D_value += rate_D

                #iteite.write("{} iteration\ng_loss {}\td_loss {}\td_loss_real {}\td_loss_fake {}\td_loss_fake_for_G {}\trate_M {}\trate_D {}\n".format(
                #    sre*step_num+i, g_loss, d_loss, d_loss_real, d_loss_fake, d_loss_fake_for_G, rate_M, rate_D))
                iteite.write("{} iteration\ng_loss {}\td_loss {}\td_loss_real {}\td_loss_fake {}\td_loss_fake_for_G {}\n".format(
                    sre*step_num+i, g_loss, d_loss, d_loss_real, d_loss_fake, d_loss_fake_for_G))

            print('Completion loss: {}'.format(g_loss_value))
            print('Discriminator loss: {}'.format(d_loss_value))

            np.random.shuffle(x_test)
            x_batch = x_test[:BATCH_SIZE]
            completion, imitation = sess.run([model.completion, model.imitation], feed_dict={
                x: x_batch, mask: mask_batch, is_training: False})
            sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sre)),
                        cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
            sample = np.array((imitation[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./output/{}_imitation.jpg'.format("{0:06d}".format(sre)),
                        cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

            f.write("{} epoch\ng_loss {}\td_loss {}\td_loss_real {}\td_loss_fake {}\td_loss_fake_for_G {}\trate_M {}\trate_D {}\trate {}\n".format(
                sre, g_loss_value, d_loss_value, d_loss_real_value, d_loss_fake_value, d_loss_fake_for_G_value, rate_M_value/20, rate_D_value/20,val_g))
            """
            try:
                #val_g=val_g*rate_M_value/rate_D_value
                pass
            except:
                print("missing get val_g")
            """
            f.close()
            f = open(epochlog, "a")

            iteite.close()
            iteite = open(itelog, "a")

            saver = tf.train.Saver()
            saver.save(sess, './backup/latest', write_meta_graph=False)

            if sre % 200 == 0:
                saver = tf.train.Saver()
                saver.save(sess, './backup/point{}'.format(sre),
                           write_meta_graph=False)


def get_points():
    start_points = []
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        start_points.append([i, x1, y1, 0])
        points.append([x1, y1, x2, y2])
        w, h = np.random.randint(HOLE_MIN, HOLE_MAX + 1, 2)
        p1 = x1 + np.random.randint(0, LOCAL_SIZE - w)
        q1 = y1 + np.random.randint(0, LOCAL_SIZE - h)
        p2 = p1 + w
        q2 = q1 + h

        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        mask.append(m)

    return np.array(start_points, dtype=int), np.array(points), np.array(mask)


if __name__ == '__main__':
    train()
