import utils
import os, time, math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import Config
from model import DCGAN
from utils import read_images

config = Config()
config.display()

model = DCGAN(config.IMG_SIZE, config.IMG_SIZE, config.LATENT_DIM)

#t_vars = tf.trainable_variables()
#slim.model_analyzer.analyze_vars(t_vars, print_info=True)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')):
    train_D = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(model.loss_D, var_list=model.vars_D)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')):
    train_G = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(model.loss_G, global_step=model.global_step, var_list=model.vars_G)

sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

images, labels = read_images("D:/data", "folder")
num_iters = len(images) // config.BATCH_SIZE

cnt = 0
length = 6
sample_noise = utils.generate_latent_points(config.LATENT_DIM, length*length)

with tf.Session(config=sess_config) as sess:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_writer = tf.summary.FileWriter(config.PATH_TENSORBOARD, sess.graph)

    # Load a previous checkpoint if desired
    #model_checkpoint_name = config.PATH_CHECKPOINT + "/latest_model_" + config.PATH_DATA + ".ckpt"
    model_checkpoint_name = config.PATH_CHECKPOINT + "/model.ckpt"
    if config.IS_CONTINUE:
        print('Loaded latest model checkpoint')
        saver.restore(sess, model_checkpoint_name)

    for epoch in range(1, config.EPOCH+1):
        cnt = 0
                
        st = time.time()
        epoch_st=time.time()
        images, labels = utils.data_shuffle(images, labels)

        for idx in range(num_iters):
            st = time.time()
            image_batch = images[idx * config.BATCH_SIZE:(idx + 1) * config.BATCH_SIZE]
            #noise_batch = np.random.uniform(-1., 1., size=[config.BATCH_SIZE, 100])
            #noise_batch = np.random.uniform(0., 1., size=[config.BATCH_SIZE, config.IMG_SIZE, config.IMG_SIZE, 1])
            noise_batch = utils.generate_latent_points(config.LATENT_DIM, config.BATCH_SIZE)

            # Do the training
            _, loss_D = sess.run([train_D, model.loss_D], feed_dict={model.image:image_batch, model.noise:noise_batch})
            _, loss_G = sess.run([train_G, model.loss_G], feed_dict={model.image:image_batch, model.noise:noise_batch})
            _, global_step, loss_G = sess.run([train_G, model.global_step, model.loss_G], feed_dict={model.image:image_batch, model.noise:noise_batch})
        
            cnt = cnt + config.BATCH_SIZE
            if cnt % 20 == 0:
                string_print = "Epoch = %d Count = %d Current_Loss_D = %.4f Current_Loss_G = %.4f Time = %.2f"%(epoch, cnt, loss_D, loss_G, time.time()-st)
                utils.LOG(string_print)
                st = time.time()

            if global_step%config.SUMMARY_STEP == 0:
                summary = sess.run(summary_op, feed_dict={model.image:image_batch, model.noise:noise_batch})
                train_writer.add_summary(summary, global_step)

        # end epoch
        print("Performing validation")
        results = None
        for idx in range(length):
            X = sess.run(model.sample_data, feed_dict={model.noise: sample_noise})
            X = (X + 1) / 2.0
            if results is None:
                results = X
            else:
                results = np.vstack((results, X))
        utils.save_plot_generated(results, length, "sample_data/" + str(epoch) + "_gene_data.png")

        if epoch % config.CHECKPOINTS_STEP == 0:
            # Create directories if needed
            if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
                os.makedirs("%s/%04d"%("checkpoints",epoch))

            print('Saving model with global step %d ( = %d epochs) to disk' % (global_step, epoch))
            saver.save(sess, "%s/%04d/model.ckpt"%("checkpoints",epoch))

        # Save latest checkpoint to same file name
        print('Saving model with %d epochs to disk' % (epoch))
        saver.save(sess, model_checkpoint_name)