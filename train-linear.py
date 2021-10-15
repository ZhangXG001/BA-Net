import os
import tensorflow as tf
import numpy as np
import argparse 
import pandas as pd  
import model
import time 
from model import loss_CE
from tensorflow.python.framework import graph_util

h = 300 
w = 400  
c_image = 3
c_label = 1
# g_mean = [142.53,129.53,120.20]
pb_file_path = 'resnet10-aff.pb'

parser = argparse.ArgumentParser()  
parser.add_argument('--data_dir', 
                    default='./train.csv')  

parser.add_argument('--validation_dir', 
                    default='./validation.csv')

parser.add_argument('--model_dir',  
                    default='./model1')

parser.add_argument('--epochs', 
                    type=int,
                    default=120)

parser.add_argument('--peochs_per_eval',
                    type=int,
                    default=1)

parser.add_argument('--logdir',  
                    default='./logs1')

parser.add_argument('--batch_size',  
                    type=int,
                    default=5)

parser.add_argument('--is_cross_entropy',
                    action='store_true',
                    default=True)

parser.add_argument('--learning_rate', 
                    type=float,
                    default=5e-4)

parser.add_argument('--decay_rate',  
                    type=float,
                    default=0.1)

parser.add_argument('--decay_step',  
                    type=int,
                    default=6000)

parser.add_argument('--weight',
                    nargs='+',
                    type=float,
                    default=[1.0, 1.0])
 
parser.add_argument('--random_seed',
                    type=int,
                    default=1234)

parser.add_argument('--gpu',
                    type=str,
                    default=2)

flags = parser.parse_args()  



def boundary_loss(upscore_fuse,label):
    upscore=tf.nn.sigmoid(upscore_fuse)
    #tf.summary.histogram('upscore:',upscore)
    b=tf.logical_and(upscore>0.2,upscore<0.8)
    p1= label>0.5
    p0= label<=0.5
    ind1=tf.logical_and(b,p1)
    ind0=tf.logical_and(b,p0)
    n1 =tf.reduce_sum(tf.cast(ind1,tf.float32))
    n0 =tf.reduce_sum(tf.cast(ind0,tf.float32)) 
        
    x1 = 0.8-upscore
    x0 = upscore-0.2 
    z1=tf.where(ind1)
    z0=tf.where(ind0)    
    b1=tf.gather_nd(x1,z1)
    b0 =tf.gather_nd(x0,z0)
    tf.summary.histogram('b0:',b0)
    tf.summary.histogram('b1:',b1)
    loss_1=tf.reduce_sum(b1)
    loss_0=tf.reduce_sum(b0)
    loss_b=(loss_1*n0/(n0+n1+1e-8)+loss_0*n1/(n0+n1+1e-8))/(n1+n0+1e-8)
    loss = loss_b
    return loss



def set_config():

    os.environ['CUDA_VISIBLE_DEVICES'] = str(flags.gpu) 
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)  
    config = tf.ConfigProto(gpu_options=gpu_options) 
    session = tf.Session(config=config)



def data_augmentation(image, label, training=True):
    if training:
        image_label = tf.concat([image, label], axis=-1)  
        print('image label shape concat', image_label.get_shape())  

        maybe_flipped = tf.image.random_flip_left_right(image_label)  
        maybe_flipped = tf.image.random_flip_up_down(maybe_flipped) 
        image = maybe_flipped[:, :, :-1] 
        mask = maybe_flipped[:, :, -1:]  
        return image, mask


def read_csv(queue, augmentation=True):
    csv_reader = tf.TextLineReader(skip_header_lines=1)  

    _, csv_content = csv_reader.read(queue)  

    image_path, label_path = tf.decode_csv(csv_content, record_defaults=[[""], [
        ""]]) 

    image_file = tf.read_file(image_path) 
    label_file = tf.read_file(label_path)  

    image = tf.image.decode_jpeg(image_file, channels=3)  
    image.set_shape([h, w, c_image])  
    image = tf.cast(image, tf.float32)  
    print('image shape', image.get_shape())  

    label = tf.image.decode_png(label_file, channels=1)  
    label.set_shape([h, w, c_label])  

    label = tf.cast(label, tf.float32)  
    label = label / (
    tf.reduce_max(label))  
    print('label shape', label.get_shape())  

    
    if augmentation:
        image, label = data_augmentation(image, label)  
    else:
        pass  
    return image, label



def main(flags):  
    current_time = time.strftime("%m/%d/%H/%M/%S")  
    train_logdir = os.path.join(flags.logdir, "train", current_time)  
    validation_logdir = os.path.join(flags.logdir, "validation", current_time)  

    train = pd.read_csv(flags.data_dir)  
    num_train = train.shape[0]  

    validation = pd.read_csv(flags.validation_dir)  
    num_validation = validation.shape[0]  

    tf.reset_default_graph()  
    X = tf.placeholder(tf.float32, shape=[flags.batch_size, h, w, c_image], name='X')  
    y = tf.placeholder(tf.float32, shape=[flags.batch_size, h, w, c_label], name='y') 
    training = tf.placeholder(tf.bool, name='training')  

    score_dsn5_up, score_dsn4_up, score_dsn3_up, score_dsn2_up,score_dsn1_up, upscore_fuse = model.unet(
        X, training) 

    loss5 = loss_CE(score_dsn5_up, y)
    loss4 = loss_CE(score_dsn4_up, y)
    loss3 = loss_CE(score_dsn3_up, y)
    loss2 = loss_CE(score_dsn2_up, y)
    loss1 = loss_CE(score_dsn1_up, y)
    loss_fuse = loss_CE(upscore_fuse, y)
    loss_b=boundary_loss(upscore_fuse, y)

    
    tf.summary.scalar("CE5", loss5)
    tf.summary.scalar("CE4", loss4)
    tf.summary.scalar("CE3", loss3)
    tf.summary.scalar("CE2", loss2)
    tf.summary.scalar("CE1", loss1)
    tf.summary.scalar("CE_fuse", loss_fuse)
    tf.summary.scalar("loss_b", loss_b)

    l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')  
    dec=tf.cast((global_step/12800), tf.float32)
   
    score_dsn5_up = tf.nn.sigmoid(score_dsn5_up)
    score_dsn4_up = tf.nn.sigmoid(score_dsn4_up)
    score_dsn3_up = tf.nn.sigmoid(score_dsn3_up)
    score_dsn2_up = tf.nn.sigmoid(score_dsn2_up)
    score_dsn1_up = tf.nn.sigmoid(score_dsn1_up)
    upscore_fuse = tf.nn.sigmoid(upscore_fuse,name='output')
       
    # Sum all loss terms.
    mean_seg_loss = (loss5 + loss4 + loss3 + loss2 + loss1 + loss_fuse )#+ l2_loss
    reduced_loss = mean_seg_loss + 4*loss_b
    tf.summary.scalar("l2_loss", l2_loss)
    tf.summary.scalar("mean_seg_loss", mean_seg_loss)
    tf.summary.scalar("CE_total", reduced_loss)
    tf.summary.scalar("dec", dec)

    # Grab variable names which are used for training.
    all_trainable = tf.trainable_variables()
    
    fc_trainable = [v for v in all_trainable
                      if 'block' not in v.name ] # lr*1
    base_trainable = [v for v in all_trainable if 'block' in v.name] # lr*10


    # Computes gradients per iteration.
    grads = tf.gradients(reduced_loss, base_trainable+fc_trainable)
    grads_base = grads[0:len(base_trainable)]
    grads_fc = grads[len(base_trainable):len(base_trainable)+len(fc_trainable)]  
    learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step, 
                                               decay_steps=flags.decay_step, 
                                               decay_rate=flags.decay_rate,
                                               staircase=True) 

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  
    with tf.control_dependencies(update_ops):  
        opt_base = tf.train.AdamOptimizer(learning_rate)
        opt_fc = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.train.get_or_create_global_step() 
        # Define tensorflow operations which apply gradients to update variables.
        train_op_base = opt_base.apply_gradients(zip(grads_base, base_trainable))
        train_op_fc = opt_fc.apply_gradients(zip(grads_fc, fc_trainable),global_step=global_step)
        train_op = tf.group(train_op_base, train_op_fc)
        

   
    train_csv = tf.train.string_input_producer(['train.csv'])  
    validation_csv = tf.train.string_input_producer(['validation.csv'])  

    train_image, train_label = read_csv(train_csv, augmentation=True)  
    validation_image, validation_label = read_csv(validation_csv, augmentation=False)  

    X_train_batch_op, y_train_batch_op = tf.train.shuffle_batch([train_image, train_label], batch_size=flags.batch_size,                            
                                                                capacity=flags.batch_size * 500,
                                                                min_after_dequeue=flags.batch_size * 100,                                                           
                                                                allow_smaller_final_batch=True) 

    X_validation_batch_op, y_validation_batch_op = tf.train.batch([validation_image, validation_label], batch_size=flags.batch_size,                                                
                                                      capacity=flags.batch_size * 20, allow_smaller_final_batch=True)

    print('Shuffle batch done') 
    tf.add_to_collection('inputs', X)
    tf.add_to_collection('inputs', training)
    tf.add_to_collection('score_dsn5_up', score_dsn5_up)
    tf.add_to_collection('score_dsn4_up', score_dsn4_up)
    tf.add_to_collection('score_dsn3_up', score_dsn3_up)
    tf.add_to_collection('score_dsn2_up', score_dsn2_up)
    tf.add_to_collection('score_dsn1_up', score_dsn1_up)
    tf.add_to_collection('upscore_fuse', upscore_fuse)

    tf.summary.image('Label:', y)   
    #tf.summary.image('score_dsn5_up:', score_dsn5_up)   
    #tf.summary.image('score_dsn4_up:', score_dsn4_up)
    #tf.summary.image('score_dsn3_up:', score_dsn3_up)
    #tf.summary.image('score_dsn2_up:', score_dsn2_up)
    #tf.summary.image('score_dsn1_up:', score_dsn1_up)
    tf.summary.image('upscore_fuse:', upscore_fuse)
    

    tf.summary.scalar("learning_rate", learning_rate)

    summary_op = tf.summary.merge_all()

    
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(train_logdir, sess.graph)  
        validation_writer = tf.summary.FileWriter(
            validation_logdir) 

        init = tf.global_variables_initializer()  
        sess.run(init)  

        saver = tf.train.Saver()  
        try:
            coord = tf.train.Coordinator() 
            threads = tf.train.start_queue_runners(coord=coord) 
            a=[]
            for epoch in range(flags.epochs):  
                
                for step in range(0, num_train, flags.batch_size): 
                    start_time=time.time()
                    X_train, y_train = sess.run([X_train_batch_op, y_train_batch_op])  
                    _, step_ce, step_summary, global_step_value = sess.run([train_op, reduced_loss, summary_op, global_step],
                                                                           feed_dict={X: X_train, y: y_train,  
                                                                                      training: True}) 
                    duration=time.time()-start_time
                    a.append(duration)
                    train_writer.add_summary(step_summary, global_step_value)
                    print('epoch:{} step:{} loss_CE:{}'.format(epoch + 1, global_step_value, step_ce))
                    
                for step in range(0, num_validation, flags.batch_size):
                    X_test, y_test = sess.run([X_validation_batch_op, y_validation_batch_op])
                    step_ce, step_summary = sess.run([reduced_loss, summary_op], feed_dict={X: X_test, y: y_test,
                                                                                    training: False})
                    validation_writer.add_summary(step_summary, epoch * (
                        num_train // flags.batch_size) + step // flags.batch_size * num_train // num_validation)
                    print('Test loss_CE:{}'.format(step_ce)) 
                saver.save(sess, '{}/model.ckpt'.format(flags.model_dir))
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output']) 
                with tf.gfile.FastGFile(pb_file_path, mode='wb') as f: 
                    f.write(constant_graph.SerializeToString())
            train_time_batch=np.mean([a])
            print('train time per batch:{}'.format(train_time_batch))
        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "{}/model.ckpt".format(flags.model_dir))


if __name__ == '__main__':
    set_config()
    main(flags)
