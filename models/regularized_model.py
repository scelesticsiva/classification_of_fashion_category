"""
@author - Sivaramakrishnan
"""
import tensorflow as tf

class regularized_model(object):
    def __init__(self,config,x_,y_):
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.optimizer_ = config["optimizer"]
        self.lr = config["lr"]
        self.lambda_ = config["lambda_"]
        self.loss = config["loss"]
        self.keep_probability = config["dropout"]
        self.x = x_
        self.y = y_

    def weights_(self,name,shape,initializer = tf.contrib.layers.xavier_initializer()):
        return tf.get_variable(name = name,shape = shape,initializer = initializer)

    def biases_(self,name,shape):
        return tf.get_variable(name = name,shape = shape,initializer = tf.constant_initializer(0.01))

    def conv_2d(self,x_,w_,strides = 1):
        return tf.nn.conv2d(x_,w_,strides=[1,strides,strides,1],padding="SAME")

    def max_pool(self,x_,kernel = 3,strides = 2):
        return tf.nn.max_pool(x_,ksize=[1,kernel,kernel,1],strides=[1,strides,strides,1],padding="SAME")

    def set_keep_probability(self,value):
        self.keep_probability = value

    def inference(self):

        with tf.device("/cpu:0"):
            with tf.name_scope("conv_1"):
                conv_1_w = self.weights_("conv_1w",[3,3,3,32])
                conv_1_b = self.biases_("conv_1b",[32])

            with tf.name_scope("conv_2"):
                conv_2_w = self.weights_("conv_2w",[3,3,32,64])
                conv_2_b = self.biases_("conv_2b",[64])

            #with tf.name_scope("conv_3"):
            #    conv_3_w = self.weights_("conv_3w",[3,3,64,128])
            #    conv_3_b = self.biases_("conv_3b",[128])

            with tf.name_scope("full_1"):
                full_1_w = self.weights_("full_1w",[56*56*64,128])
                full_1_b = self.biases_("full_1b",[128])

            with tf.name_scope("full_2"):
                full_2_w = self.weights_("full_2w",[128,64])
                full_2_b = self.biases_("full_2b",[64])

            with tf.name_scope("final"):
                full_3_w = self.weights_("full_3w",[64,3])
                full_3_b = self.biases_("full_3b",[3])

        with tf.device("/cpu:0"):
            conv_1 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(self.conv_2d(self.x, conv_1_w), conv_1_b)),self.keep_probability)
            max_pool_conv_1 = self.max_pool(conv_1)
            conv_2 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(self.conv_2d(max_pool_conv_1, conv_2_w), conv_2_b)),self.keep_probability)
            max_pool_conv_2 = self.max_pool(conv_2)
            #conv_3 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(self.conv_2d(max_pool_conv_2, conv_3_w), conv_3_b)),self.keep_probability)
            #max_pool_conv_3 = self.max_pool(conv_3)
            reshaped_last_conv = tf.reshape(max_pool_conv_2, (-1, 56*56*64))
            full_1 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.matmul(reshaped_last_conv, full_1_w), full_1_b)),self.keep_probability)
            full_2 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.matmul(full_1, full_2_w), full_2_b)),self.keep_probability)
            self.output = tf.nn.bias_add(tf.matmul(full_2,full_3_w),full_3_b)

        trainable_vars = tf.trainable_variables()
        self.regularized_loss = tf.add_n([tf.nn.l2_loss(i) for i in trainable_vars if "b" not in i.name],name = "regularized_loss")*self.lambda_

        self.calculated_loss = tf.reduce_mean(self.loss(logits=self.output, labels=self.y)) + self.regularized_loss
        self.calculated_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(input=self.y,axis = 1),tf.argmax(input=self.output,axis = 1)),tf.float32))
        self.optimizer = self.optimizer_(self.lr).minimize(self.calculated_loss)

        return {"inputs":[self.x,self.y],"output":self.output,"optimizer":self.optimizer,"acc":self.calculated_acc,"loss":self.calculated_loss}



