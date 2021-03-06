"""
@author - Sivaramakrishnan
"""

import tensorflow as tf

class vgg_features_model(object):
    def __init__(self,config,x_,vgg_features,y_):
        """
        stores the hyperparameters as class attributes
        :param config: [dict] contains all the information for training/testing
        :param x_: [tensor] contains the image batch
        :param vgg_features: [tensor] contains vgg features for the current image batch
        :param y_:[tensor] contains the labels batch
        """
        self.batch_size = config["batch_size"]
        self.optimizer_ = config["optimizer"]
        self.lr = config["lr"]
        self.lambda_ = config["lambda_"]
        self.loss = config["loss"]
        self.devices = config["devices"]
        self.x = x_
        self.y = y_
        self.vgg_features = vgg_features
        self.train_bool = tf.placeholder(tf.bool)
        self.keep_probability = tf.placeholder(tf.float32)

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

    def reset_train_bool(self,inp):
        self.train_bool = inp

    def batch_norm(self,x,scope):
        #return tf.contrib.layers.batch_norm(x,trainable = self.train_bool,center = True,scale = True,scope = scope)
        return tf.layers.batch_normalization(x, training=self.train_bool)

    def conv_max_pool_layer(self,inp_,w,b,dp,name_scope):
        #conv = tf.nn.dropout(self.batch_norm(tf.nn.relu(tf.nn.bias_add(self.conv_2d(inp_, w),b)),name_scope),dp)
        conv = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(self.conv_2d(inp_,w),b)),dp)
        max_pool = self.max_pool(conv)
        return max_pool

    def inference(self):

        #all the variables stay in CPU
        with tf.device(self.devices[0]):
            with tf.name_scope("conv_1"):
                conv_1_w = self.weights_("conv_1w",[3,3,3,64])
                conv_1_b = self.biases_("conv_1b",[64])

            with tf.name_scope("conv_2"):
                conv_2_w = self.weights_("conv_2w",[3,3,64,128])
                conv_2_b = self.biases_("conv_2b",[128])

            with tf.name_scope("conv_3"):
                conv_3_w = self.weights_("conv_3w",[3,3,128,128])
                conv_3_b = self.biases_("conv_3b",[128])

            with tf.name_scope("full_1"):
                full_1_w = self.weights_("full_1w",[28*28*128,128])
                full_1_b = self.biases_("full_1b",[128])

            with tf.name_scope("full_2"):
                full_2_w = self.weights_("full_2w",[128,64])
                full_2_b = self.biases_("full_2b",[64])

            with tf.name_scope("final"):
                full_3_w = self.weights_("full_3w",[128,3])
                full_3_b = self.biases_("full_3b",[3])

            with tf.name_scope("vgg_features"):
                full_vgg_1_w = self.weights_("full_vgg_1w",[4096,64])
                full_vgg_1_b = self.biases_("full_vgg_1b",[64])

        #all the convolution and matrix multiplication happening in GPU
        with tf.device(self.devices[1]):
            with tf.name_scope("conv_1") as conv_1_scope:
                conv_1 = self.conv_max_pool_layer(self.x,conv_1_w,conv_1_b,self.keep_probability,conv_1_scope)
            with tf.name_scope("conv_2") as conv_2_scope:
                conv_2 = self.conv_max_pool_layer(conv_1,conv_2_w,conv_2_b,self.keep_probability,conv_2_scope)
            with tf.name_scope("conv_3") as conv_3_scope:
                conv_3 = self.conv_max_pool_layer(conv_2,conv_3_w,conv_3_b,self.keep_probability,conv_3_scope)

            reshaped_last_conv = tf.reshape(conv_3, (-1,28*28*128))

            #Fully connected layers
            full_1 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.matmul(reshaped_last_conv, full_1_w), full_1_b)),self.keep_probability)
            full_2 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.matmul(full_1, full_2_w), full_2_b)),self.keep_probability)

            #Concatenation
            vgg_full_1 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.matmul(self.vgg_features,full_vgg_1_w), full_vgg_1_b)),self.keep_probability)
            concatenated_feature = tf.concat([full_2,vgg_full_1],axis = 1)

            self.output = tf.nn.bias_add(tf.matmul(concatenated_feature,full_3_w),full_3_b)

        trainable_vars = tf.trainable_variables()
        self.regularized_loss = tf.add_n([tf.nn.l2_loss(i) for i in trainable_vars if "b" not in i.name],name = "regularized_loss")*self.lambda_
        self.unregularized_loss = tf.reduce_mean(self.loss(logits=self.output, labels=self.y))

        tf.summary.scalar("regularized_loss", self.regularized_loss)
        tf.summary.scalar("unregularized_loss", self.unregularized_loss)

        self.calculated_loss =  tf.add(self.unregularized_loss,self.regularized_loss)
        self.calculated_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(input=self.y,axis = 1),tf.argmax(input=self.output,axis = 1)),tf.float32))

        tf.summary.scalar("accuracy", self.calculated_acc)
        self.model_summary = tf.summary.merge_all()

        #for batch normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = self.optimizer_(self.lr).minimize(self.calculated_loss)

        return {"inputs":[self.x,self.y],"output":self.output,"optimizer":self.optimizer,\
                "acc":self.calculated_acc,"loss":self.calculated_loss,"summary":self.model_summary,\
                "labels":self.y,"predictions":self.output}
