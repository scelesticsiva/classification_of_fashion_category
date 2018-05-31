"""
@author - Sivaramakrishnan
"""
import tensorflow as tf

class simple_vgg_features_model(object):
    def __init__(self,config,x_,vgg_features,y_):
        self.epochs = config["epochs"]
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
        conv = tf.nn.dropout(self.batch_norm(tf.nn.relu(tf.nn.bias_add(self.conv_2d(inp_, w),b)),name_scope),dp)
        max_pool = self.max_pool(conv)
        return max_pool
    def full_connected_layer(self,inp_,w,b,dp,name_scope):
        return tf.nn.dropout(self.batch_norm(tf.nn.relu(tf.nn.bias_add(tf.matmul(inp_,w),b)),name_scope),dp)

    def inference(self):

        with tf.device(self.devices[0]):
            with tf.name_scope("full_1"):
                full_1_w = self.weights_("full_1w",[4096,1024])
                full_1_b = self.biases_("full_1b",[1024])

            with tf.name_scope("full_2"):
                full_2_w = self.weights_("full_2w",[1024,512])
                full_2_b = self.biases_("full_2b",[512])

            with tf.name_scope("full_3"):
                full_3_w = self.weights_("full_3w",[512,128])
                full_3_b = self.biases_("full_3b",[128])

            with tf.name_scope("full_4"):
                full_4_w = self.weights_("full_4w",[128,3])
                full_4_b = self.biases_("full_4b",[3])

        with tf.device(self.devices[1]):
            with tf.name_scope("full_1") as full_1_scope:
                full_1 = self.full_connected_layer(self.vgg_features,full_1_w,full_1_b,self.keep_probability,full_1_scope)
            with tf.name_scope("full_2") as full_2_scope:
                full_2 = self.full_connected_layer(full_1,full_2_w,full_2_b,self.keep_probability,full_2_scope)
            with tf.name_scope("full_3") as full_3_scope:
                full_3 = self.full_connected_layer(full_2,full_3_w,full_3_b,self.keep_probability,full_3_scope)
            with tf.name_scope("full_4") as full_4_scope:
                self.output = tf.nn.bias_add(tf.matmul(full_3,full_4_w),full_4_b)

        trainable_vars = tf.trainable_variables()
        self.regularized_loss = tf.add_n([tf.nn.l2_loss(i) for i in trainable_vars if "b" not in i.name],name = "regularized_loss")*self.lambda_

        self.calculated_loss = tf.reduce_mean(self.loss(logits=self.output, labels=self.y)) + self.regularized_loss
        self.calculated_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(input=self.y,axis = 1),tf.argmax(input=self.output,axis = 1)),tf.float32))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = self.optimizer_(self.lr).minimize(self.calculated_loss)

        return {"inputs":[self.x,self.y],"output":self.output,"optimizer":self.optimizer,"acc":self.calculated_acc,"loss":self.calculated_loss}