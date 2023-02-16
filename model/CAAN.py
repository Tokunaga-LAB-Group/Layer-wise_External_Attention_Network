import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Flatten, Activation, Reshape, Dropout
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Input, Add, Multiply, Concatenate,DepthwiseConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers.experimental.preprocessing import Resizing

class AttentionBranchNet_MobileNet(tf.keras.Model):
    """ MobileNetV3 Model
    https://github.com/xiaochus/MobileNetV3/blob/master/model/mobilenet_base.py
    https://github.com/xiaochus/MobileNetV3/blob/master/model/mobilenet_v3_small.py
    """
    
    def __init__(self, shape, alpha=1.0, include_top=True):
        self.include_top = include_top
        self.shape = shape
        self.n_class = 2
        self.alpha = alpha
        
    def _relu6(self, x):
        """Relu 6
        """
        return K.relu(x, max_value=6.0)

    def _hard_swish(self, x):
        """Hard swish
        """
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _return_activation(self, x, nl):
        """Convolution Block
        This function defines a activation choice.
        # Arguments
            x: Tensor, input tensor of conv layer.
            nl: String, nonlinearity activation type.
        # Returns
            Output tensor.
        """
        # if nl == 'HS':
        #     x = Activation(self._hard_swish)(x)
        # if nl == 'RE':
        #     x = Activation(self._relu6)(x)
        x = Activation("relu")(x)
        return x

    def _conv_block(self, inputs, filters, kernel, strides, nl):

            x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
            x = BatchNormalization()(x)

            return self._return_activation(x, nl)

    def _squeeze(self, inputs):
        
        input_channels = int(inputs.shape[-1])

        x = GlobalAveragePooling2D()(inputs)
        x = Dense(input_channels, activation='relu')(x)
        x = Dense(input_channels, activation='hard_sigmoid')(x)
        x = Reshape((1, 1, input_channels))(x)
        x = Multiply()([inputs, x])

        return x

    def _bottleneck(self, inputs, filters, kernel, e, s, squeeze, nl):

        input_shape = K.int_shape(inputs)

        tchannel = int(e)
        cchannel = int(self.alpha * filters)

        r = s == 1 and input_shape[3] == filters

        x = self._conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = self._return_activation(x, nl)

        if squeeze:
            x = self._squeeze(x)

        x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)

        if r:
            x = Add()([x, inputs])

        return x
        
    def build(self, plot=False):
        
        inputs = Input(shape=self.shape, name="AAN_Inputs")
        
        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS') # 128 out
        x1 = x
        
        x = self._bottleneck(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE') # 64 out
        x2 = x
        
        x = self._bottleneck(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE') # 32 out
        x = self._bottleneck(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE')
        x3 = x
        
        x = self._bottleneck(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='HS') # 16 out
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS') 
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=120, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=144, s=1, squeeze=True, nl='HS')
        x4 = x
        
        x = self._bottleneck(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='HS') # 8
        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x5 = x
        
        x = self._conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
        x = GlobalAveragePooling2D()(x)

        x = Reshape((1, 1, 576))(x)
        x = Conv2D(1280, (1, 1), padding='same')(x)
        x = self._return_activation(x, 'HS')

        if self.include_top:
            x = Conv2D(self.n_class, (1, 1), padding='same', activation='sigmoid')(x) # softmax
            x = Reshape((self.n_class,),name="AAN")(x)
        
        out = x # classification prediction

        outputs = [out,x1,x2,x3,x4,x5]        
        model = Model(inputs, outputs)

        if plot:
            plot_model(model, to_file='AttentionBranch.png', show_shapes=True)

        return model


class AttentionBranchNet_ResNet(tf.keras.Model):
    def __init__(self, shape=(256,256,1)):
        # super().__init__()
        self.img_shape = shape
    
    def build(self):        
        
        outputs=[]
        
        num_filters = 64
        num_blocks = 4
        num_sub_blocks = 1

        inputs = Input(shape=self.img_shape, name="AAN_Inputs")
        x = inputs
        
        x = Conv2D(filters=num_filters, kernel_size=(7,7), padding='same', strides=2, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        outputs.append(x)
        
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='block2_pool_att')(x)

        for i in range(num_blocks):
            for j in range(num_sub_blocks):
                
                strides=1
                
                is_first_layer_but_not_first_block=False
                if j==0 and i>0:
                    is_first_layer_but_not_first_block=True
                    strides=2

                y = Conv2D(num_filters, kernel_size=3, padding='same', strides=strides, kernel_initializer='he_normal')(x)
                y = BatchNormalization()(y)
                y = Activation('relu')(y)
                y = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(y)
                y = BatchNormalization()(y)
                
                # Skip structure
                if is_first_layer_but_not_first_block:
                    x = Conv2D(num_filters, kernel_size=1, padding='same', strides=2, kernel_initializer='he_normal')(x)
                x = Add()([x, y])
                x = Activation('relu')(x)

                # last of sub block 
                if j==num_sub_blocks-1:
                    outputs.append(x)

            num_filters *= 2

        x    = GlobalAveragePooling2D()(x)
        pred = Dense(2, activation='softmax',name="AAN")(x)
        
        outputs.insert(0,pred)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model    


class AttentionBranchNet_Attmap(tf.keras.Model):
    def __init__(self, shape=(256,256,1), first_compress_size=256):
        self.img_shape = shape
        self.first_compress_size = first_compress_size
    
    def build(self):        
        
        interpolation_method = "bilinear"
        outputs=[]
        
        inputs = Input(shape=self.img_shape, name="AAN_Inputs")

        x = Resizing(self.first_compress_size, self.first_compress_size, interpolation=interpolation_method)(inputs)

        # x1 = Resizing(256, 256, interpolation=interpolation_method)(x)
        x1 = Resizing(128, 128, interpolation=interpolation_method)(x)
        x2 = Resizing(64, 64, interpolation=interpolation_method)(x1)
        x3 = Resizing(32, 32, interpolation=interpolation_method)(x2)
        x4 = Resizing(16, 16, interpolation=interpolation_method)(x3)
        x5 = Resizing(8, 8, interpolation=interpolation_method)(x4)

        outputs = [x5,x1,x2,x3,x4,x5]
        model = Model(inputs=inputs, outputs=outputs)
        
        return model    


