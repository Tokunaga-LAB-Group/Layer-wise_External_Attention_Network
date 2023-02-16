import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Flatten, Activation
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model

from model.selfattention_module import SE,SRM,SimAM
from model.CAAN import *
from model.component import conv_block, attention_sigmoid, attention_distillation, fusion_module


def get_model(model_ADN, model_AAN, attention_points, img_shape, attMap_shape, 
                input_method='img_and_attMap', output_method='AAN_and_ADN',
                add_module=None,f_jct='avg', jct_method='attention'):
    # Set random seed.
    # tf.random.set_seed(0)

    # Get num of inputs.
    n_inputs = int(np.where(
        input_method=='img_and_attMap',
        2,      # True
        1       # False
    ))
    # Get num of outputs.
    n_outputs = int(np.where(
        output_method=='AAN_and_ADN',
        2,      # True
        1       # False
    ))
    # Get attention_points.
    if 'None' in attention_points:
        att_points = []
    else:
        att_points = [int(a_p) for a_p in attention_points]
    
    if model_ADN=="VGG16":
        
        model = VGG16(
            self_attention_module = add_module,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            img_shape=img_shape,
            att_shape=attMap_shape,
            att_points= att_points,
            input_method =input_method,
            distil_method=f_jct, 
            sigmoid_apply=True, 
            fusion_method=jct_method, 
            att_base_model=model_AAN
        ).build()

    elif model_ADN=="ResNet18":
        
        model=ResNet18(
            self_attention_module=add_module,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            img_shape=img_shape,
            att_shape=attMap_shape,
            att_points=att_points,
            input_method =input_method,
            distil_method=f_jct, 
            sigmoid_apply=True, 
            fusion_method=jct_method, 
            att_base_model=model_AAN
        ).build()
    
    elif model_ADN=="CNN":
        model=CNN(
            self_attention_module=add_module,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            img_shape=img_shape,
            att_shape=attMap_shape,
            att_points=att_points,
            input_method =input_method,
            distil_method=f_jct, 
            sigmoid_apply=True, 
            fusion_method=jct_method, 
            att_base_model=model_AAN
        ).build()

    else:
        raise ValueError(f"Model {model_ADN} is invalid.")

    return model

class AttPointWeight(layers.Layer):
    def __init__(self, name):
        super().__init__(name=name)
        self.w = self.add_weight('attP_weight', shape=(1), initializer='zeros')
        
    def call(self, inputs):
        outputs = self.w * inputs

        return outputs

class ModelWithAttentionBranchBase:
    def __init__(self, att_shape = (256,256,1),
                    att_points = [],
                    base_model = "MobileNet",
                    input_method  = "none",
                    distil_method = "none",
                    sigmoid_apply = False, 
                    fusion_method = "none"
                ):
        
        self.att_shape = att_shape
        self.att_points = {point:False for point in range(5)}
        for i in att_points:
            assert i in self.att_points.keys(),f"Attention points {i} is invalid."
            if i in self.att_points.keys():
                self.att_points[i]=True
                
        # Base CAAN
        if base_model=="MobileNet":
            self.Att_Model = AttentionBranchNet_MobileNet(shape=att_shape).build()
        elif base_model=="ResNet":
            self.Att_Model = AttentionBranchNet_ResNet(shape=att_shape).build()
        elif base_model=="Direct":
            self.Att_Model = AttentionBranchNet_Attmap(shape=att_shape, first_compress_size=256).build()
        else:
            ValueError("Cannot find the base AAN.")

        self.att_input = self.Att_Model.inputs
        self.att_pred, *self.As = self.Att_Model.outputs

        self.input_method  = input_method  # "attention, "add", "multiply", "none", "4ch"
        self.distil_method = distil_method # "max", "avg", "conv", "none"
        self.sigmoid_apply = sigmoid_apply # True, False
        self.fusion_method = fusion_method # "attention, "concat", "add", "multiply", "none"

        # Attention Point Weights
        self.attP_weights = {point: AttPointWeight('attP_w_'+str(point)) for point in range(5)}
        
    def BranchConnection(self, x, position):
            
        if self.att_points[position]:

            A = self.As[position]
            A = attention_distillation(A, self.distil_method)
            A = attention_sigmoid(A, self.sigmoid_apply)
            A = self.attP_weights[position](A)
            x = fusion_module(x, A, self.fusion_method,SE=True)

        return x


class ResNet18(ModelWithAttentionBranchBase):

    def __init__(self, img_shape = (256,256,3),
                       att_shape = (256,256,1),
                       n_inputs = 1,
                       n_outputs = 1,
                       att_points = [],
                       att_base_model = "MobileNet",
                       input_method  = "none",
                       distil_method = "none",
                       sigmoid_apply = False, 
                       fusion_method = "none",
                       output_method = "separate",
                       self_attention_module="None"
                ):
        super(ResNet18,self).__init__(att_shape,att_points,att_base_model,input_method,distil_method,sigmoid_apply,fusion_method)
        self.img_shape = img_shape
        self.self_attention_module = self_attention_module
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
    
    def build(self):        
        
        num_filters = 64
        num_blocks = 4
        num_sub_blocks = 2

        input_ = Input(shape=self.img_shape, name="ADN_Inputs")
        x = input_
        
        x = Conv2D(filters=num_filters, kernel_size=(7,7), padding='same', strides=2, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        if self.n_inputs==2:
            x = self.BranchConnection(x,position=0) # 128 out
        x = Conv2D(num_filters, kernel_size=3, padding='same', strides=1, kernel_initializer='he_normal')(x)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

        n_position=1
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

                if (self.self_attention_module == "None") or (self.self_attention_module == None):
                    pass
                elif self.self_attention_module == "SE":
                    y = SE(y)
                elif self.self_attention_module == "SRM":
                    y = SRM(y)
                elif self.self_attention_module == "SimAM":
                    y = SimAM(y)
                else:
                    raise NotImplementedError
                
                # Skip structure
                if is_first_layer_but_not_first_block:
                    x = Conv2D(num_filters, kernel_size=1, padding='same', strides=2, kernel_initializer='he_normal')(x)
                
                x = Add()([x, y])
                x = Activation('relu')(x)

                # last of sub block 
                if j==num_sub_blocks-1 and self.n_inputs==2:
                    x = self.BranchConnection(x,position=n_position) # 64,32,16,8 out
                    # x = Conv2D(num_filters, kernel_size=3, padding='same', strides=1, kernel_initializer='he_normal')(x)
                    n_position+=1

            num_filters *= 2

        x    = GlobalAveragePooling2D()(x)
        pred = Dense(2, activation='softmax',name="ADN")(x)
        
        if self.n_inputs==2:
            inputs = [input_, self.att_input]
        elif self.n_inputs==1:
            inputs = [input_]

        if self.n_outputs==2:
            outputs = [pred, self.att_pred]
        elif self.n_outputs==1:
            outputs = [pred]
            
        model = Model(inputs=inputs, outputs=outputs)
        
        return model  


class VGG16(ModelWithAttentionBranchBase):

    def __init__(self, img_shape=(256,256,3),
                       att_shape=(256,256,1),
                       n_inputs = 1,
                       n_outputs = 1,
                       att_points=[],
                       att_base_model="MobileNet",
                       input_method  = "none",
                       distil_method = "none",
                       sigmoid_apply = False, 
                       fusion_method = "none",
                       output_method = "separate"
                ):

        super(VGG16,self).__init__(att_shape,att_points,att_base_model,input_method,distil_method,sigmoid_apply,fusion_method,output_method)
        self.img_shape = img_shape
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
    def build(self):        
        
        n_filters=[64*1,64*2,64*4,64*8,64*8]
        
        input_ = Input(shape=self.img_shape, name="ADN_Inputs")
        x = input_

        x = Conv2D(filters=n_filters[0], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[0], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
        x = self.BranchConnection(x,position=0)
        
        x = Conv2D(filters=n_filters[1], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[1], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
        x = self.BranchConnection(x,position=1)
        
        x = Conv2D(filters=n_filters[2], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[2], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[2], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
        x = self.BranchConnection(x,position=2)
        
        x = Conv2D(filters=n_filters[3], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[3], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[3], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
        x = self.BranchConnection(x,position=3)
        
        x = Conv2D(filters=n_filters[3], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[3], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = Conv2D(filters=n_filters[3], kernel_size=(3,3), padding='same', strides=1, activation="relu", kernel_initializer='he_normal')(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
        x = self.BranchConnection(x,position=4)
        
        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dense(4096, activation="relu")(x)
        x = Dense(2, activation='softmax',name="ADN")(x)
        pred = x
        
        if self.n_inputs==2:
            inputs = [input_, self.att_input]
        elif self.n_inputs==1:
            inputs = [input_]

        if self.n_outputs==2:
            outputs = [pred, self.att_pred]
        elif self.n_outputs==1:
            outputs = [pred]
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model    


class CNN(ModelWithAttentionBranchBase):

    def __init__(self, img_shape=(256,256,3),
                       att_shape=(256,256,1),
                       n_inputs = 1,
                       n_outputs = 1,
                       att_points=[],
                       att_base_model="MobileNet",
                       input_method  = "none",
                       distil_method = "none",
                       sigmoid_apply = False, 
                       fusion_method = "none",
                       output_method = "separate"
                ):
        super(CNN,self).__init__(att_shape,att_points,att_base_model,input_method,distil_method,sigmoid_apply,fusion_method,output_method)

        self.img_shape = img_shape
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.filters = [64,64*2,64*4,64*8,64*8]
    
    def build(self):    

        input_ = Input(shape=self.img_shape, name="ADN_Inputs")
        x = input_

        for i in range(len(self.filters)):
            x = conv_block(self.filters[i],x)
            x = self.BranchConnection(x,position=i)

        x = Flatten()(x)
        pred = Dense(2, activation='softmax')(x)

        if self.n_inputs==2:
            inputs = [input_, self.att_input]
        elif self.n_inputs==1:
            inputs = [input_]

        if self.n_outputs==2:
            outputs = [pred, self.att_pred]
        elif self.n_outputs==1:
            outputs = [pred]
            
        model = Model(inputs=inputs, outputs=outputs)

        return model    