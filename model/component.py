import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten,Dense,Lambda,BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Add, Multiply, Concatenate
from tensorflow.keras import backend as K

def conv_block(n_filter, x):
    
    x = Conv2D(n_filter, kernel_size=(3,3), padding='same', strides=2, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def input_attention(x,A,method):
    
    if method=="none":
        pass
    elif method=="add":
        x = Add()([x,A])
    elif method=="multiply":
        x = Multiply()([x,A])
    elif method=="attention":
        Ax = Multiply()([x,A])
        x = Add()([x,Ax])
    elif method=="4ch":
        x = Concatenate(axis=-1)([x,A])
    else:
        raise ValueError(f"Value Error!!: {method} is invalid method.")
    
    return x

def attention_distillation(A,distil_method):

    if distil_method == "none":
        pass
    elif distil_method == "max":
        A = Lambda(lambda x: K.max(x, axis=-1,keepdims=True))(A)
    elif distil_method == "avg":
        A = Lambda(lambda x: K.mean(x, axis=-1,keepdims=True))(A)
    elif distil_method == "conv":
        A = Conv2D(1, kernel_size=(1,1), padding='same', strides=1, kernel_initializer='he_normal')(A)
    else: 
        raise ValueError(f"Value Error!!: {distil_method} is invalid method.")
    
    return A

def attention_sigmoid(A, sigmoid_apply):

    if sigmoid_apply==True:
        A = Activation('sigmoid')(A)
    return A

def fusion_module(x,A,fusion_method,SE=False):

    if fusion_method=="none":
        pass
    elif fusion_method=="concat":
        x = Concatenate(axis=-1)([x,A])
        if SE:
            ("SE is applied")
            n_channel = x.shape[-1]
            x = se_block(input=x, channels=n_channel, r=8)
    elif fusion_method=="add":
        x = Add()([x,A])
    elif fusion_method=="multiply":
        x = Multiply()([x,A])
    elif fusion_method=="attention":
        Ax = Multiply()([x,A])
        x = Add()([x,Ax])
    else: 
        raise ValueError(f"Value Error!!: {fusion_method} is invalid method.")
    return x

def final_flat(x,flat_method):
    
    if flat_method == "flat":
        x = Flatten()(x)
    elif flat_method == "gap":
        x = GlobalAveragePooling2D()(x)
    else: 
        raise ValueError(f"Value Error!!: {flat_method} is invalid method.")
    return x

def output_block(x, A, output_method, flat_method, num_class):

    if A != None:
        if output_method=="separate":

            x = final_flat(x,flat_method)
            A = final_flat(A,flat_method)

            prob_x = Dense(num_class, activation='sigmoid')(x)
            prob_A = Dense(num_class, activation='sigmoid')(A)

            outputs = [prob_x, prob_A]

        elif output_method=="oneway":

            x = final_flat(x,flat_method)

            prob_x = Dense(num_class, activation='sigmoid')(x)

            outputs = [prob_x]
            
        elif output_method=="merge":
            
            output_filters = int(x.shape[-1])
            x = Concatenate(axis=-1)([x,A])
            x = Conv2D(output_filters, kernel_size=(3,3), padding='same', strides=1, kernel_initializer='he_normal')(x)

            x = final_flat(x,flat_method)
            prob_x = Dense(num_class, activation='sigmoid')(x)
            
            outputs = [prob_x]
        
        else:
            raise ValueError(f"Value Error!!: {output_method} is invalid method.")
    
    else:
        x = final_flat(x,flat_method)
        prob_x = Dense(num_class, activation='softmax')(x)
        outputs = [prob_x]

    return outputs

def se_block(input, channels, r=8):
    """ Squeeze and Excitation """
    # Squeeze
    x = GlobalAveragePooling2D()(input)
    # Excitation
    x = Dense(channels//r, activation="relu")(x)
    x = Dense(channels, activation="sigmoid")(x)
    return Multiply()([input, x])