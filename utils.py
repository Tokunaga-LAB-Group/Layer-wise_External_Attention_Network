import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import deltaE_ciede2000

def data_shuffle(X,Y=[]):
    ''' Shuffle data
    '''   
    np.random.seed(0)

    if len(Y)>0:
        assert len(X)==len(Y), "length of X and Y is not same."
        p = np.random.permutation(len(X))
        return X[p],Y[p]
    else:
        p = np.random.permutation(len(X))
        return X[p]

def plot_history_clf(history, save_path, n_outputs):
    
    if n_outputs==1:
        # Loss
        plt.figure()
        plt.plot(history["loss"])
        # plt.plot(history["val_loss"])
        plt.title('Binary Crossentropy')
        plt.savefig(os.path.join(save_path,f"history_binary_crossentropy.jpg"))
        plt.show()
        
        plt.figure()
        plt.plot(history["accuracy"])
        # plt.plot(history["val_mse"])
        plt.title('accuracy')
        plt.savefig(os.path.join(save_path,f"history_accuracy.jpg"))
        plt.show()

    elif n_outputs==2:
        # Loss
        plt.figure()
        plt.plot(history["main_pred_loss"],label="ADN")
        plt.plot(history["att_pred_loss"],label="CAAN")
        plt.title('Binary Crossentropy')
        plt.legend()
        plt.savefig(os.path.join(save_path,f"history_binary_crossentropy.jpg"))
        plt.show()
        
        plt.figure()
        plt.plot(history["main_pred_accuracy"],label="ADN")
        plt.plot(history["att_pred_accuracy"],label="CAAN")
        plt.title('accuracy')
        plt.legend()
        plt.savefig(os.path.join(save_path,f"history_accuracy.jpg"))
        plt.show()
        
        
    plt.clf(),plt.close()
    return

def plot_history(history, save_path):
    
    # Loss
    plt.figure()
    plt.plot(history["loss"])
    # plt.plot(history["val_loss"])
    plt.title('Binary Crossentropy')
    plt.savefig(os.path.join(save_path,f"history_binary_crossentropy.jpg"))
    plt.show()
    
    plt.figure()
    plt.plot(history["mse"])
    # plt.plot(history["val_mse"])
    plt.title('MSE')
    plt.savefig(os.path.join(save_path,f"history_mse.jpg"))
    plt.show()
    
    plt.clf(),plt.close()
    return

def save_raw_image(Imgs, number, save_path, cmap='viridis', vmin=0.0, vmax=1.0):
    for i,img in enumerate(Imgs):
        plt.imsave(f"{save_path}/{str(number+i).zfill(3)}.jpg",img, cmap=cmap, vmin=vmin, vmax=vmax)
    return

def make_anomalymap(X_pred,Y,task):

    if task=="colorization":
        X_ano = ciede2000(X_pred,Y) # CIEDE2000
    else:
        X_ano = l1(X_pred,Y)
        
    # return (X_train,X_test), (X_train_att,X_test_att), (X_train_color,X_test_color)
    return X_ano

def ciede2000(X_cld,X_org,params=(1,1,1)):
    ''' Calculate ciede2000 between original image and pix2pix colored image, and make the heatmap
    
        Args:
            X_org: Array of RGB Images
            X_cld: Array of RGB Images colored
        Return:
            X_cie: Array of CIEDE2000 heatmap
    '''
    X_cie = []
    for i in range(len(X_org)):
        # CIEDE2000
        anomalymap = deltaE_ciede2000(X_org[i],X_cld[i], kL=params[0], kC=params[1], kH=params[2])
        X_cie.append(anomalymap)
    X_cie = np.array(X_cie)

    return X_cie

def l1(X,Y):
    """ l1 distance """
    # return np.square(X-Y)
    return np.mean(np.abs(X-Y),axis=-1)

def exclude_positive(X_rgb,X_att,Y,rate_P):
    X_rgb = np.array(X_rgb)
    X_att = np.array(X_att)
    
    len_P = len(Y[Y==1])

    X_rgb_P = X_rgb[Y==1]
    X_rgb_N = X_rgb[Y==0]
    X_att_P = X_att[Y==1]
    X_att_N = X_att[Y==0]

    X_rgb_P_use = X_rgb_P[:int(len_P*rate_P)]
    X_att_P_use = X_att_P[:int(len_P*rate_P)]

    X_rgb_new = np.concatenate([X_rgb_N, X_rgb_P_use])
    X_att_new = np.concatenate([X_att_N, X_att_P_use])
    Y_new = np.concatenate([np.zeros(len(X_rgb_N)),np.ones(len(X_rgb_P_use))])
    
    p = np.random.permutation(len(X_rgb_new))
    X_rgb_new, X_att_new, Y_new = X_rgb_new[p], X_att_new[p], Y_new[p]
    
    return X_rgb_new, X_att_new, Y_new