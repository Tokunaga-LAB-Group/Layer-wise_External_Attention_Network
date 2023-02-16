import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter('ignore')

import gc
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from natsort import natsorted

# TF/Keras
from tensorflow.keras import optimizers, losses, callbacks

# Custom
from data_loader import DataGenerator
from utils import make_anomalymap, save_raw_image
from model.generative_models import get_model

# Parser
def getArguments():
    parser = argparse.ArgumentParser(description='LEA-Net: phase1')
    parser.add_argument('--GPU', default=1, type=int)
    parser.add_argument('--save_path', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--task', 
        choices=[
            'reconstruction', 'denoise', 'self-resolution', 
            'colorization', 'rotation', 'inpaint'
        ], required=True)
    parser.add_argument('--n_splits', default=10, type=int)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--model', choices=['AE', 'SegNet', 'Unet'], required=True)
    parser.add_argument('--loss', choices=['BCE', 'MSE'])
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-04, type=float)

    return parser.parse_args()

# Main
def main():
    # Make save path.
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    # Get daata.
    # Dataset___Positive (1): Normal
    #         |_Negative (0): Abnormal
    ## Positive.
    X_P = natsorted(glob.glob(
        os.path.join(args.dataset, 'Positive', '**', '*.*'),
        recursive=True
    ))
    Y_P = np.ones((len(X_P)), np.float32)
    ## Negative.
    X_N = natsorted(glob.glob(
        os.path.join(args.dataset, 'Negative', '**', '*.*'),
        recursive=True
    ))
    Y_N = np.zeros((len(X_N)), np.float32)

    X = np.concatenate([X_P, X_N], axis=(0))
    Y = np.concatenate([Y_P, Y_N], axis=(0))

    ## Shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    Y = Y[perm]

    print("\n")
    print(">>> X_P:", len(X_P))
    print(">>> X_N:", len(X_N))
    print(">>> Task :", args.task)
    print(">>> Model:", args.model)
    print(">>> loss :", args.loss)


    k_fold = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=False
    )

    print(">>> Start: k-fold cross validation.")
    for k, index_list in enumerate(k_fold.split(X, Y)):
        print("===============================================================")
        print("*** {} / {} ***".format(k+1, args.n_splits))
        print("===============================================================")

        # Make save dir of k.
        save_dir = os.path.join(save_path, str(k+1).zfill(2))
        os.makedirs(save_dir, exist_ok=True)

        train_index, test_index = index_list
        print(">>> Train:", len(train_index))
        print(">>> Test :", len(test_index))

        # Train Phase
        ## Get train data.
        X_train, Y_train = X[train_index], Y[train_index]
        ## Get negative data in train.
        X_train_N = X_train[Y_train==0]

        ## Make train generator for unsupervised Learning.
        train_datagen = DataGenerator(
            img_files=X_train_N, task=args.task,
            target_size=(args.img_size, args.img_size), batch_size=args.batch_size
        )

        ## Get model.
        model = get_model(args.model, input_shape=(args.img_size, args.img_size, 3))

        ## Get loss, optimizer and callbacks.
        if args.loss=='BCE':
            loss = losses.BinaryCrossentropy()
        elif args.loss=='MSE':
            loss = losses.MeanAbsoluteError()
        
        optimizer = optimizers.Adam(learning_rate=args.lr)

        es = callbacks.EarlyStopping(monitor='loss', patience=20)

        ## Compile model.
        model.compile(optimizer=optimizer, loss=loss)

        ## Train model.
        history = model.fit(
            train_datagen, epochs=args.epochs,
            steps_per_epoch=train_datagen.__len__(),
            verbose=2, callbacks=[es]
        )

        ## Save model.
        save_model_path = os.path.join(save_dir, 'phase1_model.h5')
        model.save(save_model_path, include_optimizer=False)

        ## Save history
        save_csv_path =  os.path.join(save_dir, 'phase1_history.csv')
        pd.DataFrame(history.history).to_csv(save_csv_path)

        ## Save model's outputs of train for AAN (Anomaly Attention Network).
        ### Make directories of train's result.
        dir_name_list = ['Input', 'Reconst', 'Anomap', 'GT']
        for dir_name in dir_name_list:
            os.makedirs(os.path.join(save_dir, dir_name, 'Train'), exist_ok=True)

        ### Make train generator for AAN.
        train_datagen = DataGenerator(
            img_files=X_train, task=args.task,
            target_size=(args.img_size, args.img_size), batch_size=args.batch_size
        )

        print(">>> Evaluate Train...")
        for i, data in enumerate(train_datagen):
            X_batch_dash, X_batch = data 

            ### Make reconsts and anomaps.
            reconsts = model.predict_on_batch(X_batch_dash)
            anomaps = make_anomalymap(reconsts, X_batch, args.task)

            ### Save result.
            ### 'Input', 'Reconst', 'Anomap', 'GT'
            result_list = [X_batch_dash, reconsts, anomaps, X_batch]
            cmap_list = ['viridis', 'viridis', 'gray', 'viridis']
            for dir_name, result, cmap in zip(dir_name_list, result_list, cmap_list):
                save_raw_image(
                    result, i*args.batch_size,
                    os.path.join(save_dir, dir_name, 'Train'),
                    cmap=cmap
                )
        
        # Test phase
        ## Get test data.
        X_test, Y_test = X[test_index], Y[test_index]
        ## Make directories of test's result.
        dir_name_list = ['Input', 'Reconst', 'Anomap', 'GT']
        for dir_name in dir_name_list:
            os.makedirs(os.path.join(save_dir, dir_name, 'Test'), exist_ok=True)
        
        ## Make test generator for AAN.
        test_datagen = DataGenerator(
            img_files=X_test, task=args.task,
            target_size=(args.img_size, args.img_size), batch_size=args.batch_size
        )

        print(">>> Evaluate Test...")
        for i, data in enumerate(test_datagen):
            X_batch_dash, X_batch = data    # [前処理済み画像のバッチ, 画像のバッチ]

            ### Make reconsts and anomaps.
            reconsts = model.predict_on_batch(X_batch_dash)
            anomaps = make_anomalymap(reconsts, X_batch, args.task)

            ### Save result.
            ### 'Input', 'Reconst', 'Anomap', 'GT'
            result_list = [X_batch_dash, reconsts, anomaps, X_batch]
            cmap_list = ['viridis', 'viridis', 'gray', 'viridis']
            for dir_name, result, cmap in zip(dir_name_list, result_list, cmap_list):
                save_raw_image(
                    result, i*args.batch_size,
                    os.path.join(save_dir, dir_name, 'Test'),
                    cmap=cmap
                )
        
        # Save ground truth labels of train and test.
        np.savez(os.path.join(save_dir, 'Label'), train=Y_train, test=Y_test)

        del model
        gc.collect()
        
    print(">>> End...")

if __name__=='__main__':
    args = getArguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    print(args)
    main()