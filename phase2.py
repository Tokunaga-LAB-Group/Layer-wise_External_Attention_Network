import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter('ignore')

import gc
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from natsort import natsorted

# TF/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing as pre_layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, losses, callbacks

# Custom
from model.LEANet import get_model
from data_loader import ClassificationDataLoader


# Parser
def getArguments():
    parser = argparse.ArgumentParser(description='LEA-Net')
    parser.add_argument('--GPU', default=1, type=int)
    parser.add_argument('--save_path', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)

    parser.add_argument('--n_splits', default=10, type=int)

    parser.add_argument('--input_method',
        choices=['img_and_attMap', 'img_only', 'attMap_only', '4_channel', 'attMap_to_img', 'multiply'],
        default='img_and_attMap')
    parser.add_argument('--output_method',
        choices=['ADN_only', 'AAN_and_ADN'],
        default='AAN_and_ADN')
    parser.add_argument('--img_size', default=256, type=int)

    parser.add_argument('--model_ADN', choices=['VGG16', 'ResNet18', 'CNN'], default='ResNet18')
    parser.add_argument('--additional_module', choices=['None', 'SE', 'SimAM', 'SRM'], default='None')
    parser.add_argument('--model_AAN', choices=['MobileNet', 'ResNet', 'Direct'], default='MobileNet')
    parser.add_argument('--attention_points', choices=['None','0','1','2','3','4'], nargs='*')

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-04, type=float)

    return parser.parse_args()

# Main
def main():
    # Make save path.
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    print(">>> Start: k-fold cross validation.")
    for k in range(int(args.n_splits)):
        print("===============================================================")
        print("*** {} / {} ***".format(k+1, args.n_splits))
        print("===============================================================")
        # Make save dir.
        save_dir = os.path.join(save_path, str(k+1).zfill(2))
        os.makedirs(save_dir, exist_ok=True)

        # Get data.
        ## Label: one-hot-encoding
        dataset_dir = os.path.join(args.dataset, str(k+1).zfill(2))
        Y = np.load(os.path.join(dataset_dir, 'Label.npz'))
        Y_train, Y_test = Y['train'], Y['test']

        label_train, label_test = to_categorical(Y_train, num_classes=2), to_categorical(Y_test, num_classes=2)

        print(">>> Train: Positive={}, Negative={}".format(np.sum(Y_train==1), np.sum(Y_train==0)))
        print(">>> Test : Positive={}, Negative={}".format(np.sum(Y_test==1), np.sum(Y_test==0)))

        ## Files from result of phase1.
        GT_train_list = natsorted(glob.glob(os.path.join(dataset_dir, 'GT', 'Train', '*.*')))          # Train image in phase2
        GT_test_list = natsorted(glob.glob(os.path.join(dataset_dir, 'GT', 'Test', '*.*')))            # Test image in phase2
        Anomap_train_list = natsorted(glob.glob(os.path.join(dataset_dir, 'Anomap', 'Train', '*.*')))  # Train attention map in phase2
        Anomap_test_list = natsorted(glob.glob(os.path.join(dataset_dir, 'Anomap', 'Test', '*.*')))    # Test attention map in phase2

        # Make data genarator.
        ## Image data: resize -> nomalize -> standardize
        pre_transform = Sequential([
            pre_layers.Resizing(args.img_size, args.img_size),
            pre_layers.Rescaling(scale=1./255.)
        ])

        ## train data generator.
        train_datagen = ClassificationDataLoader(
            img_list=GT_train_list,
            attMap_list=Anomap_train_list,
            label=label_train,
            input_method=args.input_method,
            output_method=args.output_method,
            pre_transform=pre_transform,
            batch_size=args.batch_size
        )

        # Make model.
        ## input img shape. Here, we fix to RGB.
        img_shape = (args.img_size, args.img_size, 3)
        attMap_shape = (args.img_size, args.img_size, 1)

        model = get_model(
            model_ADN=args.model_ADN, model_AAN=args.model_AAN, 
            attention_points=args.attention_points,
            img_shape=img_shape, attMap_shape=attMap_shape,
            input_method=args.input_method, output_method=args.output_method,
            add_module=args.additional_module, f_jct='avg', jct_method='attention'
        )

        # Compile model.
        optimizer = optimizers.Adam(args.lr)
        loss = losses.BinaryCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # Train model.
        es = callbacks.EarlyStopping(monitor='loss', min_delta=0.0, patience=20)

        history = model.fit(
            train_datagen,
            epochs=args.epochs,
            steps_per_epoch=train_datagen.__len__(),
            verbose=2,
            callbacks=[es]
        )

        # Save history.
        save_csv_path = os.path.join(save_dir, 'phase2_history.csv')
        pd.DataFrame(history.history).to_csv(save_csv_path)

        # Evaluation
        print(">>> Evaluate Test...")
        ## Get test generator.
        test_datagen = ClassificationDataLoader(
            img_list=GT_test_list,
            attMap_list=Anomap_test_list,
            label=label_test,
            input_method=args.input_method,
            output_method=args.output_method,
            pre_transform=pre_transform,
            batch_size=len(GT_test_list)
        )

        ## Predict all images.
        for data in test_datagen:
            ### Get data.
            inputs, _ = data
            ### Predict.
            pred_list = model.predict_on_batch(inputs)
        
        ## Get num of output.
        N_output = len(model.output_shape)

        ## Show and Save evaluation.
        if N_output==2:
            preds_ADN, preds_AAN = pred_list        # (batch_size, 2), (batch_size, 2)
            Y_ADN, Y_AAN = np.argmax(preds_ADN, axis=(1)), np.argmax(preds_AAN, axis=(1))
            
            print("\n========== ADN (Anomaly Detection Network) ==========\n")
            print(f"Confusion Matrix \n{confusion_matrix(Y_test, Y_ADN)}")
            print(f"Multiple Scores \n{classification_report(Y_test, Y_ADN)}")
            print(f"F score = {f1_score(Y_test,Y_ADN)}")
            
            print("\n========== AAN (Anomaly Attention Network) ==========\n")
            print(f"Confusion Matrix \n{confusion_matrix(Y_test, Y_AAN)}")
            print(f"Multiple Scores \n{classification_report(Y_test, Y_AAN)}")
            print(f"F score = {f1_score(Y_test,Y_AAN)}")

            save_result_path = os.path.join(save_dir, 'result.csv')
            label_df = pd.DataFrame(columns=['pred_ADN', 'pred_AAN', 'true'])
            label_df['pred_ADN'] = Y_ADN
            label_df['pred_AAN'] = Y_AAN
            label_df['true'] = Y_test
            label_df.to_csv(save_result_path)
        else:
            preds_ADN= pred_list        # (batch_size, 2)
            Y_ADN = np.argmax(preds_ADN, axis=(1))
            
            print("\n========== ADN (Anomaly Detection Network) ==========\n")
            print(f"Confusion Matrix \n{confusion_matrix(Y_test, Y_ADN)}")
            print(f"Multiple Scores \n{classification_report(Y_test, Y_ADN)}")
            print(f"F score = {f1_score(Y_test,Y_ADN)}")
            
            save_result_path = os.path.join(save_dir, 'result.csv')
            label_df = pd.DataFrame(columns=['pred_ADN', 'true'])
            label_df['pred_ADN'] = Y_ADN
            label_df['true'] = Y_test
            label_df.to_csv(save_result_path)

        del model
        gc.collect()
        
    print(">>> End...")


if __name__=='__main__':
    args = getArguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    print(args)
    main()