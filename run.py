from __future__ import division, print_function
import keras
from  keras.datasets import mnist
import deeplift
from deeplift.layers import NonlinearMxtsMode
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.util import compile_func
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from deeplift.util import get_integrated_gradients_function
from collections import OrderedDict
import time
import pickle
from util import *

def getModels():
    saved_model_file = "model/keras2_mnist_cnn_allconv.h5"
    return keras.models.load_model(saved_model_file), kc.convert_model_from_saved_files(h5_file=saved_model_file)

def getData():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test[:,:,:,None]
    X_train = X_train[:,:,:,None]
    return X_train, y_train, X_test, y_test

def getDeepliftLayers(deeplift_model):
    layers = []
    for idx, layer in enumerate(deeplift_model.get_layers()):
        if "activations" in str(layer):
            layers.append(idx)

    if (len(layers) != 0):
        del layers[-1]
    return layers

def getScoresFunction(deeplift_model, layers):
    deeplift_default_func = deeplift_model.get_target_contribs_func(
                            find_scores_layer_idx=layers,
                            target_layer_idx=-2)
    return deeplift_default_func

def getScores(X_train, deeplift_default_func, isSavedBefore):
    start = time.time()
    if isSavedBefore:
        infile = open('method_to_task_to_scores','rb')
        method_to_task_to_scores = pickle.load(infile)
        infile.close()
    else:
        method_names = ['deeplift_default']
        method_runtimes = []
        method_to_task_to_scores = OrderedDict()

        for method_name, score_func in [
            ('deeplift_default', deeplift_default_func)]:
            print("Computing scores for:",method_name)

            method_to_task_to_scores[method_name] = []
            
            for task_idx in range(10): #10 output tasks
                print("\tComputing scores for task: "+str(task_idx))
                
                #scores = contribs of layer to task_id output
                scores = [np.array(x) for x in score_func(
                            task_idx=task_idx,
                            input_data_list=[X_train],
                            input_references_list=[np.zeros_like(X_train)],
                            batch_size=200,
                            progress_update=None)]
                
                #method_to_task_to_scores: [method][task_id][layer][i]
                method_to_task_to_scores[method_name].append(scores)

        outfile = open('method_to_task_to_scores','wb')
        pickle.dump(method_to_task_to_scores,outfile)
        outfile.close()
    end = time.time()
    print('task_to_scores time: ' + str(end-start))
    return method_to_task_to_scores

def getImportanceScores(method_to_task_to_scores, training_model_predictions, layers, deeplift_default_func, isSavedBefore): 
    start = time.time()
    if isSavedBefore:
        infile = open('imp_scores','rb')
        imp_scores = pickle.load(infile)
        infile.close()
    else:
        imp_scores = OrderedDict()
        max_class_indices = np.argmax(training_model_predictions, axis=1) 

        for method_name, score_func in [
            ('deeplift_default', deeplift_default_func)]:
            print("Computing importance scores for:",method_name)

            imp_scores[method_name] = []

            #next step for computing importance scores
            for layer_index in range(len(layers)):
                scores = []
                for i in range(len(X_train)):
                    max_class_idx = max_class_indices[i]
                    raw_score = method_to_task_to_scores[method_name][max_class_idx][layer_index][i]
                    avg_score = np.mean([method_to_task_to_scores[method_name][c][layer_index][i] 
                                for c in range(10)],axis=0)
                    scores.append(raw_score - avg_score) #if a neuron contributes equally to each of the 10 output tasks,
                                                        #then its importance score to a particular output is equivalent to nothing
                                                        #so subtract the mean of scores across all output tasks
                scores = np.array(scores)
                imp_scores[method_name].append(scores)
        outfile = open('imp_scores','wb')
        pickle.dump(imp_scores,outfile)
        outfile.close()
    end = time.time()
    print('importance scores time: ' + str(end-start))
    return imp_scores

def getImportantNeurons(layers, X_train, imp_scores):
    important_neurons = {} 
    for idx, layer in enumerate(layers):
        total_scores = 0
        for index in range(len(X_train)):
            total_scores += imp_scores['deeplift_default'][idx][index].flatten()

        mean_scores = total_scores/len(X_train)
        sorted_mean_scores = np.argsort(-mean_scores)
        important_neurons[layer] = list(sorted_mean_scores[0:6])

    return important_neurons

if __name__ == "__main__":
    keras_model, deeplift_model = getModels()
    X_train, y_train, X_test, y_test = getData()

    layers = getDeepliftLayers(deeplift_model)

    deeplift_default_func = getScoresFunction(deeplift_model, layers)

    method_to_task_to_scores = getScores(X_train, deeplift_default_func, isSavedBefore=True)

    training_model_predictions = keras_model.predict(X_train, batch_size=200)

    imp_scores = getImportanceScores(method_to_task_to_scores, training_model_predictions, layers, deeplift_default_func, isSavedBefore=True)

    important_neurons = getImportantNeurons(layers, X_train, imp_scores)

    train_layer_outs = get_layer_outs_new(keras_model, X_train)
    test_layer_outs = get_layer_outs_new(keras_model, X_test)

    relevant_neurons = important_neurons[8]

    quantized = quantizeSilhouette(train_layer_outs[7], relevant_neurons)

    idc = measure_idc(X_test, 7, relevant_neurons, test_layer_outs, quantized)
    print(idc)


print('ok')