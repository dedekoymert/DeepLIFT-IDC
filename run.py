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
from keras.models import model_from_json, load_model, save_model
from deeplift.util import get_integrated_gradients_function
from collections import OrderedDict
import time
import pickle
from util import *

def getModels(model_name):
    if "LeNet" in model_name:

        json = "model/" + model_name + ".json"
        h5 = "model/" + model_name + ".h5"

        json_file = open(json, 'r') #Read Keras model parameters (stored in JSON file)
        file_content = json_file.read()
        json_file.close()

        model = model_from_json(file_content)
        model.load_weights(h5)
        model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])
            
        deeplift_model = kc.convert_model_from_saved_files(h5_file=h5, json_file=json)
        needed_layers = list(deeplift_model.get_name_to_layer().values())
        deeplift_model = deeplift.models.SequentialModel(needed_layers)

        return model, deeplift_model
    else:
        saved_model_file = "model/" + model_name + ".h5"
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
    return [layers[-1]]

def getScoresFunction(deeplift_model, layers):
    deeplift_default_func = deeplift_model.get_target_contribs_func(
                            find_scores_layer_idx=layers,
                            target_layer_idx=-2)
    return deeplift_default_func

def getScores(X_train, deeplift_default_func, model_name, isSavedBefore):
    start = time.time()
    if isSavedBefore:
        infile = open('experiments/' + model_name ,'rb')
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
                            batch_size=1000,
                            progress_update=None)]
                
                #method_to_task_to_scores: [method][task_id][layer][i]
                method_to_task_to_scores[method_name].append(scores)

        outfile = open('experiments/' + model_name,'wb')
        pickle.dump(method_to_task_to_scores,outfile)
        outfile.close()
    end = time.time()
    print('task_to_scores time: ' + str(end-start))
    return method_to_task_to_scores

def getImportanceScores(method_to_task_to_scores, training_model_predictions, layers, deeplift_default_func, model_name, isSavedBefore): 
    start = time.time()
    if isSavedBefore:
        infile = open('experiments/imp_scores_' + model_name,'rb')
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
        outfile = open('experiments/imp_scores_' + model_name,'wb')
        pickle.dump(imp_scores,outfile)
        outfile.close()
    end = time.time()
    print('importance scores time: ' + str(end-start))
    return imp_scores

def getImportantNeurons(layers, m, X_train, imp_scores):
    important_neurons = {} 
    for idx, layer in enumerate(layers):
        total_scores = 0
        for index in range(len(X_train)):
            total_scores += imp_scores['deeplift_default'][idx][index].flatten()

        mean_scores = total_scores/len(X_train)
        sorted_mean_scores = np.argsort(-mean_scores)
        important_neurons[layer] = list(sorted_mean_scores[0:m])

    return important_neurons

def getSubjectLayer(keras_model):
    layers = []
    for idx, layer in enumerate(keras_model.layers[1:]):
        if "Dense" in str(layer):
            layers.append(idx)

    if (len(layers) != 0):
        del layers[-1]
    return layers[-1]

if __name__ == "__main__":
    model_name = 'LeNet4'
    m = 6
    isSavedBefore = True

    keras_model, deeplift_model = getModels(model_name)
    X_train, y_train, X_test, y_test = getData()

    layers = getDeepliftLayers(deeplift_model)

    deeplift_default_func = getScoresFunction(deeplift_model, layers)

    method_to_task_to_scores = getScores(X_train, deeplift_default_func, model_name, isSavedBefore=isSavedBefore)

    training_model_predictions = keras_model.predict(X_train, batch_size=200)

    imp_scores = getImportanceScores(method_to_task_to_scores, training_model_predictions, layers, deeplift_default_func, model_name, isSavedBefore=isSavedBefore)

    important_neurons = getImportantNeurons(layers, m, X_train, imp_scores)

    train_layer_outs = 0
    test_layer_outs = 0
    if "LeNet" in model_name:
        train_layer_outs = get_layer_outs_new_lenet(keras_model, X_train)
        test_layer_outs = get_layer_outs_new_lenet(keras_model, X_test)
    else: 
        train_layer_outs = get_layer_outs_new(keras_model, X_train)
        test_layer_outs = get_layer_outs_new(keras_model, X_test)

    relevant_neurons = important_neurons[layers[0]]
    
    subject_layer = getSubjectLayer(keras_model)

    quantized = quantizeSilhouette(train_layer_outs[subject_layer], relevant_neurons)

    idc = measure_idc(X_test, subject_layer, relevant_neurons, test_layer_outs, quantized)
    print(quantized)
    print(idc)
