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
import deeplift
from deeplift.layers import NonlinearMxtsMode
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.util import compile_func
import numpy as np
from keras import backend as K
import deeplift
from deeplift.util import get_integrated_gradients_function
from collections import OrderedDict
import time
import pickle

def getModels():
    saved_model_file = "model/keras2_mnist_cnn_allconv.h5"
    return keras.models.load_model(saved_model_file), kc.convert_model_from_saved_files(h5_file=saved_model_file)

def getData():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test[:,:,:,None]
    X_train = X_train[:,:,:,None]
    return X_train, y_train, X_test, y_test

def getLayers(deeplift_model):
    layers = []
    for idx, layer in enumerate(deeplift_model.get_layers()):
        if "activations" in str(layer):
            layers.append(idx)

    if (len(layers) != 0):
        del layers[-1]
    return layers

def getScoresFunction():
    deeplift_default_func = deeplift_model.get_target_contribs_func(
                            find_scores_layer_idx=layers,
                            target_layer_idx=-2)
    return deeplift_default_func

def getScores(isSavedBefore):
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
            start = time.time() #start timer

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

            end = time.time()
            method_runtimes.append(end-start)
            print(method_name + " time: " + str(end-start) +'\n')

        outfile = open('method_to_task_to_scores','wb')
        pickle.dump(method_to_task_to_scores,outfile)
        outfile.close()
    return method_to_task_to_scores

def getImportanceScores(method_to_task_to_scores, training_model_predictions, isSavedBefore): 
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
                imp_scores[method_name].append(scores)
        outfile = open('imp_scores','wb')
        pickle.dump(imp_scores,outfile)
        outfile.close()
    return imp_scores

def getImportantNeurons(layers, X_train, imp_scores):
    for layer in range(len(layers)):
        total_scores = 0
        for index in range(len(X_train)):
            total_scores += imp_scores['deeplift_default'][layer][index].flatten()

        mean_scores = total_scores/len(X_train)
        sorted_mean_scores = np.argsort(-mean_scores)
        print('Most important 6 neuron indexes and scores for layer: ' + str(layers[layer]))
        print(sorted_mean_scores[0], mean_scores[sorted_mean_scores[0]])
        print(sorted_mean_scores[1], mean_scores[sorted_mean_scores[1]])
        print(sorted_mean_scores[2], mean_scores[sorted_mean_scores[2]])
        print(sorted_mean_scores[3], mean_scores[sorted_mean_scores[3]])
        print(sorted_mean_scores[4], mean_scores[sorted_mean_scores[4]])
        print(sorted_mean_scores[5], mean_scores[sorted_mean_scores[5]])

if __name__ == "__main__":
    keras_model, deeplift_model = getModels()
    X_train, y_train, X_test, y_test = getData()

    layers = getLayers(deeplift_model)

    deeplift_default_func = getScoresFunction()

    method_to_task_to_scores = getScores(isSavedBefore=False)

    training_model_predictions = keras_model.predict(X_train, batch_size=200)
    imp_scores = getImportanceScores(method_to_task_to_scores, training_model_predictions, isSavedBefore=False)

    getImportantNeurons(layers, X_train, imp_scores)

print('ok')