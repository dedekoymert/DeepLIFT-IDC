from keras import models
from sklearn.metrics import silhouette_score
from sklearn import cluster
import numpy as np
import time

def limit_precision(values, prec=2):
    limited_values = []
    for v in values:
        limited_values.append(round(v,prec))
    return limited_values

def get_layer_outs_new(model, inputs, skip=[]):
    evaluater = models.Model(inputs=model.input,
                             outputs=[layer.output for index, layer in enumerate(model.layers)])
    return evaluater.predict(inputs)


def quantizeSilhouette(out_vectors, relevant_neurons):
    start = time.time()
    quantized_ = []

    for i in relevant_neurons:

      out_i = []
      for l in out_vectors:
          l = l.flatten()
          out_i.append(np.mean(l[...,i]))

      values = []
      if not len(out_i) < 10: #10 is threshold of number positives in all test input activations
            clusterSize = range(2, 5)#[2, 3, 4]
            clustersDict = {}
            for clusterNum in clusterSize:
                kmeans          = cluster.KMeans(n_clusters=clusterNum)
                clusterLabels   = kmeans.fit_predict(np.array(out_i).reshape(-1, 1))
                silhouetteAvg   = silhouette_score(np.array(out_i).reshape(-1, 1), clusterLabels)
                clustersDict [silhouetteAvg] = kmeans

            maxSilhouetteScore = max(clustersDict.keys())
            bestKMean          = clustersDict[maxSilhouetteScore]

            values = bestKMean.cluster_centers_.squeeze()
      values = list(values)
      values = limit_precision(values)

        # if not conv: values.append(0) #If it is convolutional layer we dont add  directly since thake average of whole filter.
      if len(values) == 0:
          values.append(0)

      quantized_.append(values)
    
    end = time.time()
    print('quantize silhouette time: ' + str(end-start))
    return quantized_

def determine_quantized_cover(lout, quantized):
    covered_comb = []
    for idx, l in enumerate(lout):
        #if l == 0:
        #    covered_comb.append(0)
        #else:
        closest_q = min(quantized[idx], key=lambda x:abs(x-l))
        covered_comb.append(closest_q)

    return covered_comb

def measure_idc(test_inputs, subject_layer,
                                   relevant_neurons,
                                   test_layer_outs, qtized,
                                   covered_combinations=()):
  for test_idx in range(len(test_inputs)):
      lout = []
      for r in relevant_neurons:
        lout.append(np.mean(test_layer_outs[subject_layer][test_idx][...,r]))

      comb_to_add = determine_quantized_cover(lout, qtized)

      if comb_to_add not in covered_combinations:
          covered_combinations += (comb_to_add,)

  max_comb = 1#q_granularity**len(relevant_neurons)
  for q in qtized:
      max_comb *= len(q)

  covered_num = len(covered_combinations)
  coverage = float(covered_num)/max_comb

  return coverage*100, covered_combinations, max_comb