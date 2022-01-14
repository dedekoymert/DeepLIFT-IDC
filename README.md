# DeepLIFT-IDC #

### Prequisite ###

* Python = 3.7.12
* tensorflow = 1.15.2
* Keras = 2.3.1
* scikit-learn = 1.0.2

Install deeplift
* pip install deeplift

Install deeplift model
* wget https://raw.githubusercontent.com/AvantiShri/model_storage/d65951145fab2ad5de91b3c5f1bca7c378fabf93/deeplift/mnist/keras2_mnist_cnn_allconv.h5 -O model/keras2_mnist_cnn_allconv.h5


### Run ###

python run.py

change runing model change line 163
change most important neuron number line 164
not to calculate DeepLIFT scores change line 165
