import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from shutil import rmtree
import matplotlib.pyplot as plt
from SpectralLayer import Spectral

def load_regression(conf):
    exponent = conf['exponent']
    size_train = conf['size_train']
    size_test = conf['size_test']
    size = size_train + size_test
    x = np.random.uniform(0, 1, size) #centro, deviazione standard, mentre size è per restituire uno scalare
    y = x ** exponent + np.random.normal(0, .03, size)
    return [(x[:size_train, np.newaxis], y[:size_train, np.newaxis]),
            (x[size_train:, np.newaxis], y[size_train:, np.newaxis])]
# v[y:] dall' elemento y incluso in poi, v[:a] dall'inizio fino ad y escluso

def model_build(config):
    """
    config è un dizionario (struttura) ovvero un vettore i cui coefficienti sono etichetti da stringhe e non da numeri
    naturali
    """
    model = tf.keras.Sequential()
    """ Crea lo "scheletro" della rete: la prima è il layer di input (il numero x nel nostro caso), il secondo 
        """
    # Primo Layer
    #model.add(tf.keras.Input(shape=(1)))

    # Layers Nascosti
    for cont, lay in enumerate(config['hidden_layers']):
        """ crea tanti layer nascosti quanto la dimensione della lista config['hidden_layers'], associando a ciascuno 
        il numero di nodi indicato (entrambi indicati nel main)
        """
        nm = 'Hidden' + str(cont) + '_' + str(lay)

        if config['layer_type'] == 'Dense':
            model.add(tf.keras.layers.Dense(lay,
                                            activation='elu',
                                            use_bias=False,
                                            name=nm))
        else:
            model.add(Spectral(lay,
                               activation='elu',
                               use_bias=False,
                               is_base_trainable=False))

    # Ultimo Layer
    if config['layer_type'] == 'Dense':
        model.add(tf.keras.layers.Dense(1,
                                        activation=None,
                                        use_bias=False))
    else:
        model.add(Spectral(1,
                           activation=None))



    return model


def train_model(config, dati, tensorboard=True, early_stop=False):
    print('\nTraining...\n')
    (x_train, y_train), (x_test, y_test) = dati[0], dati[1] #dataset diviso

    model = model_build(config) #modello creato (con spazio delle ipotesi)
    if config['layer_type'] == 'Dense':
        rt = 0.01 #0.005
    else:
        rt = 0.03 #0.01
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=rt),
                  loss='mse',
                  metrics=['mse'],
                  run_eagerly=False)
    """Il modello della regressione, dopo aver definito lo spazio delle ipotesi (a cui appartiene la funzione cercata),
     è completato con l'indicazione della funzione loss e dell'algoritmo di ottimizzazione per minimizzare la loss
     stessa"""
    callbacks = []
    rmtree('.\\logs', ignore_errors=True)
    if tensorboard:
        callbacks.append(tf.keras.callbacks.TensorBoard('.\\logs',
                                                        update_freq=2,
                                                        write_images=True))
    if early_stop:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            restore_best_weights=True,
            patience=60,
            verbose=2,
            mode="min"))

    model.fit(x_train,
              y_train,
              validation_split=0.2,
              epochs=config['epochs'],
              batch_size=config['batch_size'],
              verbose=2,
              callbacks=callbacks)

    print('\nAccuracy sul testset:\n')
    R = model.evaluate(x_test,
                       y_test,
                       batch_size=1000,
                       verbose=1)

    return model, R
