import matplotlib.pyplot as plt
import numpy as np

from regression_function import *

configur = {'exponent': 0.6,
            'size_train': 3000,
            'size_test': 600,
            'batch_size': 200,
            'epochs': 1000,
            # tanto introducendo l'early stop, per trovare il minimo della val_loss, si interrompe prima
            'hidden_layers': [10, 20, 20, 10],
            'ripetitions': 30,
            'layer_type': 'Spectral'
            }
"""conigur è quello che poi è config
"""
dataset = load_regression(configur)
(x_train, y_train), (x_test, y_test) = dataset

""" etichettatura della lista generata dalla load regression, indicando la prima tupla come l'insieme su cui
    addestrare la rete e la seconda come quello su cui verificare la generalità dell'addestramento.
"""
# %%

# Plot Dataset
plt.plot(x_train, y_train, 'bo', markersize=2)
"""plt.plot(np.sort(x_test.reshape(-1)), np.sort(x_test.reshape(-1) ** configur['exponent']), '-r', linewidth=2,
         label='Real_f')"""
# plt.title('Train Set (per {:} Learning)'.format(configur['layer_type']))
plt.show()

"""Tra gli argomenti della funzione pyplot possiamo aggiungere, oltre che ascisse ed ordinate, anche lo spessore del
    marcatore e il tipo di barcatore ('o' indica un semplice cerchio e 'b' indica il colore blu - per il nero si usa
    invece 'k'). Tra gli altri possibili marer '.' è il punto, ',' il pixel, '*' una stella, '|' linee verticali e '_'
    linee orizzontali. Per le linee invece abbiamo '-'('solid'), '--'('dashed'), '-.'('dashdot'), ':'('dotted'),
    potendo impostare anche la 'linewidth' ('lw'). 
"""

# Addestramento
loss = np.zeros(configur['ripetitions'])
model_list = []
print(loss)

for count in range(configur['ripetitions']):
    model = model_build(configur)
    modello, ris = train_model(config=configur,
                                           dati=dataset,
                                           early_stop=True,
                                           tensorboard=False)
    model_list.append(modello)
    loss[count] = ris[0]
    print('Funzione di Loss:', loss[count])

""" In questa fase il programma mostra l'andamento del learning con il progressivo avanzare dell'indice delle epoche,
    per ciascuna delle quai viene indicato il tempo impiegato all'addestramento, la funzione di loss;
    arrivato alla fine delle 150 epoche impostate restituisce anche la loss (MSE), vista come accuracy del test, ed il 
    tempo impiegato per ciascun step (indicativamente 1ms/step). Il processo si ripete poi 10 volte per raccogliere dati
"""
# %%
# Plot Risultati
#plt.plot(x_train, y_train, 'bo', markersize=2, label='Train_set')
# plt.plot(x_test, y_test, 'yo', markersize=2, label='Test_set')
# print('Risultati per la Loss:', loss)
print('Risultati per la Loss: {:.5f}, std: {:.5f}'.format(loss.mean(), loss.std()))
"""plt.plot(np.sort(x_test.reshape(-1)), np.sort(x_test.reshape(-1) ** configur['exponent']), linewidth=2, zorder=5,
         label='Real_f')"""
array_pred = [model_list[i](np.sort(x_test.reshape(-1))) for i in range(configur['ripetitions'])]
array_pred = np.array(array_pred).reshape([configur['ripetitions'], -1])

plt.errorbar(np.sort(x_test.reshape(-1)), array_pred.mean(axis=0), yerr=0.5*(array_pred.max(axis=0)-array_pred.min(axis=0)), c='tab:orange', zorder = 5, label='NN_f')
plt.plot(x_train, y_train, 'bo', markersize=2, zorder=0, label='Train_set')
plt.title('{:}'.format(configur['layer_type']))
plt.legend()
plt.show()
"""Si usano i valori test per plottare l'andamento funzionale che vogliamo ottenere, invece il risultato
    dell'addestramento della rete prende i valori del data set di train
    Per accedere al tensoboard (che tiene traccia dell'andamento della val e di quando il sistema pone l'early stop,
    visualizzandolo graficamente, digitare nel terminal tensorboard --logdir logs/
 """
