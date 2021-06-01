import matplotlib.pyplot as plt
import numpy as np

def verImagen(imagen):
    plt.figure()
    plt.imshow(imagen)
    plt.colorbar
    plt.grid(False)
    plt.show()

nombresEtiquetas = ['Camiseta', 'Pantalón', 'Jersey', 'Vestido', 'Abrigo',
                    'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botines']

def verImagenes5x5(grupoImagenes, etiquetas):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(grupoImagenes[i+300], cmap=plt.cm.binary)
        plt.xlabel(nombresEtiquetas[etiquetas[i+300]])
    plt.show()

#=========================================
def verImagenCifraPrediccion(i, predicciones, etiquetasPruebas, imagenesPrueba):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    verImagenPrediccion(i, predicciones[i], etiquetasPruebas, imagenesPrueba)
    plt.subplot(1,2,2)
    verValorPredicciones(i, predicciones[i],  etiquetasPruebas)
    plt.show()

def verImagenPrediccion(i, conjuntoPredicciones, etiqueta, img):
    conjuntoPredicciones, etiqueta, img = conjuntoPredicciones, etiqueta[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    etiquetaPrediccion = np.argmax(conjuntoPredicciones)
    #etiquetaPrediccion = max(conjuntoPredicciones)
    if etiquetaPrediccion == etiqueta:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"{nombresEtiquetas[etiquetaPrediccion]} {100*np.max(conjuntoPredicciones):2.0f}% ({nombresEtiquetas[etiqueta]})",
                color=color)


def verValorPredicciones(i, conjuntoPredicciones, etiqueta):
    conjuntoPredicciones, etiqueta = conjuntoPredicciones, etiqueta[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    visualizarCifras = plt.bar(range(10), conjuntoPredicciones, color="#777777")
    plt.ylim([0, 1])
    etiquetaPrediccion = np.argmax(conjuntoPredicciones)

    visualizarCifras[etiquetaPrediccion].set_color('red')
    visualizarCifras[etiqueta].set_color('blue')

import random
def verificarPrediccion(filas, columnas, predicciones, etiquetasPruebas, imagenesPrueba):
    num_imagenes = filas*columnas
    plt.figure(figsize=(2*2*columnas, 2*filas))
    for i in range(num_imagenes):
        #x = random.randint(0,3000)
        x = i
        plt.subplot(filas, 2*columnas, 2*i+1)
        verImagenPrediccion(x, predicciones, etiquetasPruebas, imagenesPrueba)
        plt.subplot(filas, 2*columnas, 2*i+2)
        verValorPredicciones(x, predicciones[x], etiquetasPruebas)
    plt.tight_layout()
    plt.show()
