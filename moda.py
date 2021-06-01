import tensorflow as tf
from tensorflow import keras

from funciones import *
#Reconocimiento de Vestuario

def iniciarModelo():
    ropa_mnist = keras.datasets.fashion_mnist

    print(type(ropa_mnist))

    (imgEntre, etiEntre), (imgEje, etiEje) = ropa_mnist.load_data()
    imgEntre = imgEntre / 255.0

    imgEje = imgEje / 255.0

    verImagen(imgEntre[5])
    verImagen(imgEje[5])
    verImagenes5x5(imgEje, etiEje)

    # configurar capas
    modelo1 = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])
    # compilar el modelo
    modelo1.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    # entremanos
    imagenes = imgEntre
    conocimiento = etiEntre
    ciclos = 3
    modelo1.fit(imagenes, conocimiento, epochs=ciclos)

    # analizar le exactitud del modelo
    imagenesEjemplo = imgEje
    conocimientoEjemplo = etiEje
    valorPerdida, valorExactitud = modelo1.evaluate(imagenesEjemplo, conocimientoEjemplo, verbose=2)

    # crear predicciones
    print(type(modelo1))
    valorPrediccion = modelo1.predict(imgEje)

    # n=5
    # verImagen(imgEje[n])
    # print(f'prediccion: {valorPrediccion[n]}')

    # ejemplo
    verImagenCifraPrediccion(10, valorPrediccion, etiEje, imgEje)
    verImagenCifraPrediccion(5, valorPrediccion, etiEje, imgEje)
    verImagenCifraPrediccion(20, valorPrediccion, etiEje, imgEje)
    # verificarPrediccion(5, 3, valorPrediccion, etiEje, imgEje)


if __name__ == '__main__':
    iniciarModelo()
