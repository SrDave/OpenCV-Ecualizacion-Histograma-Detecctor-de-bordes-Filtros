# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:55:26 2024

@author: colmi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (9.0, 9.0)

def histogram_equalization(image, no_levels=256):

    hist, _ = np.histogram(image.flatten(), bins=no_levels, range=[0, no_levels])

    cdf = np.zeros_like(hist)
    cdf[0] = hist[0]

    for i in range(1, no_levels):
        cdf[i] = cdf[i-1] + hist[i]

    cdf_normalizado = cdf * (no_levels - 1) / cdf[-1]


    imagen_ecualizada = np.interp(image.flatten(), range(no_levels), cdf_normalizado).reshape(image.shape)

    hist_ecualizado, _ = np.histogram(imagen_ecualizada.flatten(), no_levels, [0, no_levels])

    return imagen_ecualizada, hist, hist_ecualizado


imagen = cv2.imread("./fotos/foto1.jpg")

imagen_ecualizada, hist, hist_equalized = histogram_equalization(imagen)

imagen_ecualizada = np.uint8(imagen_ecualizada)


plt.figure(figsize=(20, 10))

blur = cv2.bilateralFilter(imagen_ecualizada,79,27,27)

cv2.imwrite("imagen_ecualizada.jpg", blur)
plt.figure(figsize=[23,10])
plt.subplot(121),plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)),plt.title('Median filter')
plt.xticks([]), plt.yticks([])

# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
# plt.title('Imagen Original')

# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(imagen_ecualizada, cv2.COLOR_BGR2RGB))
# plt.title('Imagen Ecualizada')

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.bar(range(256), hist, color='b')
plt.title('Histograma de la Imagen Original')
plt.xlabel('Nivel de Gris')
plt.ylabel('Frecuencia')

plt.subplot(1, 2, 2)
plt.bar(range(256), hist_equalized, color='r')
plt.title('Histograma de la Imagen Ecualizada')
plt.xlabel('Nivel de Gris')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()

