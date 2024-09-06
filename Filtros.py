import cv2
import numpy as np
import matplotlib.pyplot as plt

''' Enfatizar borde '''

# Cargar imagen
src_image = cv2.imread("./fotos/foto1.jpg")
# Definir el kernel 3x3
kernel_enfatizar = np.array([
  [0, -1, 0],
  [-1, 5, -1],
  [0, -1, 0]
])

resulting_image_enfatizar = cv2.filter2D(src_image, -1, kernel_enfatizar)

# Mostrar las imágenes
cv2.imshow("Original", src_image)
cv2.imshow("Enfatizar borde", resulting_image_enfatizar)
cv2.imwrite("Filter2d Sharpened Image.jpg", resulting_image_enfatizar)


''' Detector de bordes '''

# Cargar imagen
src_image_bordes = cv2.imread("./fotos/foto1.jpg")
# Definir el kernel 3x3
kernel_bordes = np.array([
  [-1, -1, -1],
  [-1, 8, -1],
  [-1, -1, -1]
])

resulting_image_bordes = cv2.filter2D(src_image_bordes, -1, kernel_bordes)

# Mostrar la imagen filtrada
cv2.imshow("Detector de bordes", resulting_image_bordes)
cv2.imwrite("Filter2d Outline Image.jpg", resulting_image_bordes)


''' Añadir relieve '''

# Cargar imagen
src_image = cv2.imread("./fotos/foto1.jpg")
# Definir el kernel 3x3
kernel_relieve = np.array([
  [-2, -1, 0],
  [-1, 1, 1],
  [0, 1, 2]
])

resulting_image_relieve = cv2.filter2D(src_image, -1, kernel_relieve)

# Mostrar la imagen filtrada
cv2.imshow("Añadir relieve", resulting_image_relieve)
cv2.imwrite("Filter2d Emboss Image.jpg", resulting_image_relieve)


# Esperar brevemente en un bucle para continuar el código
while True:
    # Esperar 1 milisegundo entre cada verificación de ventanas
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Puedes presionar 'q' para salir del bucle
        break
    if cv2.getWindowProperty("Original", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.getWindowProperty("Enfatizar borde", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.getWindowProperty("Detector de bordes", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.getWindowProperty("Añadir relieve", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()


''' Filtros para Suavizar Imágenes '''

img_bgr = cv2.imread('./fotos/foto4_1.jpg')
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Promediado

blur = cv2.blur(img,(5,5))
plt.figure(figsize=[23,10])
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Borrosa')
plt.xticks([]), plt.yticks([])
plt.show()

# DEsenfoque Gaussiano

blur = cv2.GaussianBlur(img,(5,5),0)
plt.figure(figsize=[23,10])
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Desenfoque gaussiano')
plt.xticks([]), plt.yticks([])

# Filtro Mediana

# genera imagen ruido
row,col,ch= img.shape
mean = 0
var = .015
sigma = var**0.5
gauss1 = np.random.normal(mean,sigma,(row,col)) #vamos a copiar mismo ruido en los 3 canales de color
gauss2 = gauss1.reshape(row,col)
gauss = np.repeat(gauss2[:, :, np.newaxis], 3, axis=2)


#convierte imagen a valores normalizados -0.5,0.5
img_1=(img-128)/255

noisy1 = img_1 + gauss
noisy2 = noisy1 *255+ 128
noisy3 = noisy2.astype(np.uint8)
noisy = np.clip(noisy3,0,255)

median = cv2.medianBlur(noisy,5)

plt.figure(figsize=[23,10])
plt.subplot(121),plt.imshow(noisy),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('Median filter')
plt.xticks([]), plt.yticks([])

# Filtro Bilateral

img=cv2.imread("./fotos/foto6_1.jpg")
blur = cv2.bilateralFilter(img,79,27,27)
plt.figure(figsize=[23,10])
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Median filter')
plt.xticks([]), plt.yticks([])

# Detector bordes

import cv2 as cv

img = cv.imread('./fotos/foto6_3.jpg',0)
# Output dtype = cv.CV_8U
sobelx8u = cv.Sobel(img,cv.CV_8U,1,0,ksize=5)
# Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
sobelx64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
plt.figure(figsize=[20,20])
plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()

# Detector Sobel vertical

img = cv.imread("./fotos/foto6_2.png",0)
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
plt.figure(figsize=[20,20])
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(abs(sobelx),cmap = 'gray')  #observa el efecto de plotear el valor absoluto; ojo con el tipo de dato, entero, decimal, etc
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()