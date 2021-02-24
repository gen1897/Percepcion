import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour

from skimage.filters import gaussian
from skimage.segmentation import (active_contour, quickshift)

import skimage.filters as skfilt
from sklearn.cluster import KMeans
from skimage.color import rgb2gray, rgb2hsv
from skimage.morphology import disk
from skimage.morphology import erosion
from skimage.morphology import disk
from skimage.filters import gaussian
from skimage.segmentation import quickshift

from typing import List, Tuple

from Graph import Graph

def dice_score(y_test: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(y_test, y_pred)

    return 2. * intersection.sum() / (y_test.sum() + y_pred.sum())

def generate_ground_truth_mask(mask_filename: str) -> np.ndarray:
    mask_image = plt.imread(mask_filename)[: ,: ,0]
    mask = (mask_image > 0.).astype(np.uint8)
    
    return mask

class ActiveContour():
    def __init__(self) -> None:
        pass

    def segment(self, img, filename):
        if filename == 'zakopane.jpg':
            img = rgb2gray(img)
            # Inicialización del contorno activo: se inicializa con una elipse
            s = np.linspace(0, 2*np.pi,400)
            x = 340 + 140*np.cos(s) # x + radio
            y = 330 + 65*np.sin(s) # y + radio
            init = np.array([x,y]).T
            # Contorno activo
            snake = active_contour(gaussian(img,2), 
                                init, # contorno inicializado
                                alpha=0.012, beta = 3, gamma = 0.0001); # Continuity, Curvature and Gradient

            mas = skimage.measure.grid_points_in_poly([img.shape[1],img.shape[0]],snake)
            image = cv2.rotate(np.float32(mas), cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask_pred = cv2.flip(image, 0)
        elif filename == 'lago.jpg':
            img = rgb2gray(img)
            # Inicialización del contorno activo: se inicializa con una elipse
            s2 = np.linspace(0, 2*np.pi,400)
            x2 = 380 + 215*np.cos(s2) 
            y2 = 360 + 60*np.sin(s2) 
            init2 = np.array([x2,y2]).T
            # Contorno activo
            snake2 = active_contour(gaussian(img,2), 
                                init2, 
                                alpha=0.020, beta = 30, gamma = 0.0001); 

            mas2 = skimage.measure.grid_points_in_poly([img.shape[1],img.shape[0]],snake2)
            image2 = cv2.rotate(np.float32(mas2), cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask_pred = cv2.flip(image2, 0)
        elif filename == 'lagoMontaña.jpg':
            img = rgb2gray(img)
            # Inicialización del contorno activo: se inicializa con una elipse rotada
            s3 = np.linspace(0, 2*np.pi,400)
            o = np.pi/23
            
            x3 = 355 + 395*np.cos(s3)*np.cos(o) +395*np.sin(s3)*np.sin(o)  
            y3 = 540 + 65*np.sin(s3)*np.cos(o) - 395*np.cos(s3)*np.sin(o) 
            init3 = np.array([x3,y3]).T

            snake3 = active_contour(gaussian(img,3), 
                                init3, 
                                alpha=0.0015, beta = 5, gamma = 0.0001); 

            mas3 = skimage.measure.grid_points_in_poly([img.shape[1],img.shape[0]],snake3)
            image3 = cv2.rotate(np.float32(mas3), cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask_pred = cv2.flip(image3, 0)
        elif filename == 'OtroLago.JPG':
            img = rgb2gray(img)
            # Inicialización del contorno activo: se inicializa con una elipse 
            s4 = np.linspace(0, 2*np.pi,400)
            x4 = 290 + 190*np.cos(s4) # x + radio
            y4 = 200 + 110*np.sin(s4) # y + radio
            init4 = np.array([x4,y4]).T

            snake4 = active_contour(gaussian(img,7), 
                                init4, 
                                alpha=0.0005, beta = 3, gamma = 0.005); 

            mas4 = skimage.measure.grid_points_in_poly([img.shape[1],img.shape[0]],snake4)
            image4 = cv2.rotate(np.float32(mas4), cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask_pred = cv2.flip(image4, 0)
            
        return mask_pred

class KMeansSegmenter():
    def __init__(self) -> None:
        pass

    def segmentacion(self, img):
        img = gaussian(img,2) #Aplicamos un filtro Gaussiano a la imagen
        hsv_img = rgb2hsv(img) # Pasamos la imagen a hsvm (Hue, Saturation, Value)
        hue_img = hsv_img[:, :, 0] # Nos quedamos con el canal Hue del hsv, para segmentar de la mejor manera posible.

        ar_hue_img = np.asarray(hue_img,dtype=np.float) # Nos aseguramos que tenemos un array
        col_ar_hue_img = ar_hue_img.reshape((-1, 1)) # Pasamos los datos del array de la imagen a un formato columna

        k_means = KMeans(n_clusters=2,random_state=42)  #Función K means
        k_means.fit(col_ar_hue_img) #Entrenamos el algoritmo Kmeans para los datos que tenemos

        centroides = k_means.cluster_centers_ #Extraemos los centroides
        etiquetas = k_means.labels_  #Extraemos las etiquetas

        eleccion = np.choose(etiquetas, centroides) #Realiza la elección de cada centroide para cada etiqueta
        eleccion.shape = ar_hue_img.shape # Iguala dimensiones imagen original
        comparador = eleccion.copy() #Crea copia de eleccion
        
        #Binarizamos la imagen   
        comparador[comparador >= 0.9] = 0 
        comparador[comparador < 0.15] = 0
        comparador.shape
        thresh =  skfilt.threshold_otsu(comparador)

        binary = comparador > thresh
        binary_neg = ~binary
        
        invert_mask = np.invert(binary_neg)
        
        #Aplicamos una erosion para quitar ruido    
        elemento_estructural=disk(2) 
        img_erosion=erosion(invert_mask,elemento_estructural) 
        
        return img_erosion

class QuickShift():
    def __init__(self) -> None:
        self.parameters = {
            'lago.jpg': {
                'params': {
                    'kernel_size': 31,
                    'max_dist': 110,
                    'ratio': 1,
                    'sigma': 0.3
                },
                'segment': 1
            },
            'zakopane.jpg': {
                'params': {
                    'kernel_size': 15,
                    'max_dist': 50,
                    'ratio': 0.8,
                    'sigma': 0.4
                },
                'segment': 9
            },
            'lagoMontaña.jpg': {
                'params': {
                    'kernel_size': 25,
                    'max_dist': 80,
                    'ratio': 0.8,
                    'sigma': 0.6
                },
                'segment': 12
            },
            'OtroLago.JPG': {
                'params': {
                    'kernel_size': 25,
                    'max_dist': 60,
                    'ratio': 0.8,
                    'sigma': 0.6
                },
                'segment': 6
            }
        }

    def segment_image(self, image, filename: str) -> Tuple[float, np.ndarray, np.ndarray]:
        segments = quickshift(image, **self.parameters[filename]['params'])

        y_pred = (segments == self.parameters[filename]['segment']).astype(np.uint8)
        
        return y_pred

class SuperSegmenter():
    def __init__(self, masks_pred: List[np.ndarray], shape):
        self._masks_pred = masks_pred
        self._shape = shape
    
    def generate_mask(self):
        mask = np.zeros(self._shape)
        
        for row in range(self._shape[0]):
            for column in range(self._shape[1]):
                votes = [mask[row, column] for mask in self._masks_pred]
                majority = self._get_majority(votes)
                
                mask[row, column] = majority
                
        return mask
    
    def _get_majority(self, votes: List[int]):
        count = np.bincount(votes)
        
        return np.argmax(count)
        
        
class Felzenswab(Graph):
    
    def __init__(self,file_path, points):
        super().__init__(file_path)
        self.point = points
        
    def mask(self):
        self.run()
        color = self.file_output[self.point[0],self.point[1]]
        lower = np.array(color-10, dtype= "uint8")
        upper = np.array(color+10, dtype= "uint8")
        
        mask = cv2.inRange(self.file_output, lower, upper)
        output = cv2.bitwise_and(self.file_output, self.file_output, mask=mask)
        
        kernel = np.ones((2,2), np.uint8)
        
        temp = cv2.erode(mask, kernel)
        temp = cv2.dilate(temp, kernel)
        
        return temp
        
