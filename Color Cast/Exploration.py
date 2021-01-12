import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import Tuple

import cv2

from skimage import img_as_ubyte, img_as_float
from skimage.io import imread, imshow, imsave
from skimage.exposure import histogram, cumulative_distribution

from scipy.stats import norm

class Exploration:
    
    def describe(self, filename: object, read: bool=True) -> pd.DataFrame:
        """
        Gets statistics for a given image.
        
        Parameters
        ----------
        filename : object
            Image filename or Image data.
        read : bool
            Decide when to read the image. Default is True.
            
        Returns
        -------
        df : DataFrame
            DataFrame containing the image statistics.
        """
        # Si se pasa la ruta lee imagen, sino toma "filename" como imagen
        if read:
            img = plt.imread(filename)
        else:
            img = filename
        
        # Lista vacia para almacenar valores
        df_color = []
        
        # Para cada color, calcula una serie de datos estadisticos que permitan como se conforma la imagen.
        for channel in range(0, 3):
            min_color = np.min(img[:,:,channel])
            max_color =np.max(img[:,:,channel]) # Maximo rango dinamico por canal de color
            mean_color = round(np.mean(img[:,:,channel])) # Media de color por cada rango
            median_color = round(np.median(img[:,:,channel])) # Mediana de color por cada rango (representa el valor de la variable de posición central en un conjunto de datos ordenados.)
            std_color = round(img[:,:,channel].std())
            sum_color = round(img[:,:,channel].sum())
            # Calcula percentiles. Por ejemplo: el percentil 90 es el valor en el cual 90 porciento de las intensidades son inferiores.
            perc_01 = np.percentile(img[:,:,channel], 1, axis=(0,1))
            perc_05 = np.percentile(img[:,:,channel], 5, axis=(0,1))
            perc_10 = np.percentile(img[:,:,channel], 10, axis=(0,1))
            perc_90 = np.percentile(img[:,:,channel], 90, axis=(0,1)) 
            perc_95 = np.percentile(img[:,:,channel], 95, axis=(0,1))
            perc_99 = np.percentile(img[:,:,channel], 99, axis=(0,1))
            
            # Añade los datos a un dataframe.
            row = (channel, min_color,max_color, mean_color, median_color, std_color, sum_color, 
                   perc_01, perc_05, perc_10, perc_90, perc_95, perc_99)     
            df_color.append(row)

            color_map = {0:"Red",1:"Green",2:"Blue"}
        
        if read:
            df = pd.DataFrame(df_color, 
                                index = [filename.split("\\")[1], filename.split("\\")[1], filename.split("\\")[1]],
                                columns = ["Channel", 'Min','Max', 'Mean', 'Median',"Std","Sum",
                                        "P_01","P_05","P_10",'P_90',' P_95', 'P_99'])
        else:
            df = pd.DataFrame(df_color, 
                              columns = ["Channel", 'Min','Max', 'Mean', 'Median',"Std","Sum",
                                        "P_01","P_05","P_10",'P_90',' P_95', 'P_99'])
        df["Channel"] = df["Channel"].map(color_map)
            
        return df
    
    def cumulative_distribution_comparation(self, filename: object, read: bool=True) -> None:
        """
        Plots the cumulative ditribution function for each color channel of a given image.
        
        Parameters
        ----------
        filename : object
            Image filename or Image data.
        read : bool
            Decide when to read the image. Default is True.
        """
        # Si se pasa la ruta lee imagen, sino toma "filename" como imagen
        if read:
            img = plt.imread(filename) # Leo la imagen
        else:
            img = filename
        
        # Retorna la funcion de distribucion acumulada por canal. # En este caso un lista de y-> probabilidad acumulada,
        # x-> valor de color del pixel, por tanto devuelve un array de 3x2, 3 canales, eje y y eje x
        freq_bins = [cumulative_distribution(img[:,:,i]) for i in range(3)] 
        # Bins para el eje x para todos los valores de intensidad.                                                                    
        target_bins = np.arange(255)                    
        # Retorna un array de 0 a 1, 255 elementos
        target_freq = np.linspace(0, 1, len(target_bins)) 
        # Titulos para lo distintos plots.
        names = ['Red', 'Green', 'Blue']
        # Colores de las lineas en el grafico.
        line_color = ['red','green','blue']
        f_size = 20
        # Crea figura
        fig, ax = plt.subplots(1, 3, figsize=(7,3))
        
        # Añade histogramas acumulados a la figura
        for n, ax in enumerate(ax.flatten()): # array 1d
            ax.set_title(f'{names[n]}', fontsize = f_size)
            ax.step(freq_bins[n][1], freq_bins[n][0], c=line_color[n],label='Actual FDA') # Eje x -> [n][1] valores de 0-255; eje y -> [n][0]
            ax.plot(target_bins, target_freq, c='gray', label='Target CDF', linestyle = '--')
            
    def acumulative_hist(self, filename: object, read: bool=True) -> None:
        """
        Computes the cumulative distribution function for a given image.
        
        Parameters
        ----------
        filename : object
            Image filename or Image data.
        read : bool
            Decide when to read the image. Default is True.
        """
        # Si se pasa la ruta lee imagen, sino toma "ruta_imagen" como imagen
        if read:
            img = plt.imread(filename) # Leo la imagen
        else:
            img = filename
        
        # Crea histogramas de los distintos colores
        new_hist_R, _ = np.histogram(img[:,:,0].ravel(),256,[0,256]) 
        new_hist_G, _ = np.histogram(img[:,:,1].ravel(),256,[0,256]) 
        new_hist_B, _ = np.histogram(img[:,:,2].ravel(),256,[0,256]) 

        # Calcula el valor acumulado
        new_cdf_R = new_hist_R.cumsum()
        new_cdf_G = new_hist_G.cumsum()
        new_cdf_B = new_hist_B.cumsum()
        
        # Crea figura
        f, axarr = plt.subplots(nrows=1,ncols=3, figsize=(8,3))
        if read:
            plt.suptitle("Acumulative hist - {}".format(filename.split("\\")[1]))
        else:
            plt.suptitle("Acumulative hist")
        # Representa en cada subplot
        plt.sca(axarr[0]); 
        plt.plot(new_cdf_R, color='r'); plt.title('Red')
        plt.sca(axarr[1]); 
        plt.plot(new_cdf_G, color='g'); plt.title('Green')
        plt.sca(axarr[2]); 
        plt.plot(new_cdf_B, color='b'); plt.title('Blue')
        plt.show()
              
    def gray_hist(self,  filename: object, read: bool=True) -> None:
        """
        Computes the gray channel histogram.
        
        Parameters
        ----------
        filename : object
            Image filename or Image data.
        read : bool
            Decide when to read the image. Default is True.
        """
        # Si se pasa la ruta lee imagen, sino toma "filename" como imagen
        if read:
            img = plt.imread(filename) # Leo la imagen
        else:
            img = filename
            
        # Crea un histograma a partir de la imagen
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        # Crea el gráfico
        if read:
            plt.title("Gray Image Histogram - {}".format(filename.split("\\")[1]))
        else:
            plt.title("Gray Image Histogram")
        plt.xlabel("Bins")
        plt.ylabel("Number of Pixels")
        plt.plot(hist)
        # Muestra el grafico.
        plt.show()
        
        
    def color_hist(self,  filename: object, read: bool=True) -> None:
        """
        Computes the histograms for each channel of an image.
        
        Parameters
        ----------
        filename : object
            Image filename or Image data.
        read : bool
            Decide when to read the image. Default is True.
        """
        # Si se pasa la ruta lee imagen, sino toma "filename" como imagen
        if read:
            img = plt.imread(filename) # Leo la imagen
        else:
            img = filename
        # Crea una figura con 3 plots, uno para  cada canal.
        f, axarr = plt.subplots(nrows=1,ncols=3, figsize=(8,3))
        if read:
            plt.suptitle(filename.split("\\")[1])
        plt.sca(axarr[0]); 
        plt.hist(img[:,:,0].ravel(),256,[0,256], color='r'); plt.title('Red')
        plt.sca(axarr[1]); 
        plt.hist(img[:,:,1].ravel(),256,[0,256], color='g') ; plt.title('Green')
        plt.sca(axarr[2]); 
        plt.hist(img[:,:,2].ravel(),256,[0,256], color='b') ; plt.title('Blue')
        plt.show()