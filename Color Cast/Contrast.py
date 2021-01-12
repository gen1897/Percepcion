import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import Tuple

import cv2

from skimage import img_as_ubyte, img_as_float
from skimage.io import imread, imshow, imsave
from skimage.exposure import histogram, cumulative_distribution

from scipy.stats import norm
class Contrast:
    
    def compute_cdfs(self, data_R: np.ndarray, data_G: np.ndarray, data_B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the cumulative distribution functions for each channel
        
        Parameters
        ----------
        data_R : ndarray
            Red channel data
        data_G : ndarray
            Green channel data
        data_B : ndarray
            Blue channel data
            
        Returns
        -------
        cdf_R, cdf_B, cdf_G : ndarray
            Cumulative distribution functions for each channel
        """
        hist_R, _ = np.histogram(data_R.flatten(), 256, [0, 256]) 
        hist_G, _ = np.histogram(data_G.flatten(), 256, [0, 256]) 
        hist_B, _ = np.histogram(data_B.flatten(), 256, [0, 256]) 
        
        cdf_R, cdf_G, cdf_B = hist_R.cumsum(), hist_G.cumsum(), hist_B.cumsum()
        
        return cdf_R, cdf_G, cdf_B

    def search_limits(self, cdf: np.ndarray, low_saturation: float, high_saturation: float) -> Tuple[int, int]:
        """
        Computes the low and high limits using quantiles.
        
        Parameters
        ----------
        cdf : ndarray
            Cumulative distribution function
        low_saturation : float
            Saturation to compute the low limit
        high_saturation : float
            Saturation to compute the high limit
            
        Returns
        -------
        low_limit, high_limit : Tuple[int, int]
            Low and high limits
        """
        size = cdf.max()
        
        minimum = (cdf <= size * low_saturation)
        maximum = (cdf >= size * (1 - high_saturation))
        
        return np.argmin(minimum), np.argmax(maximum)

    def autocontrast(self, pixel: int, a_low: int, a_high: int) -> int:
        """
        Performs the autocontrast technique in a given pixel intensity

        Parameters
        ----------
        pixel : int
            Pixel intensity
        a_low : int
            Lowest intensity possible
        a_high : int
            Highest intensity possible  
        
        Returns
        -------
        int
            Intensity mapped to the new range
        """
        if pixel <= a_low:
            return 0
        elif a_low < pixel < a_high:
            numerator = (pixel - a_low) * 255
            denominator = a_high - a_low
            
            return numerator / denominator
        else:
            return 255

    def create_new_RGB_image(self, shape: Tuple[int, int, int], data_R: np.ndarray, data_G: np.ndarray, data_B: np.ndarray) -> np.ndarray:
        """
        Creates a new image
        
        Parameters
        ----------
        shape : Tuple[int, int]
            Shape of the new image
        data_R : ndarray
            Red channel data
        data_G : ndarray
            Green channel data
        data_B : ndarray
            Blue channel data
            
        Returns
        -------
        new_img : ndarray
            New RGB image
        """
        new_img = np.zeros(shape).astype(np.uint8) # En el collab sale image.shape
        new_img[:, :, 0], new_img[:, :, 1], new_img[:, :, 2] = data_R, data_G, data_B
        
        return new_img

    def color_balance_RGB_image(self, filename: str, red_saturation_low: float = 0.08, red_saturation_high: float = 0.08,
                          green_saturation_low: float = 0.08, green_saturation_high: float = 0.08,
                          blue_saturation_low: float = 0.08, blue_saturation_high: float = 0.08, read: bool = True) -> np.ndarray:
        """
        Performs color balancing on a given image

        Parameters
        ----------
        filename : object
            Image filename or Image data.
        read : bool
            Decide when to read the image. Default is True
        red_saturation_low : float
            Low saturation for the red channel
        red_saturation_high : float
            High saturation for the red channel
        green_saturation_low : float
            Low saturation for the green channel
        green_saturation_high : float
            High saturation for the green channel 
        blue_saturation_low : float
            Low saturation for the blue channel
        blue_saturation_high : float
            High saturation for the blue channel 

        Returns
        -------
        new_image : ndarray
            Color balanced image
            
        """
        
        if read:
            img = plt.imread(filename)
        else:
            img = filename
            
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError('It is not an RGB image')

        # Gets the data from each channel
        data_R, data_G, data_B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Generated the cumulative distribution function for each channel
        cdf_R, cdf_G, cdf_B = self.compute_cdfs(data_R, data_G, data_B)

        # Computes the low and high limits for each channel
        low_limit_R, high_limit_R = self.search_limits(cdf_R, red_saturation_low, red_saturation_high)
        low_limit_G, high_limit_G = self.search_limits(cdf_G, green_saturation_low, green_saturation_high)
        low_limit_B, high_limit_B = self.search_limits(cdf_B, blue_saturation_low, blue_saturation_high)

        # Apply the autocontrast transformation to each channel
        new_data_R = np.vectorize(self.autocontrast)(data_R, low_limit_R, high_limit_R)
        new_data_G = np.vectorize(self.autocontrast)(data_G, low_limit_G, high_limit_G)
        new_data_B = np.vectorize(self.autocontrast)(data_B, low_limit_B, high_limit_B)

        # Creates the new image
        new_image = self.create_new_RGB_image(img.shape, new_data_R, new_data_G, new_data_B)

        return new_image
    
    def normalize(self, pixel: int, min: int, max: int, max_intensity: int) -> int:
        """
        Normalize a value to a maximum value
        
        Parameters
        ----------
        pixel : int
            Intensity value
        min : int
            Minimum intensity value
        max : int
            Maximum intensity value
        max_intensity : int
            New maximum intensity value
            
        Returns
        -------
        int
            Normalized value to the new range
        """
        numerator = (pixel - min) * max_intensity
        denominator = max - min

        return (numerator / denominator)
    
    def rgb_adjuster_lin(self, image: object, plot: bool=True, read: bool=True) -> None:
        """
        Adjust the intensities of each color channel. 
        
        Parameters
        ----------
        image : object
            Image filename or Image data.
        plot : bool
            Decide when to plot the results
        read : bool
            Decide when to read the image. Default is True.
        """
    
        if read:
            img = plt.imread(image)
        else:
            img = image


        # Corregir rango dinamico, no hay rango de 255 distintos :( (min-max)
        ceros = np.zeros(img.shape)
        ceros[:,:,0] = (np.vectorize(self.normalize)(img[:,:,0],img[:,:,0].min(),img[:,:,0].max(),255))
        ceros[:,:,1] = (np.vectorize(self.normalize)(img[:,:,1],img[:,:,1].min(),img[:,:,1].max(),255))
        ceros[:,:,2] = (np.vectorize(self.normalize)(img[:,:,2],img[:,:,2].min(),img[:,:,2].max(),255))

        img = ceros.astype(np.uint8)

        target_bins = np.arange(255)
        target_freq = np.linspace(0, 1, len(target_bins))
        freq_bins = [cumulative_distribution(img[:,:,i]) for i in range(3)]
        names = ['Reds', 'Greens', 'Blues']
        line_color = ['red','green','blue']
        adjusted_figures = []
        f_size = 20

        # Interpolacion y visualizacion de las imagenes
        
        fig, ax = plt.subplots(1,3, figsize=[8,3])
        for n, ax in enumerate(ax.flatten()):
            # Lo que queremos interpolar son las y, por eso esta al reves, nos da un objeto interpolacion que lo que hacemos es poner lo que queremos interpolar, y los ejes donde queremos sacar dichos puntos
            target_bins = np.arange(0,img[:,:,n].max())
            target_freq = np.linspace(0, 1, len(target_bins))

            interpolation = np.interp(freq_bins[n][0], target_freq, target_bins) # (a,x,y) a->The x-coordinates at which to evaluate the interpolated values. 
            adjusted_image = img_as_ubyte(interpolation[img[:,:,n]].astype(int))
            ax.set_title(f'{names[n]}', fontsize = f_size)
            ax.imshow(adjusted_image, cmap = names[n])
            adjusted_figures.append([adjusted_image])    
        fig.tight_layout() 

        if plot == False:
            plt.close()

        if plot:
            # Creacion de los graficos interpolados
            fig, ax = plt.subplots(1,3, figsize=[8,3])
            for n, ax in enumerate(ax.flatten()):
                interpolation = np.interp(freq_bins[n][0], target_freq, target_bins)
                adjusted_image = img_as_ubyte(interpolation[img[:,:,n]].astype(int))
                freq_adj, bins_adj = cumulative_distribution(adjusted_image)
                ax.set_title(f'{names[n]}', fontsize = f_size)
                ax.step(bins_adj, freq_adj, c=line_color[n], label='Actual CDF')
                ax.plot(target_bins, target_freq, c='gray', label='Target CDF',linestyle = '--')
            fig.tight_layout()
        img_aj = np.dstack((adjusted_figures[0][0],adjusted_figures[1][0],adjusted_figures[2][0]))
        fig, ax = plt.subplots(1,2, figsize=[8,3])
        ax[0].imshow(img)
        ax[1].imshow(np.dstack((adjusted_figures[0][0],adjusted_figures[1][0],adjusted_figures[2][0])));
        
        if plot == False:
            plt.close()
        return img_aj
        # plt.imsave('F_light_6_equalized44.jpg',img_aj)       
        