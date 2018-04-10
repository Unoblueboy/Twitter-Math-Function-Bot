import numpy as np
from matplotlib import pyplot as plt
# from colorsys import hls_to_rgb
import colorsys as cs

from skimage import feature
from skimage.filters import gaussian, sobel

class Domain(object):
    ''' A class to represent the domain a function should be plotted in
    Created so the Function Plot function initialisation variables
    Don't get too messy
    
    Typical use:
    
        d = Domain(-1,2,-3,4)       # A domain with corners -1+4i and 2-3i
    '''

    def __init__(self, min_re, max_re, min_im, max_im):
        '''Initialises the class with the maximum and minimum, real and
        imaginary values to be used'''
        self.min_re = min_re
        self.max_re = max_re
        self.min_im = min_im
        self.max_im = max_im

class FunctionPlot():
    '''A class dedicated to the domain colouring of a complex function
    
    The function once initialised will go through the process of creating a domain plot
    as well as grid plot and blend the two to end up with the end result. The domain plot 
    created depends on the domain colouring algorithm used, which is defined by the user,
    and a grid plot is made if the user so decides it.
    
    Typical use:
    
        d = Domain(-3,3,-3,3)
        FunctionPlot(lambda z: z**2, d_c, d, 512, 512)  # Here d_c is some domain colouring algorithm
        
    '''
    def __init__(self, func, domain_colouring, domain, width, height, grid = False):
        '''A function to intialise FunctionPlot as a well as show the function

        Parameters:
            func:               function
                A mathematical function
            domain_colouring:   function
                A domain colouring function
            domain:             Domain
                The borders of where the function will be plotted
            width:              Int
                The width of the image in pixels
            height:             Int
                The height of the image in pixels
            grid:               Boolean
                Whether to include the grid plot or not
        '''
        self.f = np.vectorize(func)
        self.colouring = domain_colouring
        self.domain = domain
        self.w = width
        self.h = height
        self.complex_plane = self._gen_complex_plane()
        self.function_plane = self.f(self.complex_plane)
        self.color_plot = self.color(self.function_plane)
        self.grid_plot = self.grid_color(self.function_plane)
        if grid:
            self.res = self.blend(self.color_plot, self.grid_plot, 0.5)
        else:
            self.res = self.color_plot

    def _gen_complex_plane(self):
        re = np.linspace(self.domain.min_re, self.domain.max_re, self.w)
        im = np.linspace(self.domain.min_im, self.domain.max_im, self.h)
        mesh_re, mesh_im = np.meshgrid(re, im)
        complex_plane = mesh_re + 1j*mesh_im
        complex_plane = np.flip(complex_plane, 0)
        return complex_plane
        
    def color(self, function_plane):
        rgb = np.empty(list(function_plane.shape)+[4])

        for i in range(0, self.h):
            for j in range(0, self.w):
                rgb[i,j] = self.colouring(function_plane[i,j])
        
        return rgb
    
    def grid_color(self, function_plane):
        # Considering points that are MAPPED ONTO A GRID
        # Not points that are mapped from a grid
        bw_checkered = np.empty(list(function_plane.shape))

        for i in range(0, self.h):
            for j in range(0, self.w):
                real_int = np.ceil(function_plane[i,j].imag)
                imag_int = np.ceil(function_plane[i,j].real)
                if (real_int - imag_int) % 2 == 1: # If one is even and the other odd
                    bw_checkered[i,j] = 0
                else:
                    bw_checkered[i,j] = 1
        
        bw_checkered = sobel(bw_checkered)

        bw_grid = np.empty(list(bw_checkered.shape)+[4])
        for i in range(0, self.h):
            for j in range(0, self.w):
                if bw_checkered[i,j]==0:
                    bw_grid[i,j] = [1,1,1,0]
                else:
                    bw_grid[i,j] = [0,0,0,1]
        return bw_grid

    def blend(self, color_plot, grid_plot, weight):
        result = np.zeros(color_plot.shape)
        for i in range(0, self.h):
            for j in range(0, self.w):
                if grid_plot[i,j,3]==0:
                    result[i,j] = color_plot[i,j]
                else:
                    result[i,j] = color_plot[i,j]*(1-weight)
                    result[i,j,3] = 1
        return result

    def plot(self):
        plt.axis('off')
        plt.imshow(self.res, interpolation='nearest')
        plt.show()
    
    def save(self, filename=None):
        domain_width = self.domain.max_re-self.domain.min_re
        domain_height = self.domain.max_im-self.domain.min_im
        fig = plt.figure(figsize=(domain_width, 
                                    domain_height))
        ax1 = fig.add_subplot(1,1,1)
        ax1.imshow(self.res, interpolation='nearest')
        ax1.axis('off')
        extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if filename:
            fig.savefig(filename, bbox_inches=extent)
        else:
            fig.savefig('temp.png', bbox_inches=extent)
        # plt.axis('off')
        # plt.imshow(self.res, interpolation='nearest')
        # if filename:
        #     plt.savefig(filename)
        # else:
        #     plt.savefig('temp.png')
        


def domain_colouring(x):
    # needs to return rgba with all values in interval [0,1]
    a = 0.5
    hue = (0.5 + (np.pi + np.angle(x))/(2*np.pi)) % 1
    value = a + (np.log(abs(x))/np.log(2) % 1)*(1-a)
    saturation = 1
    rgb_tuple = cs.hsv_to_rgb(hue, saturation, value)
    return [rgb_tuple[0],rgb_tuple[1],rgb_tuple[2],1]

if __name__ == '__main__':
    # Cool looking seed values
    # 2, 52, 94, 124396, 123456789, 123454321
    from function_generator import FunctionGenerator as FG
    from scipy.special import gamma
    unary_ops = {
                "sin" : lambda x : np.sin(x),
                "cos" : lambda x : np.cos(x),
                "tan" : lambda x : np.tan(x),
                "sinh" : lambda x : np.sinh(x),
                "cosh" : lambda x : np.cosh(x),
                "tanh" : lambda x : np.tanh(x),
                "repr" : lambda x : 1/x,
            }
    f_gen = FG(unary_op=unary_ops)
    f1 = f_gen.generate_function(rand_seed=2)
    def f(z):
        ans = f1(z)
        if np.isfinite(ans):
            return ans
        else:
            return 0
    d = Domain(-3,3,-3,3)
    f_p = FunctionPlot(f, domain_colouring, d, 512, 512, grid=False)
    f_p.save()
    f_p.plot()