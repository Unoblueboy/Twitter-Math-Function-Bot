import numpy as np
from matplotlib import pyplot as plt
from colorsys import hls_to_rgb

def custom_hls_to_rgb(hls_list):
    print(hls_list)
    return hls_to_rgb(**hls_list)

# Things to do
# Generate a Function

def f(x):
    return x
f = np.vectorize(f)


# Choose a domain to plot it on

min_re = -10
max_re = 10
min_im = -10
max_im = 10

# Choose how refined I want the grid to calculate the function to be

im = np.linspace(min_im, max_im, 7)
re = np.linspace(min_re, max_re, 7)
mesh_re, mesh_im = np.meshgrid(re, im)
complex_plane = mesh_re + 1j*mesh_im

# Calculate Function on all points of the grid

function_plane = f(complex_plane)
abs_function_plane = np.absolute(function_plane)

# Turn all of the points into colour values
# According to the domain coloring given on this page https://en.wikipedia.org/wiki/Domain_coloring
# But must convert HSL to HSV for plotting

def domain_colouring(x):
    # needs to return rgb with all values in interval [0,1]
    hue = (np.pi + np.angle(x))/(2*np.pi)
    light = 1 - 2**(-abs(x))
    saturation = 1
    rgb_tuple = hls_to_rgb(hue, light, saturation)
    return [rgb_tuple[0],rgb_tuple[1],rgb_tuple[2]]

domain_colouring = np.vectorize(domain_colouring, otypes=[np.ndarray])

# hsl_hue = (np.full(function_plane.shape,np.pi)+np.angle(function_plane))/np.full(function_plane.shape,2*np.pi)
# hsl_light = 1-np.power(2,-abs_function_plane)
# hsl_saturation = np.ones(hsl_light.shape)

# # print(hsl_light)
# hls = np.stack([hsl_hue, hsl_light, hsl_saturation],axis=-1)

# rgb = np.apply_along_axis(lambda lis: hls_to_rgb(*lis), 2, hls)
rgb = domain_colouring(function_plane)
print(rgb)
# print(hsv_value)
# Put colour Values on an image

plt.imshow(rgb, interpolation='nearest')
plt.show()