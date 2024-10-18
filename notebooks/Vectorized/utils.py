import numpy as np

def uniform(z):
    return np.where(np.abs(z) <= 1., 1., 0.)

def rectangle(z):
    return np.where(np.abs(z) <= 1., 0.5, 0.)

def triangle(z):
    return np.where(np.abs(z) <= 1., 1 - np.abs(z), 0.)

def epanechnikov(z):
    return np.where(np.abs(z) <= 1., 0.75 * (1 - z**2), 0.)

def biweight(z):
    return np.where(np.abs(z) <= 1., 0.9375 * (1 - z**2)**2, 0.)

def tricube(z):
    return np.where(np.abs(z) <= 1., (1 - np.abs(z)**3)**3, 0.)

def gaussian(z):
    return 1./np.sqrt(2 * np.pi) * np.exp(-z**2 / 2)

def silverman(z):
    return 0.5 * np.exp(-np.abs(z) / np.sqrt(2)) * np.sin(np.abs(z) / np.sqrt(2) + np.pi/4)
