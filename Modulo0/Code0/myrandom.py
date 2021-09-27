from numba import jit_module
import numpy as np

def cong_rand_gen(x_arr, y_arr, seed, len_loop, a, c, m):

    xk = seed
    for i in range(1,len_loop):
        xtemp = xk*a + c  #linear transformation
        xkp1 = xtemp % m  #this is the mod(xtemp,m)=xtemp - m*(xtemp/m) operation
        xk = xkp1     #Notice that going through xkp1 is useless, could save one line,just for the sake of clarity.. 
        x = xk/m  #x in [0,1), the actual random number 
        x_arr[i] = x
    
        #we repeat twice to plot a pair of consecutive random numbers in the sequence
        xtemp = xk*a + c  
        xkp1 = xtemp % m 
        xk = xkp1      
        y = xk/m
        y_arr[i] = y
        
    return x_arr, y_arr

def log_map_gen(x_arr, y_arr, seed, len_loop, r = 4.):
    xk = seed
    m = 2147483647
    a = 48271
    c = 0
    xtemp = xk*a + c
    xkp1 = xtemp % m
    xk1 = xkp1/m
    for i in range(1, len_loop):
        xk = r * xk * ( 1 - xk )
        x_arr[i] = xk
        xk1 = r * xk1 * ( 1 - xk1 )
        y_arr[i] = xk1
    return x_arr, y_arr




jit_module(fastmath = True, nopython=True, cache = True)

