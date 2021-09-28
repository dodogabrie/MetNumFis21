from numba import jit_module, boolean
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

def binaryToDecimal(arr):
    m=1
    y=0
    for i in range(arr.size):
        y=y + 2**i * arr[i]
    return y


def decimalToBinary(x, nbit):
    arr = np.zeros(nbit, dtype = boolean)
    i = 0
    while((x!=0) or (i ==nbit)):
        if x<1:
            x = x*10
        arr[i] = x % 2
        x = x // 2
        i = i + 1
    return arr

def log_map_gen(x_arr, y_arr, seed, len_loop, r = 4.):
    xk = seed
    m = 2147483647
    a = 48271
    c = 0
    xtemp = xk*a + c
    xkp1 = xtemp % m
    xk1 = xkp1/m
    nbit = 32
    to_pack = decimalToBinary(xk, nbit)
    to_pack1 = decimalToBinary(xk1, nbit)
    for i in range(1, len_loop):
        xk = r * xk * ( 1 - xk )
        b = round(xk)
        to_pack[1:] = to_pack[:-1]
        to_pack[0] = b
        x_arr[i] = binaryToDecimal(to_pack)/2**nbit

        xk1 = r * xk1 * ( 1 - xk1 )
        b = round(xk1)
        to_pack1[1:] = to_pack1[:-1]
        to_pack1[0] = b
        y_arr[i] = binaryToDecimal(to_pack1)/2**nbit
    return x_arr, y_arr

jit_module(fastmath = True, nopython=True, cache = True)

