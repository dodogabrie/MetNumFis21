import readfile as rf 
import numpy as np
import matplotlib.pyplot as plt
import time

def create_data():
    data = np.arange(int(4e7)).reshape(int(1e7),4)
    np.savetxt('test.txt', data)
    return 

def read_data():
    start = time.time()
    data = rf.fastload('test.txt')
    print('fastload time:', time.time() - start)
    print(data)
    print(data.shape)
    start = time.time()
    data = np.loadtxt('test.txt')
    print('numpy load time:', time.time() - start)
    print(data)
    print(data.shape)

if __name__ == "__main__":
    create_data()
    read_data()
