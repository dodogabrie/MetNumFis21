import readfile as rf 
import numpy as np
import matplotlib.pyplot as plt
import time

def create_data():
    data = np.arange(int(2e6)).reshape(int(1e6),2)
    np.savetxt('test.txt', data)
    return 

def read_data():
    print(rf.get_data_shape('test.txt'))
    start = time.time()
    data = rf.fastload('test.txt')
    fs_time = time.time()-start
    print('fastload time:', fs_time)
    print(data)
    print(data.shape)
    start = time.time()
    data = np.loadtxt('test.txt')
    np_time = time.time()-start
    print('numpy load time:', np_time)
    print(data)
    print(data.shape)
    print('speed up:', np_time/fs_time)

if __name__ == "__main__":
    create_data()
    read_data()
