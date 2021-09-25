import integrators
import matplotlib.pyplot as plt
def test_int():

    def f(x):
        return x/2
    x0 = 10
    step = 100
    h = 0.0001
    x = integrators.easy_integrate(f, x0, step, h)
    plt.plot(x, marker='*')
    plt.show()



if __name__=='__main__':
    test_int()
