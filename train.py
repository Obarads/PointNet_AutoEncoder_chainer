import chainer
import numpy as np
from use_senser import dab

class A():
    def __init__(self):
        self.n = 113

    def ab(self): return dab

    def ac(self):
        return self.ab()(19)

def main():
    a = A()
    print(a.ac())

if __name__ == '__main__':
    main()
