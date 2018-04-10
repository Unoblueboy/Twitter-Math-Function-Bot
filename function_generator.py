from math import (sin, cos, tan, 
                 sinh, cosh, tanh)
from random import randint, sample, randrange, seed

class FunctionGenerator(object):
    def __init__(self, binary_op = {}, unary_op = {}):
        if len(binary_op) == 0:
            binary_op = {
                "add" : lambda x,y : x+y,
                "sub" : lambda x,y : x-y,
                "mul" : lambda x,y : x*y,
                "div" : lambda x,y : x/y,
            }
        self.binary_ops = binary_op
        if len(unary_op) == 0:
            unary_op = {
                "sin" : lambda x : sin(x),
                "cos" : lambda x : cos(x),
                "tan" : lambda x : tan(x),
                "sinh" : lambda x : sinh(x),
                "cosh" : lambda x : cosh(x),
                "tanh" : lambda x : tanh(x),
                "repr" : lambda x : 1/x,
            }
        self.unary_ops = unary_op
        self.num_bin_ops = len(self.binary_ops)
        self.num_una_ops = len(self.unary_ops)
        self.total_ops = self.num_bin_ops+self.num_una_ops
        self.current_function = None
    
    def generate_function(self,  iteration_depth = 10, rand_seed = None):
        if rand_seed:
            seed(rand_seed)
        functions = (lambda z : z,)
        for _ in range(iteration_depth):
            func = functions[-1]
            num = randint(1,self.total_ops)
            if num <= self.num_bin_ops:
                bin_op_key = sample(self.binary_ops.keys(),1)[0]
                bin_op = self.binary_ops[bin_op_key]
                val = randrange(-10,10)
                if randint(1,2) == 1:
                    g = lambda z, op=bin_op, f=func : op(f(z), val)
                else:
                    g = lambda z, op=bin_op, f=func : op(val, f(z))
            else:
                una_op_key = sample(self.unary_ops.keys(),1)[0]
                una_op = self.unary_ops[una_op_key]
                g = lambda z, op=una_op, f=func : op(f(z))
            functions = functions.__add__((g,))
        self.current_function = functions[-1]
        return self.current_function


if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt

    unary_ops = {
                "sin" : lambda x : np.sin(x),
                "cos" : lambda x : np.cos(x),
                "tan" : lambda x : np.tan(x),
                "sinh" : lambda x : np.sinh(x),
                "cosh" : lambda x : np.cosh(x),
                "tanh" : lambda x : np.tanh(x),
                "repr" : lambda x : np.divide(np.ones(x.shape),x),
            }
    f_gen = FunctionGenerator(unary_op=unary_ops)
    f1 = f_gen.generate_function()
    xs = np.linspace(-3,3, 500)
    plt.plot(xs,f1(xs))
    plt.show()


