import dynet as dy
import math


def get_loss(u, sign):
    if sign == 1:
        # From the data
        return -dy.log_sigmoid(u)
    else:
        return -dy.log_sigmoid(-u)


def one_hot(idx, size):
    x = [0 for i in range(size)]
    if idx < size:
        x[idx] = 1
    return x

def dy_log_sigmoid(val):
    return dy.log_sigmoid(dy.scalarInput(val)).value()

print("Dynet:")
print(dy_log_sigmoid(3))
print("Maths:")
print(math.log(1/(1 + math.exp(-3))))