from linear_algebra import Vector, dot, distance, add, scalar_multiply, vector_mean
from typing import Callable, TypeVar, List, Iterator
import matplotlib.pyplot as plt
import random

def sum_of_squares(v):
    return dot(v, v)

def difference_quotient(f, x, h):
    return (f(x+h) - f(x)) / h

def square(x):
    return x * x

def derivative(x):
    return 2 * x

def estimate_gradient(f, v, h= 0.0001):
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]

def gradient_step(v, gradient, step_size):
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]

inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

def linear_gradient(x, y, theta):
    slope, intercept = theta
    predicted = slope * x + intercept # prediction of the model
    error = (predicted - y) # error is (predicted - actual)
    sqared_error = error ** 2 # minimize squared error
    grad = [2 * error * x, 2 * error] # using gradient
    return grad

def partial_difference_quotient(f, v, i, h):
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

T = TypeVar('T') # this allows us to type "generic" functions    

def minibatches(dataset, batch_size, shuffle=True):
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts) # shuffle the batches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]

def main():
    xs = range(-10, 11)
    actuals = [derivative(x) for x in xs]
    estimates = [difference_quotient(square, x, h=0.001) for x in xs]
    
    plt.title("Actual Derivatives vs. Estimates")
    plt.plot(xs, actuals, 'rx', label='Actual')       # red  x
    plt.plot(xs, estimates, 'b+', label='Estimate')   # blue +
    plt.legend(loc=9)
    # plt.show()
    plt.close()


    # "Using the Gradient" example
    
    # pick a random starting point
    v = [random.uniform(-10, 10) for i in range(3)]
    print(v)

    for epoch in range(1000):
        grad = sum_of_squares_gradient(v)    # compute the gradient at v
        v = gradient_step(v, grad, -0.01)    # take a negative gradient step
        print(epoch, v)
    
    assert distance(v, [0, 0, 0]) < 0.001    # v should be close to 0

    # Start with random values for slope and intercept.
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
    learning_rate = 0.001
    
    for epoch in range(5000):
        # Compute the mean of the gradients
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
        # Take a step in that direction
        theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    
    slope, intercept = theta
    assert 19.9 < slope < 20.1,   "slope should be about 20"
    assert 4.9 < intercept < 5.1, "intercept should be about 5"


    # Minibatch gradient descent example
    
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
    for epoch in range(1000):
        for batch in minibatches(inputs, batch_size=20):
            grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    
    slope, intercept = theta
    assert 19.9 < slope < 20.1,   "slope should be about 20"
    assert 4.9 < intercept < 5.1, "intercept should be about 5"


    # Stochastic gradient descent example
    
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
    for epoch in range(100):
        for x, y in inputs:
            grad = linear_gradient(x, y, theta)
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    
    slope, intercept = theta
    assert 19.9 < slope < 20.1,   "slope should be about 20"
    assert 4.9 < intercept < 5.1, "intercept should be about 5"

if __name__ == "__main__": main()