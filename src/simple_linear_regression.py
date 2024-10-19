from linear_algebra import Vector
from typing import Tuple
from statistics_ds import correlation, standard_deviation, mean, de_mean, num_friends_good, daily_minutes_good
import random, tqdm
from gradient_descent import gradient_step

def predict(alpha, beta, x_i):
    return beta * x_i + alpha

def error(alpha, beta, x_i, y_i):
    """
    The error from predicting beta * x_i + alpha
    whenm the actual value is y_i
    """
    return predict(alpha, beta, x_i) - y_i

def sum_of_sqerrors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x,y))

def least_squares_fit(x, y):
    """
    Given two vectors x and y,
    find the least-squares values of alpha and beta
    """
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x]


# Should find that y = 3x - 5
assert least_squares_fit(x, y) == (-5, 3)

alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
assert 22.9 < alpha < 23.0
assert 0.9 < beta < 0.905

def total_sum_of_squares(y):
    """the total squared variation of y_i's from their mean"""
    return sum(v ** 2 for v in de_mean(y))

def r_squared(alpha, beta, x, y):
    """
    the fraction of variation in y captured by the model, which equals
    1 - the fraction of variation in y not captured by the model
    """
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) /
                  total_sum_of_squares(y))
    
rsq = r_squared(alpha, beta, num_friends_good, daily_minutes_good)
assert 0.328 < rsq < 0.330

def main():
    num_epochs = 10000
    random.seed(0)
    
    guess = [random.random(), random.random()]
    
    learning_rate = 0.00001
    
    with tqdm.trange(num_epochs) as t:
        for _ in t:
            alpha, beta = guess
            
            grad_a = sum(2 * error(alpha, beta, x_i, y_i)
                         for x_i, y_i in zip(num_friends_good, daily_minutes_good))
            
            grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
                         for x_i, y_i in zip(num_friends_good, daily_minutes_good))
            
            loss = sum_of_sqerrors(alpha, beta, num_friends_good, daily_minutes_good)
            
            t.set_description(f"loss: {loss:.3f}")
            
            guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)
            
    alpha, beta = guess
    
    assert 22.9 < alpha < 23.0
    assert 0.9 < beta < 0.905
    
if __name__ == "__main__": main()