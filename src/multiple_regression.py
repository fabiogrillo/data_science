from typing import List, TypeVar, Callable
from linear_algebra import dot, Vector, vector_mean, add
from gradient_descent import gradient_step
import random, tqdm, datetime
from simple_linear_regression import total_sum_of_squares
from statistics_ds import median, standard_deviation, daily_minutes_good
from probability import normal_cdf

inputs = [
    [1.0, 49, 4, 0],
    [1, 41, 9, 0],
    [1, 40, 8, 0],
    [1, 25, 6, 0],
    [1, 21, 1, 0],
    [1, 21, 0, 0],
    [1, 19, 3, 0],
    [1, 19, 0, 0],
    [1, 18, 9, 0],
    [1, 18, 8, 0],
    [1, 16, 4, 0],
    [1, 15, 3, 0],
    [1, 15, 0, 0],
    [1, 15, 2, 0],
    [1, 15, 7, 0],
    [1, 14, 0, 0],
    [1, 14, 1, 0],
    [1, 13, 1, 0],
    [1, 13, 7, 0],
    [1, 13, 4, 0],
    [1, 13, 2, 0],
    [1, 12, 5, 0],
    [1, 12, 0, 0],
    [1, 11, 9, 0],
    [1, 10, 9, 0],
    [1, 10, 1, 0],
    [1, 10, 1, 0],
    [1, 10, 7, 0],
    [1, 10, 9, 0],
    [1, 10, 1, 0],
    [1, 10, 6, 0],
    [1, 10, 6, 0],
    [1, 10, 8, 0],
    [1, 10, 10, 0],
    [1, 10, 6, 0],
    [1, 10, 0, 0],
    [1, 10, 5, 0],
    [1, 10, 3, 0],
    [1, 10, 4, 0],
    [1, 9, 9, 0],
    [1, 9, 9, 0],
    [1, 9, 0, 0],
    [1, 9, 0, 0],
    [1, 9, 6, 0],
    [1, 9, 10, 0],
    [1, 9, 8, 0],
    [1, 9, 5, 0],
    [1, 9, 2, 0],
    [1, 9, 9, 0],
    [1, 9, 10, 0],
    [1, 9, 7, 0],
    [1, 9, 2, 0],
    [1, 9, 0, 0],
    [1, 9, 4, 0],
    [1, 9, 6, 0],
    [1, 9, 4, 0],
    [1, 9, 7, 0],
    [1, 8, 3, 0],
    [1, 8, 2, 0],
    [1, 8, 4, 0],
    [1, 8, 9, 0],
    [1, 8, 2, 0],
    [1, 8, 3, 0],
    [1, 8, 5, 0],
    [1, 8, 8, 0],
    [1, 8, 0, 0],
    [1, 8, 9, 0],
    [1, 8, 10, 0],
    [1, 8, 5, 0],
    [1, 8, 5, 0],
    [1, 7, 5, 0],
    [1, 7, 5, 0],
    [1, 7, 0, 0],
    [1, 7, 2, 0],
    [1, 7, 8, 0],
    [1, 7, 10, 0],
    [1, 7, 5, 0],
    [1, 7, 3, 0],
    [1, 7, 3, 0],
    [1, 7, 6, 0],
    [1, 7, 7, 0],
    [1, 7, 7, 0],
    [1, 7, 9, 0],
    [1, 7, 3, 0],
    [1, 7, 8, 0],
    [1, 6, 4, 0],
    [1, 6, 6, 0],
    [1, 6, 4, 0],
    [1, 6, 9, 0],
    [1, 6, 0, 0],
    [1, 6, 1, 0],
    [1, 6, 4, 0],
    [1, 6, 1, 0],
    [1, 6, 0, 0],
    [1, 6, 7, 0],
    [1, 6, 0, 0],
    [1, 6, 8, 0],
    [1, 6, 4, 0],
    [1, 6, 2, 1],
    [1, 6, 1, 1],
    [1, 6, 3, 1],
    [1, 6, 6, 1],
    [1, 6, 4, 1],
    [1, 6, 4, 1],
    [1, 6, 1, 1],
    [1, 6, 3, 1],
    [1, 6, 4, 1],
    [1, 5, 1, 1],
    [1, 5, 9, 1],
    [1, 5, 4, 1],
    [1, 5, 6, 1],
    [1, 5, 4, 1],
    [1, 5, 4, 1],
    [1, 5, 10, 1],
    [1, 5, 5, 1],
    [1, 5, 2, 1],
    [1, 5, 4, 1],
    [1, 5, 4, 1],
    [1, 5, 9, 1],
    [1, 5, 3, 1],
    [1, 5, 10, 1],
    [1, 5, 2, 1],
    [1, 5, 2, 1],
    [1, 5, 9, 1],
    [1, 4, 8, 1],
    [1, 4, 6, 1],
    [1, 4, 0, 1],
    [1, 4, 10, 1],
    [1, 4, 5, 1],
    [1, 4, 10, 1],
    [1, 4, 9, 1],
    [1, 4, 1, 1],
    [1, 4, 4, 1],
    [1, 4, 4, 1],
    [1, 4, 0, 1],
    [1, 4, 3, 1],
    [1, 4, 1, 1],
    [1, 4, 3, 1],
    [1, 4, 2, 1],
    [1, 4, 4, 1],
    [1, 4, 4, 1],
    [1, 4, 8, 1],
    [1, 4, 2, 1],
    [1, 4, 4, 1],
    [1, 3, 2, 1],
    [1, 3, 6, 1],
    [1, 3, 4, 1],
    [1, 3, 7, 1],
    [1, 3, 4, 1],
    [1, 3, 1, 1],
    [1, 3, 10, 1],
    [1, 3, 3, 1],
    [1, 3, 4, 1],
    [1, 3, 7, 1],
    [1, 3, 5, 1],
    [1, 3, 6, 1],
    [1, 3, 1, 1],
    [1, 3, 6, 1],
    [1, 3, 10, 1],
    [1, 3, 2, 1],
    [1, 3, 4, 1],
    [1, 3, 2, 1],
    [1, 3, 1, 1],
    [1, 3, 5, 1],
    [1, 2, 4, 1],
    [1, 2, 2, 1],
    [1, 2, 8, 1],
    [1, 2, 3, 1],
    [1, 2, 1, 1],
    [1, 2, 9, 1],
    [1, 2, 10, 1],
    [1, 2, 9, 1],
    [1, 2, 4, 1],
    [1, 2, 5, 1],
    [1, 2, 0, 1],
    [1, 2, 9, 1],
    [1, 2, 9, 1],
    [1, 2, 0, 1],
    [1, 2, 1, 1],
    [1, 2, 1, 1],
    [1, 2, 4, 1],
    [1, 1, 0, 1],
    [1, 1, 2, 1],
    [1, 1, 2, 1],
    [1, 1, 5, 1],
    [1, 1, 3, 1],
    [1, 1, 10, 1],
    [1, 1, 6, 1],
    [1, 1, 0, 1],
    [1, 1, 8, 1],
    [1, 1, 6, 1],
    [1, 1, 4, 1],
    [1, 1, 9, 1],
    [1, 1, 9, 1],
    [1, 1, 4, 1],
    [1, 1, 2, 1],
    [1, 1, 9, 1],
    [1, 1, 0, 1],
    [1, 1, 8, 1],
    [1, 1, 6, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 5, 1],
]


def predict(x, beta):
    """assumes that the first element of c is 1"""
    return dot(x, beta)


[1, 49, 4, 0]


def error(x, y, beta):
    return predict(x, beta) - y


def squared_error(x, y, beta):
    return error(x, y, beta) ** 2


x = [1, 2, 3]
y = 30
beta = [4, 4, 4]  # so prediction = 4 + 8 + 12 = 24

assert error(x, y, beta) == -6
assert squared_error(x, y, beta) == 36


def sqerror_gradient(x, y, beta):
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]


assert sqerror_gradient(x, y, beta) == [-12, -24, -36]


def least_squares_fit(xs, ys, learning_rate=0.001, num_steps=1000, batch_size=1):
    """
    Find the beta that minimizes the sum of squared errors
    assuming the model y = dot(x, beta).
    """
    # Start with a random guess
    guess = [random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start : start + batch_size]
            batch_ys = ys[start : start + batch_size]

            gradient = vector_mean(
                [sqerror_gradient(x, y, guess) for x, y in zip(batch_xs, batch_ys)]
            )

            guess = gradient_step(guess, gradient, -learning_rate)

    return guess


def multiple_r_squared(xs, ys, beta):
    sum_of_squared_errors = sum(error(x, y, beta) ** 2 for x, y in zip(xs, ys))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(ys)


x = TypeVar("X")
Stat = TypeVar("Stat")


def bootstrap_sample(data):
    """randomly samples len(data) elements with replacement"""
    return [random.choice(data) for _ in data]


def bootstrap_statistics(data, stats_fn, num_samples):
    """evaluates stats_fn on num_samples bootstrap samples from data"""
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]


# 101 points all very close to 100
close_to_100 = [99.5 + random.random() for _ in range(101)]

# 101 points, 50 of them near 0, 50 of them near 200
far_from_100 = (
    [99.5 + random.random()]
    + [random.random() for _ in range(50)]
    + [200 + random.random() for _ in range(50)]
)

medians_close = bootstrap_statistics(close_to_100, median, 100)

medians_far = bootstrap_statistics(far_from_100, median, 100)

assert standard_deviation(medians_close) < 1
assert standard_deviation(medians_far) > 90


def p_value(beta_hat_j, sigma_hat_j):
    if beta_hat_j > 0:
        # if the coefficient is positive, we need to compute twice the
        # probability of seeing an even *larger* value
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        # otherwise twice the probability of seeing a *smaller* value
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)


assert p_value(30.58, 1.27) < 0.001  # constant term
assert p_value(0.972, 0.103) < 0.001  # num_friends
assert p_value(-1.865, 0.155) < 0.001  # work_hours
assert p_value(0.923, 1.249) > 0.4  # phd


# alpha is a *hyperparameter* controlling how harsh the penalty is
# sometimes it's called "lambda" but that already means something in Python
def ridge_penalty(beta, alpha):
    return alpha * dot(beta[1:], beta[1:])


def squared_error_ridge(x, y, beta, alpha):
    """estimate error plus ridge penalty on beta"""
    return error(x, y, beta) ** 2 + ridge_penalty(beta, alpha)


def ridge_penalty_gradient(beta, alpha):
    """gradient of just the ridge penalty"""
    return [0.0] + [2 * alpha * beta_j for beta_j in beta[1:]]


def sqerror_ridge_gradient(x, y, beta, alpha):
    """
    the gradient corresponding to the ith squared error term
    including the ridge penalty
    """
    return add(sqerror_gradient(x, y, beta), ridge_penalty_gradient(beta, alpha))


learning_rate = 0.001


def least_squares_fit_ridge(xs, ys, alpha, learning_rate, num_steps, batch_size=1):
    # Start guess with mean
    guess = [random.random() for _ in xs[0]]

    for i in range(num_steps):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start : start + batch_size]
            batch_ys = ys[start : start + batch_size]

            gradient = vector_mean(
                [
                    sqerror_ridge_gradient(x, y, guess, alpha)
                    for x, y in zip(batch_xs, batch_ys)
                ]
            )
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess


def lass_penalty(beta, alpha):
    return alpha * sum(abs(beta_i) for beta_i in beta[1:])


def main():
    random.seed(0)

    # I used trial and error to choose niters and step_size.
    # This will run for a while.
    learning_rate = 0.001

    beta = least_squares_fit(inputs, daily_minutes_good, learning_rate, 5000, 25)
    assert 30.50 < beta[0] < 30.70  # constant
    assert 0.96 < beta[1] < 1.00  # num friends
    assert -1.89 < beta[2] < -1.85  # work hours per day
    assert 0.91 < beta[3] < 0.94  # has PhD

    assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta) < 0.68

    def estimate_sample_beta(pairs):
        x_sample = [x for x, _ in pairs]
        y_sample = [y for _, y in pairs]

        beta = least_squares_fit(x_sample, y_sample, learning_rate, 5000, 25)
        print("Bootstrap sample", beta)
        return beta

    # This will take a couple of minutes!
    bootstrap_betas = bootstrap_statistics(
        list(zip(inputs, daily_minutes_good)), estimate_sample_beta, 100
    )

    bootstrap_standard_errors = [
        standard_deviation([beta[i] for beta in bootstrap_betas]) for i in range(4)
    ]

    print(bootstrap_standard_errors)

    # [1.272,    # constant term, actual error = 1.19
    #  0.103,    # num_friends,   actual error = 0.080
    #  0.155,    # work_hours,    actual error = 0.127
    #  1.249]    # phd,           actual error = 0.998

    random.seed(0)
    beta_0 = least_squares_fit_ridge(
        inputs, daily_minutes_good, 0.0, learning_rate, 5000, 25  # alpha
    )
    # [30.51, 0.97, -1.85, 0.91]
    assert 5 < dot(beta_0[1:], beta_0[1:]) < 6
    assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0) < 0.69

    beta_0_1 = least_squares_fit_ridge(
        inputs, daily_minutes_good, 0.1, learning_rate, 5000, 25  # alpha
    )
    # [30.8, 0.95, -1.83, 0.54]
    assert 4 < dot(beta_0_1[1:], beta_0_1[1:]) < 5
    assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0_1) < 0.69

    beta_1 = least_squares_fit_ridge(
        inputs, daily_minutes_good, 1, learning_rate, 5000, 25  # alpha
    )
    # [30.6, 0.90, -1.68, 0.10]
    assert 3 < dot(beta_1[1:], beta_1[1:]) < 4
    assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_1) < 0.69

    beta_10 = least_squares_fit_ridge(
        inputs, daily_minutes_good, 10, learning_rate, 5000, 25  # alpha
    )
    # [28.3, 0.67, -0.90, -0.01]
    assert 1 < dot(beta_10[1:], beta_10[1:]) < 2
    assert 0.5 < multiple_r_squared(inputs, daily_minutes_good, beta_10) < 0.6


if __name__ == "__main__":
    main()
