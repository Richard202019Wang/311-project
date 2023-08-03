import matplotlib.pyplot as plt
from utils import *

import numpy as np

from starter_code.utils import load_train_csv, load_valid_csv, load_public_test_csv, load_train_sparse


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    # create an interaction matrix of size (max(user_ids)+1) x (max(question_ids)+1) filled with NaNs
    interaction_matrix = np.full((np.max(data['user_id']) + 1, np.max(data['question_id']) + 1), np.nan)
    # fill the interaction matrix with is_correct values at corresponding indices
    interaction_matrix[data['user_id'], data['question_id']] = data['is_correct']
    # this produces a matrix representing the difference between user ability and question difficulty
    ability_diff_matrix = theta.reshape((-1, 1)) - beta.reshape((1, -1))
    # compute the negative log-likelihood
    log_lklihood = np.nansum(interaction_matrix * ability_diff_matrix - np.log(1 + np.exp(ability_diff_matrix)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Define an empty array for each parameter's gradient
    theta_gradient = np.zeros_like(theta)
    beta_gradient = np.zeros_like(beta)

    # Loop over all interactions in the data
    for user_id, question_id, is_correct in zip(data['user_id'], data['question_id'], data['is_correct']):
        # Compute the difference in ability and difficulty
        ability_diff = theta[user_id] - beta[question_id]

        # Compute the sigmoid of the ability difference
        prob = sigmoid(ability_diff)

        # Compute the gradient of the log likelihood with respect to ability and difficulty
        gradient = is_correct - prob

        # Accumulate the gradient
        theta_gradient[user_id] += gradient
        beta_gradient[question_id] += -gradient

    # Apply gradient ascent to update the parameters
    theta += lr * theta_gradient
    beta += lr * beta_gradient
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # create an interaction matrix of size (max(user_ids)+1) x (max(question_ids)+1) filled with NaNs
    interaction_matrix = np.full((np.max(data['user_id']) + 1, np.max(data['question_id']) + 1), np.nan)
    # fill the interaction matrix with is_correct values at corresponding indices
    interaction_matrix[data['user_id'], data['question_id']] = data['is_correct']
    # TODO: Initialize theta and beta.
    theta = np.zeros(len(interaction_matrix))
    beta = np.zeros(len(interaction_matrix[0]))

    val_acc_lst = []
    lld_train = []
    lld_validation = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        lld_train.append(neg_lld)
        neg_lld_validation = neg_log_likelihood(data=val_data, theta=theta, beta=beta)
        lld_validation.append(neg_lld_validation)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, lld_train, lld_validation


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lrt_rate = 0.015
    iterations = 30
    theta, beta, val_acc_lst, lld_train, lld_validation = irt(train_data, val_data, lrt_rate, iterations)
    plt.plot(range(1, iterations + 1), lld_train)
    plt.ylabel("Training Negative Log-Likelihood")
    plt.xlabel("Training Iterations")
    plt.title("Training Negative Log-Likelihood vs. Training Iterations")
    plt.savefig('./part4_2(b)_training')
    plt.show()

    plt.plot(range(1, iterations + 1), lld_validation)
    plt.ylabel("Validation Negative Log-Likelihood")
    plt.xlabel("Validation Iterations")
    plt.title("Validation Negative Log-Likelihood vs. Validation Iterations")
    plt.savefig('./part4_2(b)_validation')
    plt.show()

    print("------------------------------")
    print(f"The final Test Accuracies are {evaluate(test_data, theta, beta)}\n"
          f"Also the final Validation Accuracies are {evaluate(val_data, theta, beta)}")
    print("------------------------------")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    sorted_theta = np.sort(theta)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_theta, sigmoid(sorted_theta - beta[1]), label='question j_1 = 1', color='pink')
    plt.plot(sorted_theta, sigmoid(sorted_theta - beta[100]), label='question j_2 = 100', color='yellow')
    plt.plot(sorted_theta, sigmoid(sorted_theta - beta[1000]), label='question j_3 = 1000', color='grey')
    plt.ylabel("Probability of the correct response")
    plt.xlabel("Theta")
    plt.title("p(c_ij = 1) vs Function of theta")
    plt.legend()
    plt.savefig('./part4_2(d)_validation')
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
