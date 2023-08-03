import numpy as np
from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt

from starter_code.utils import sparse_matrix_evaluate, load_train_sparse, load_valid_csv, load_public_test_csv


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(np.transpose(matrix))
    acc = sparse_matrix_evaluate(valid_data, np.transpose(mat))
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # Declare necessary variables
    k_vals = [1, 6, 11, 16, 21, 26]
    users_accuracies = []
    questions_accuracies = []
    user_best_k = 0
    question_best_k = 0
    best_user_accuracy = 0
    best_question_accuracy = 0

    # User Based Collaborative Filtering
    for k in k_vals:
        current_accuracy = knn_impute_by_user(sparse_matrix, val_data, k)
        users_accuracies.append(current_accuracy)
        if current_accuracy > best_user_accuracy:
            best_user_accuracy = current_accuracy
            user_best_k = k

    # Question Based Collaborative Filtering
    for k in k_vals:
        current_accuracy = knn_impute_by_item(sparse_matrix, val_data, k)
        questions_accuracies.append(current_accuracy)
        if current_accuracy > best_question_accuracy:
            best_question_accuracy = current_accuracy
            question_best_k = k

    # Plot the accuracies for user based collaborative filtering
    plt.plot(k_vals, users_accuracies, 's--g')
    plt.title('Accuracy vs k by User')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.savefig('./part4_1(a)')
    plt.show()

    # Print the best k for user based collaborative filtering
    print(f"\nWhen choosing the k = {user_best_k}, it has the highest performance on validation data and test data \n"
          f"with test accuracy as {knn_impute_by_user(sparse_matrix, test_data, user_best_k)} and validation accuracy as {best_user_accuracy} \n")

    # Plot the accuracies for question based collaborative filtering
    plt.plot(k_vals, questions_accuracies, 's--g')
    plt.title('Accuracy vs k By Question')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.savefig('./part4_1(c)')
    plt.show()

    # Print the best k for question based collaborative filtering
    print(f"\nWhen choosing the k = {question_best_k}, it has the highest performance on validation data and test data \n"
          f"with test accuracy as {knn_impute_by_item(sparse_matrix, test_data, question_best_k)} and validation accuracy as {best_question_accuracy} \n")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
