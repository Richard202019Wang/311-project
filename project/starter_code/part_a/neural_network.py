from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt

from starter_code.utils import load_train_sparse, load_valid_csv, load_public_test_csv


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        intermediate_result = self.g(inputs)  # Apply the g function
        intermediate_result = torch.sigmoid(intermediate_result)  # Apply sigmoid activation
        final_result = self.h(intermediate_result)  # Apply the h function
        out = torch.sigmoid(final_result)  # Apply sigmoid activation again
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.
    Loss_for_traing = []
    Accuracy_validation = []
    maximum_accuracy = 0.0
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            if lamb and lamb > 0:
                loss += (lamb / 2) * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        Accuracy_validation.append(valid_acc)
        Loss_for_traing.append(train_loss)

        if valid_acc > maximum_accuracy:
            maximum_accuracy = valid_acc
    return Loss_for_traing, Accuracy_validation, maximum_accuracy
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def plot_results(best_train_loss, best_val_accuracy, num_epoch):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    epochs = range(num_epoch)
    ax1.plot(epochs, best_train_loss, 'r-', label='Training Loss')
    ax1.set_title('Training Loss with different Epoch')
    ax1.set_xlabel('Epoch(k)')
    ax1.set_ylabel('Training Loss')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(epochs, best_val_accuracy, 'b-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy with different Epoch')
    ax2.set_xlabel('Epoch(k)')
    ax2.set_ylabel('Validation Accuracy')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('./part4_3(d)')
    plt.show()


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    torch.manual_seed(100)
    # Set model hyperparameters.
    k = [10, 50, 100, 200, 500]
    model = None
    number_of_ques = train_matrix.shape[-1]

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 20
    lamb = 0.

    best_k = 0
    best_model = None
    best_train_loss = None
    best_val_accuracy = [0]

    for each_k in k:
        print(f"Currently, we are training at k = {each_k}")
        model = AutoEncoder(number_of_ques, each_k)
        Loss_for_traing, Accuracy_validation, maximum_accuracy = \
            train(model, lr, lamb, train_matrix, zero_train_matrix,
              valid_data, num_epoch)

        if maximum_accuracy > max(best_val_accuracy):
            best_k = each_k
            best_val_accuracy = Accuracy_validation
            best_train_loss = Loss_for_traing
            best_model = model

    print(f'The best Validation accuracy is {max(best_val_accuracy)} under the k value {best_k}')
    test_accuracy = evaluate(best_model, zero_train_matrix, test_data)
    print(f"The test accuracy under the k value {best_k} is {test_accuracy}")

    ## Plot the graphs ##
    plot_results(best_train_loss, best_val_accuracy, num_epoch)

    ### e ####
    regularization_penalties = (0.001, 0.01, 0.1, 1)
    regularization_penalties_results = {}

    for penalty in regularization_penalties:
        print(f'Training with penalty = {penalty}')
        model = AutoEncoder(number_of_ques, best_k)
        Loss_for_traing, Accuracy_validation, maximum_accuracy = train(model, lr, penalty, train_matrix, zero_train_matrix,
                                           valid_data, num_epoch)
        maximum_validation_accuracy = max(Accuracy_validation)
        test_accuracy = evaluate(model, zero_train_matrix, test_data)
        regularization_penalties_results[penalty] = (maximum_validation_accuracy, test_accuracy)

    regularization_penalties_str = [str(l) for l in regularization_penalties]
    val_accs = [res[0] for res in regularization_penalties_results.values()]
    test_accs = [res[1] for res in regularization_penalties_results.values()]

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(regularization_penalties_str, val_accs, alpha=0.7, label='Validation Accuracy')
    bars2 = plt.bar(regularization_penalties_str, test_accs, alpha=0.7, label='Test Accuracy', bottom=val_accs)
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation and Test Accuracy for Different Lambda Values')

    for bar, acc in zip(bars1, val_accs):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval - 0.05, round(acc, 3), ha='center', color='white')

    for bar, acc, val_acc in zip(bars2, test_accs, val_accs):
        yval = bar.get_height() + val_acc
        plt.text(bar.get_x() + bar.get_width() / 2, yval - 0.05, round(acc, 3), ha='center', color='white')

    plt.savefig('./part4_3(e)')
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
