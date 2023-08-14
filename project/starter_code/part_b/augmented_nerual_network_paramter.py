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


class AugmentedAutoEncoder(nn.Module):
    def __init__(self, num_question, k=200, l=50, act_fn='sigmoid'):  # Added act_fn parameter
        super(AugmentedAutoEncoder, self).__init__()

        self.act_fn_name = act_fn

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, l)
        self.i = nn.Linear(l, num_question)

    def activation_function(self, x):
        if self.act_fn_name == 'sigmoid':
            return torch.sigmoid(x)
        elif self.act_fn_name == 'relu':
            return torch.relu(x)
        else:
            raise ValueError(f"Activation function {self.act_fn_name} not recognized")

    def get_weight_norm(self):
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        i_w_norm = torch.norm(self.i.weight, 2) ** 2
        return g_w_norm + h_w_norm + i_w_norm

    def forward(self, inputs):
        intermediate_result = self.activation_function(self.g(inputs))
        intermediate_result2 = self.activation_function(self.h(intermediate_result))
        final_result = self.i(intermediate_result2)
        out = torch.sigmoid(final_result)
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, B):
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
    num_batches = int(np.ceil(num_student / B))

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for b in range(num_batches):
            start_idx = b * B
            end_idx = (b + 1) * B

            inputs = Variable(zero_train_data[start_idx:end_idx])
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to compute the gradient of valid entries.
            nan_mask = torch.isnan(train_data[start_idx:end_idx])

            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.)
            if lamb and lamb > 0:
                loss += (lamb / 2) * model.get_weight_norm()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        if epoch == 19:
            print("Epoch: {} \tTraining Cost: {:.6f}\t "
                  "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        Accuracy_validation.append(valid_acc)
        Loss_for_traing.append(train_loss)

        if valid_acc > maximum_accuracy:
            maximum_accuracy = valid_acc
    return Loss_for_traing, Accuracy_validation, maximum_accuracy


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
    plt.savefig('./partb_3(d)')
    plt.show()


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    torch.manual_seed(100)

    # Define possible hyperparameter values
    K_values = [100, 200]
    L_values = [30, 50, 70]
    lr_values = [0.01, 0.1]
    B_values = [50, 100, 200]
    activation_functions = ['sigmoid']
    lamb = 0.
    num_epoch = 20
    number_of_ques = train_matrix.shape[-1]

    results = {}  # To store validation and test accuracies

    for k in K_values:
        for l in L_values:
            for lr in lr_values:
                for B in B_values:
                    for act_fn in activation_functions:
                        print(f"Training with k={k}, l={l}, lr={lr}, B={B}, act_fn={act_fn}")

                        model = AugmentedAutoEncoder(number_of_ques, k=k, l=l, act_fn=act_fn)

                        _, _, _ = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, B)

                        val_accuracy = evaluate(model, zero_train_matrix, valid_data)
                        test_accuracy = evaluate(model, zero_train_matrix, test_data)

                        # Store the results
                        results[(k, l, lr, B, act_fn)] = (val_accuracy, test_accuracy)

    # Print the results for each combination
    for hyperparams, accuracies in results.items():
        print(
            f"For k={hyperparams[0]}, l={hyperparams[1]}, lr={hyperparams[2]}, B={hyperparams[3]}, act_fn={hyperparams[4]}:")
        print(f"Validation Accuracy: {accuracies[0]:.3f}, Test Accuracy: {accuracies[1]:.3f}\n")

    # Find the best hyperparameters based on validation accuracy
    best_hyperparams = max(results, key=results.get)
    print(
        f"Best Validation Accuracy is {results[best_hyperparams][0]:.3f} with hyperparameters k={best_hyperparams[0]}, l={best_hyperparams[1]}, lr={best_hyperparams[2]}, B={best_hyperparams[3]}, act_fn={best_hyperparams[4]}")


if __name__ == "__main__":
    main()

    # Best Validation Accuracy is 0.690 with hyperparameters k=100, l=70, lr=0.1, B=50, act_fn=sigmoid
