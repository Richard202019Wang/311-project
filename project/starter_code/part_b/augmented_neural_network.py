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
    def __init__(self, num_question, k=200, l=50):  # Added an extra dimension 'l' for the new hidden layer
        super(AugmentedAutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, l)
        self.i = nn.Linear(l, num_question)  # New hidden layer

    def get_weight_norm(self):
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        i_w_norm = torch.norm(self.i.weight, 2) ** 2  # New hidden layer weight norm
        return g_w_norm + h_w_norm + i_w_norm  # Include new weight norm in regularization

    def forward(self, inputs):
        intermediate_result = self.g(inputs)
        intermediate_result = torch.sigmoid(intermediate_result)
        intermediate_result2 = self.h(intermediate_result)  # Pass through second hidden layer
        intermediate_result2 = torch.sigmoid(intermediate_result2)
        final_result = self.i(intermediate_result2)  # Pass through the third layer
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
    plt.savefig('./partb_3(d)')
    plt.show()


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    torch.manual_seed(100)
    # Set model hyperparameters.
    B_values = [1, 50, 100, 200, 500, 1000]
    l = 30  # You can tune this hyperparameter too if you like
    model = None
    number_of_ques = train_matrix.shape[-1]

    # Set optimization hyperparameters.
    lr = 0.1
    num_epoch = 20
    lamb = 0.

    best_batch_size = 0
    best_model = None
    best_train_loss = None
    best_val_accuracy = [0]

    batch_size_results = {}

    for B in B_values:
        print(f"Currently, we are training with batch size = {B}")

        # Use AugmentedAutoEncoder here instead of AutoEncoder
        model = AugmentedAutoEncoder(number_of_ques, B, l)

        Loss_for_traing, Accuracy_validation, maximum_accuracy = \
            train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, B)

        test_accuracy = evaluate(model, zero_train_matrix, test_data)
        batch_size_results[B] = (maximum_accuracy, test_accuracy)

    B_str = [str(b) for b in B_values]
    val_accs = [res[0] for res in batch_size_results.values()]
    test_accs = [res[1] for res in batch_size_results.values()]

    # Get batch size with the best test accuracy
    best_batch_size = max(batch_size_results, key=lambda k: batch_size_results[k][1])
    best_test_accuracy = batch_size_results[best_batch_size][1]
    print(
        f"The best batch size based on test accuracy is {best_batch_size} with an accuracy of {best_test_accuracy:.3f}")

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(B_str, val_accs, alpha=0.7, label='Validation Accuracy')
    bars2 = plt.bar(B_str, test_accs, alpha=0.7, label='Test Accuracy', bottom=val_accs)
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation and Test Accuracy for Different Batch Sizes')

    for bar, acc in zip(bars1, val_accs):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval - 0.05, round(acc, 3), ha='center', color='white')

    for bar, acc, val_acc in zip(bars2, test_accs, val_accs):
        yval = bar.get_height() + val_acc
        plt.text(bar.get_x() + bar.get_width() / 2, yval - 0.05, round(acc, 3), ha='center', color='white')

    plt.savefig('./partb(a)_batch_sizes')
    plt.show()

# rest of the code


if __name__ == "__main__":
    main()
