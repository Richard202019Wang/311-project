# TODO: complete this file.
from sklearn.impute import KNNImputer
from item_response import irt, sigmoid
from neural_network import AutoEncoder, train
import numpy as np
import torch
from torch.autograd import Variable
from starter_code.utils import load_train_csv, load_valid_csv, load_public_test_csv, load_train_sparse


def resample_data(data, size):
    elements = np.random.randint(0, len(data['user_id']), size)
    resampled_data = {
        'user_id': [data['user_id'][i] for i in elements],
        'question_id': [data['question_id'][i] for i in elements],
        'is_correct': [data['is_correct'][i] for i in elements]
    }
    return resampled_data


def create_matrix(data_dict):
    max_user = max(data_dict['user_id']) + 1
    max_question = max(data_dict['question_id']) + 1
    matrix = np.full((max_user, max_question), np.nan)
    for i in range(len(data_dict['user_id'])):
        matrix[data_dict['user_id'][i], data_dict['question_id'][i]] = data_dict['is_correct'][i]
    return matrix


def train_base_models(train_data, val_data, hyperparameters):
    train_data_knn = resample_data(train_data, len(train_data['user_id']))
    train_data_irt = resample_data(train_data, len(train_data['user_id']))
    train_data_nn = resample_data(train_data, len(train_data['user_id']))

    # Train k-Nearest Neighbors
    train_mat_knn = create_matrix(train_data_knn)
    k = hyperparameters['knn_k']
    nbrs = KNNImputer(n_neighbors=k)
    knn_mat = nbrs.fit_transform(train_mat_knn)

    # Train Item Response Theory
    lr = hyperparameters['irt_lr']
    iterations = hyperparameters['irt_iterations']
    theta, beta, val_acc_lst, lld_train, lld_validation = irt(train_data_irt, val_data, lr, iterations)

    # Train Neural Network
    train_mat_nn = create_matrix(train_data_nn)
    nn_model = AutoEncoder(train_mat_nn.shape[1], hyperparameters['nn_k'])
    nn_zero_train_data = np.nan_to_num(train_mat_nn, nan=0.0) # Assuming you need a zero-padded version for training
    nn_zero_train_data = torch.FloatTensor(nn_zero_train_data)
    train_loss, acc_val, max_acc = train(nn_model, hyperparameters['nn_lr'], hyperparameters.get('nn_lambda', 0),
                                        torch.FloatTensor(train_mat_nn), nn_zero_train_data, val_data, hyperparameters['nn_epochs'])

    return knn_mat, theta, beta, nn_model, train_mat_nn


def ensemble_predict(models, test_data):
    knn_mat, theta, beta, nn_model, train_mat = models

    pred_knn = knn_mat[test_data['user_id'], test_data['question_id']]
    pred_irt = [sigmoid(theta[user] - beta[question]) >= 0.5 for user, question in zip(test_data['user_id'], test_data['question_id'])]
    pred_nn = []
    for idx in range(len(test_data["user_id"])):
        user = test_data["user_id"][idx]
        question = test_data["question_id"][idx]
        user_input = Variable(torch.tensor(train_mat[user], dtype=torch.float32)).unsqueeze(0)
        out = nn_model(user_input)
        prediction = out.squeeze()[question].item() >= 0.5
        pred_nn.append(prediction)

    # Average the predictions
    pred_ensemble = (np.array(pred_knn) + np.array(pred_irt) + np.array(pred_nn)) / 3
    return pred_ensemble >= 0.5


def main():
    train_data = load_train_csv("../data")

    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    hyperparameters = {
        'knn_k': 11,
        'irt_lr': 0.01,
        'irt_iterations': 30,
        'nn_k': 200,
        'nn_lr': 0.01,
        'nn_epochs': 20
    }

    # Train base models
    models = train_base_models(train_data, val_data, hyperparameters)

    # Evaluate ensemble on validation and test data
    val_pred = ensemble_predict(models, val_data)
    test_pred = ensemble_predict(models, test_data)

    val_accuracy = np.mean(val_pred == val_data['is_correct'])
    test_accuracy = np.mean(test_pred == test_data['is_correct'])

    print(f"Ensemble Validation Accuracy: {val_accuracy}")
    print(f"Ensemble Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    torch.manual_seed(100)
    main()
