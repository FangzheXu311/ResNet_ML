import my_resnet
import argparse
import info
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
'''breast_data, chest_data, derma_data, oct_data, organ_axial_data, organ_coronal_data, \
    organ_sagittal_data, path_data, pneumonia_data, retina_data = my_resnet.load_data()
'''

def model_fit(train_model, data, lr, epochs, device="cuda:0"):
    train_loader, val_loader, test_loader, flag, classes = my_resnet.data_pre_process(data, 64)

    if len(train_loader.dataset[0][0].shape) != 2:
        in_channels = train_loader.dataset[0][0].shape[0]
    else:
        in_channels = 1

    data_loader = [train_loader, val_loader, test_loader]

    if train_model == 'ResNet_18':
        model = my_resnet.ResNet(my_resnet.Baseblock, info.num_blocks_18, in_channels=in_channels, num_classes=classes[0])
    elif train_model == 'ResNet_50':
        model = my_resnet.ResNet(my_resnet.Bottleneck, info.num_blocks_50, in_channels=in_channels, num_classes=classes[0])
    elif train_model == 'ResNet_18_drop':
        model = my_resnet.ResNet_drop(my_resnet.Baseblock, info.num_blocks_18, in_channels=in_channels, num_classes=classes[0])
    elif train_model == 'ResNet_50_branch':
        model = my_resnet.ResNet(my_resnet.Bottleneck_2branch, info.num_blocks_50, in_channels=in_channels, num_classes=classes[0])

    if flag == 'binary':
        loss_func = nn.BCEWithLogitsLoss()
    elif flag == 'multi':
        loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.7)

    metrics = my_resnet.model_run(epochs, data_loader, model, loss_func, optimizer, flag, device=device)
    return metrics


def main(dir_path, data_name, model_type, learning_rate, epochs):
    '''breast_data, chest_data, derma_data, oct_data, organ_axial_data, organ_coronal_data, \
    organ_sagittal_data, path_data, pneumonia_data, retina_data = my_resnet.load_data()'''
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    data = np.load(dir_path+'\\'+data_name+'.npz')
    metrics = model_fit(model_type, data, learning_rate, epochs, device)
    plt.plot(range(epochs),metrics['val auc'])
    plt.title("validation auc")
    plt.figure()
    plt.plot(range(epochs), metrics['val acc'])
    plt.title("validation accuracy")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST')
    parser.add_argument('--data_name',
                        default='pathmnist',
                        help='subset of MedMNIST',
                        type=str)
    parser.add_argument('--dir_path',
                        default='.\\Machine Learning\\MedMnist\\dataset',
                        type=str)
    parser.add_argument('--model_type',
                        default='ResNet_18',
                        type=str)
    parser.add_argument('--learning_rate',
                        default=0.001,
                        type=float)
    parser.add_argument('--epochs',
                        default=20,
                        type=int)
    args = parser.parse_args()
    data_name = args.data_name
    model_type = args.model_type
    dir_path = args.dir_path
    learning_rate = args.learning_rate
    epochs = args.epochs
    main(dir_path, data_name, model_type, learning_rate, epochs)
