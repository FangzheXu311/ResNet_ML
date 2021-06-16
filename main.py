import numpy as np
import torch
from torch import nn
import os
import my_3DUnet
from dataset.fracnet_dataset import FracNetTrainDataset, FracNetInferenceDataset
from dataset import transforms as tsfm
from tqdm import tqdm
import argparse
from utils.metrics import dice, recall, precision, fbeta_score, accuracy
from my_3DUnet import MixLoss, DiceLoss
from my_3DUnet import _predict_single_image, _make_submission_files
import nibabel as nib
from sklearn import metrics
import matplotlib.pyplot as plt
from model.model import resunet3d

DEFAULT_KEY_FP = (0.5, 1., 2., 4., 8.)


def get_froc(pred, target, batch_size):
    total_num_images = batch_size
    pred = pred.reshape(-1,1)
    target = target.reshape(-1,1)
    num_detected = np.sum(target)
    total_candidates = pred.shape[0]
    fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label=1)
    fps = fpr*(total_candidates-num_detected)/total_num_images
    FROC = 0
    for i in range(len(DEFAULT_KEY_FP)):
        FROC += tpr[fps == DEFAULT_KEY_FP[i]].mean()
    return FROC/len(DEFAULT_KEY_FP)


def load_data(dir_path, batch_size, num_workers):
    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]
    train_image_dir = dir_path + "/train/images"
    train_label_dir = dir_path + "/train/labels"
    val_image_dir = dir_path + "/val/images"
    val_label_dir = dir_path + "/val/labels"
    ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,
                                   transforms=transforms)
    dl_train = FracNetTrainDataset.get_dataloader(ds_train, batch_size, False,
                                                  num_workers)
    ds_val = FracNetTrainDataset(val_image_dir, val_label_dir,
                                 transforms=transforms)
    dl_val = FracNetTrainDataset.get_dataloader(ds_val, batch_size, False,
                                                num_workers)
    return dl_train, dl_val


def train_loop(model, train_loader, loss_func, optimizer, device):
    length = len(train_loader.dataset)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.train()
    loss_num = 0
    for X,y in tqdm(train_loader):
        X, y = X.to(device), y.to(device)
        X.requires_grad_(True)
        pred = model(X)
        loss = loss_func(pred, y)
        loss_num += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss_num/length)


def val_loop(model, val_loader, loss_func, thresh, batch_size, device):
    length = len(val_loader.dataset)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    acc, prec, rec, loss_sum = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
       for X, y in tqdm(val_loader):
       #for (X,y) in val_loader.dataset:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss_sum += loss_func(pred,y).item()
            acc += accuracy(pred, y).item()
            prec += precision(pred, y, thresh).item()
            rec += recall(pred, y, thresh).item()
            pred = pred.cpu().numpy()
            #y = y.cpu().numpy()
            #FROC += get_froc(pred, y, batch_size)
    print("Val loss: %.3f, acc: %.3f, precision: %.3f, recall: %.3f\n"
          %(loss_sum/length, acc/length, prec/length, rec/length))


def predict(model, test_dir, postprocess, prob_thresh, bone_thresh, size_thresh, pred_dir):
    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]
    batch_size = 16
    num_workers = 4
    image_path_list = sorted([os.path.join(test_dir, file)
                              for file in os.listdir(test_dir) if "nii" in file])
    image_id_list = [os.path.basename(path).split("-")[0]
                     for path in image_path_list]
    progress = tqdm(total=len(image_id_list))
    for image_id, image_path in zip(image_id_list, image_path_list):
        dataset = FracNetInferenceDataset(image_path, transforms=transforms)
        dataloader = FracNetInferenceDataset.get_dataloader(dataset,
            batch_size, num_workers)
        pred_arr = _predict_single_image(model, dataloader, postprocess,
            prob_thresh, bone_thresh, size_thresh)
        pred_image = _make_submission_files(pred_arr,
            dataset.image_affine)
        pred_path = os.path.join(pred_dir, f"{image_id}.nii.gz")
        nib.save(pred_image, pred_path)
        progress.update()



def model_fit(model, train_loader, val_loader, loss_func, optimizer, num_epochs, batch_size, model_path, device="cuda:0"):
    for epoch in range(50,num_epochs):
        num = epoch+1
        print("At Epoch"+str(num)+"\n")
        train_loop(model, train_loader, loss_func, optimizer, device)
        save_path = model_path + "/epoch_" +str(epoch+1) + "_model.pth.tar"
        torch.save(model.state_dict(), save_path)
        val_loop(model, val_loader, loss_func, thresh=0.1, batch_size=batch_size, device=device)
        torch.cuda.empty_cache()


def main(args):
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    dir_path = args.dir_path
    num_workers = args.num_workers
    lr = args.learning_rate
    model_path = args.model_path
    output_val = args.output_val
    output_test = args.output_test
    postprocess = args.postprocess
    prob_thresh = args.prob_thresh
    bone_thresh = args.bone_thresh
    size_thresh = args.size_thresh
    train_loader, val_loader = load_data(dir_path, batch_size, num_workers)
    test_dir = dir_path + '/test/images'
    val_dir = dir_path + '/val/images'
    #model = my_3DUnet.UNet_3D(my_3DUnet.EncoderBlock, my_3DUnet.DecoderBlock, 1, 1, 'Upsample')
    model = my_3DUnet.UNet(1,1,first_out_channels=16)
    #model = resunet3d(1,1)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #loss_func = my_3DUnet.SoftDiceLoss()
    loss_func = MixLoss(nn.BCEWithLogitsLoss(), 0.5, DiceLoss(), 1)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #model.load_state_dict(torch.load('../saved_models_off/epoch_50_model.pth.tar'))
    model_fit(model, train_loader, val_loader, loss_func, optimizer, num_epochs, batch_size, model_path, device)
    predict(model, val_dir, postprocess, prob_thresh, bone_thresh, size_thresh, output_val)
    predict(model, test_dir, postprocess, prob_thresh, bone_thresh, size_thresh, output_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run model for UNet")
    #parser.add_argument('--dir_path', default='./Data', type=str)
    parser.add_argument('--dir_path', default="../ribfrac-data", type=str)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--model_path', default='../saved_models_off', type=str)
    parser.add_argument('--output_val', default='../Output_val_off', type=str)
    parser.add_argument('--output_test', default='../Output_test_off', type=str)
    parser.add_argument('--prob_thresh', default=0.1, type=float)
    parser.add_argument('--bone_thresh', default=300, type=int)
    parser.add_argument('--size_thresh', default=100, type=int)
    parser.add_argument('--postprocess', default=True, type=bool)
    args = parser.parse_args()
    main(args)