import torch
from torch import nn
from torch.nn import functional as F
import torch.optim
import matplotlib
import os
import numpy as np
from dataset.data_split import split_data,Concat
from scipy import ndimage
from skimage.measure import label, regionprops
import nibabel as nib
from skimage.morphology import disk, remove_small_objects


class MixLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
    def forward(self, x, y):
        lf, lfw = [], []
        for i, v in enumerate(self.args):
            if i % 2 == 0:
                lf.append(v)
            else:
                lfw.append(v)
        mx = sum([w*l(x,y) for l, w in zip(lf, lfw)])
        return mx

class DiceLoss(nn.Module):
    def __init__(self, image=False):
        super().__init__()
        self.image = image
    def forward(self, x, y):
        x = x.sigmoid()
        i, u = [t.flatten(1).sum(1) if self.image else t.sum() for t in [x*y, x+y]]
        dc = (2*i+1)/(u+1)
        dc = 1-dc.mean()
        return dc


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, first_out_channels=16):
        super().__init__()
        self.first = ConvBlock(in_channels, first_out_channels)
        in_channels = first_out_channels
        self.down1 = Down(in_channels, 2 * in_channels)
        self.down2 = Down(2 * in_channels, 4 * in_channels)
        self.down3 = Down(4 * in_channels, 8 * in_channels)
        self.up1   = Up(8 * in_channels, 4 * in_channels)
        self.up2   = Up(4 * in_channels, 2 * in_channels)
        self.up3   = Up(2 * in_channels, in_channels)
        self.final = nn.Conv3d(in_channels, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x  = self.up1(x4, x3)
        x  = self.up2(x, x2)
        x  = self.up3(x, x1)
        x  = self.final(x)
        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
'''
class ResBlock(nn.Module):
    def __init(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels),
                                      nn.BatchNorm1d(out_channels))
'''
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, y):
        x = self.conv2(x)
        x = self.conv1(torch.cat([y, x], dim=1))
        return x

'''
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=k_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        x = F.elu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = 16
        self.num_conv_blocks = 2
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for i in range(self.num_conv_blocks):
                if depth == 0:
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                else:
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        h = []
        for k, op in self.module_dict.items():
            x = op(x)
            if k.startswith("conv") and int(k[-1]) == self.num_conv_blocks - 1:
                h.append(x)
        return x, h


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
                                                   kernel_size=k_size, stride=stride,
                                                   padding=padding, output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose(x)


class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='trilinear'):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True)

    def forward(self, x):
        return self.upsample(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, model_depth=4, upmode='ConvTranspose'):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = 16
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth - 2, -1, -1):
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            if upmode == 'ConvTranspose':
                self.deconv = ConvTranspose(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
            elif upmode == 'Upsample':
                self.deconv = Upsample(2, 'trilinear')
            self.module_dict["deconv_{}".format(depth)] = self.deconv
            for i in range(self.num_conv_blocks):
                if i == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            if depth == 0:
                self.final_conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels)
                self.module_dict["final_conv"] = self.final_conv

    def forward(self, x, down_sampling_features):
        for k, op in self.module_dict.items():
            if k.startswith("deconv"):
                x = op(x)
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
            elif k.startswith("conv"):
                x = op(x)
            else:
                x = op(x)
        return x
'''

def dice(logits, targets, class_index):
    inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
    union = torch.sum(logits[:, class_index, :, :, :] + torch.sum(targets[:, class_index, :, :, :]))
    dice = (2. * inter + 1) / (union + 1)
    return dice


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.contiguous().view(num, -1)
        m2 = targets.contiguous().view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class UNet_3D(nn.Module):
    def __init__(self, encoder, decoder, in_channels=1, out_channels=1, upmode='ConvTranspose'):
        super(UNet_3D, self).__init__()
        self.in_planes = 1
        self.Encoder = encoder(in_channels)
        self.Decoder = decoder(out_channels, upmode=upmode)
        
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    def forward(self, x):
        out, h = self.Encoder(x)
        out = self.Decoder(out, h)
        return out


'''
dir_path = "/mnt/Datasets/ribfrac-data"


def load_data(dir_path):
    train_image_path = dir_path + "/train/images"
    train_label_path = dir_path + "/train/labels"
    val_image_path = dir_path + "/val/images"
    val_label_path = dir_path + "/val/labels"
    test_image_path = dir_path + "/test/images"

    train_imagename_list = os.listdir(train_image_path)
    train_labelname_list = os.listdir(train_label_path)
    val_imagename_list = os.listdir(val_image_path)
    val_labelname_list = os.listdir(val_label_path)
    test_imagename_list = os.listdir(test_image_path)

    train_imagename_list.sort()
    train_labelname_list.sort()
    val_imagename_list.sort()
    val_labelname_list.sort()
    test_imagename_list.sort()

    train_image_ls = []
    train_label_ls = []
    val_image_ls = []
    val_label_ls = []
    test_image_ls = []

    for i in range(len(train_imagename_list)):
        train_image_ls.append(nib.load(os.path.join(train_image_path, train_imagename_list[i])).dataobj)
        train_label_ls.append(nib.load(os.path.join(train_label_path, train_labelname_list[i])).dataobj)

    for i in range(len(val_imagename_list)):
        val_image_ls.append(nib.load(os.path.join(val_image_path, val_imagename_list[i])).dataobj)
        val_label_ls.append(nib.load(os.path.join(val_label_path, val_labelname_list[i])).dataobj)

    for i in range(len(test_imagename_list)):
        test_image_ls.append(nib.load(os.path.join(test_image_path, test_imagename_list[i])).dataobj)

    dataset = dict()
    dataset['train'] = {"image": train_image_ls, "label": train_label_ls}
    dataset['val'] = {"image": val_image_ls, "label": val_label_ls}
    dataset['test'] = {"image": test_image_ls}

    return dataset


def data_preprocess(data, device):
    data = np.asarray(data).transpose((2, 1, 0))
    dimensions = data.shape[0]
    min_value = np.min(data)
    max_value = np.max(data)
    data -= min_value
    data /= (max_value - min_value)
    if dimensions % 32 != 0:
        tmp = int(dimensions / 32) + 1
        dimensions_pad = tmp * 32 - dimensions
        pad_shape = (int(dimensions_pad / 2), dimensions_pad - int(dimensions_pad / 2))
        data = np.pad(data, (pad_shape, (0, 0), (0, 0)), 'constant', constant_values=(min_value, min_value))
    data = data.reshape(1, 1, -1, data.shape[1], data.shape[2])
    data = torch.from_numpy(data).type(torch.FloatTensor)
    data = data.to(device)
    data = data.reshape(-1, data.shape[1], data.shape[2])
    return data


def data_block_train_and_val(data):
    #ori_shape = data.shape
    data_block = split_data(data)
    data_block = data_block.reshape(-1,data_block.shape[3],data_block.shape[4],data_block.shape[5])
    return data_block
'''


def _remove_low_probs(pred, prob_thresh):
    pred = np.where(pred > prob_thresh, pred, 0)
    return pred


def _remove_spine_fp(pred, image, bone_thresh):
    image_bone = image > bone_thresh
    image_bone_2d = image_bone.sum(axis=-1)
    image_bone_2d = ndimage.median_filter(image_bone_2d, 10)
    image_spine = (image_bone_2d > image_bone_2d.max() // 3)
    kernel = disk(7)
    image_spine = ndimage.binary_opening(image_spine, kernel)
    image_spine = ndimage.binary_closing(image_spine, kernel)
    image_spine_label = label(image_spine)
    max_area = 0

    for region in regionprops(image_spine_label):
        if region.area > max_area:
            max_region = region
            max_area = max_region.area
    image_spine = np.zeros_like(image_spine)
    image_spine[
        max_region.bbox[0]:max_region.bbox[2],
        max_region.bbox[1]:max_region.bbox[3]
    ] = max_region.convex_image > 0

    return np.where(image_spine[..., np.newaxis], 0, pred)


def _remove_small_objects(pred, size_thresh):
    pred_bin = pred > 0
    pred_bin = remove_small_objects(pred_bin, size_thresh)
    pred = np.where(pred_bin, pred, 0)

    return pred


def _post_process(pred, image, prob_thresh, bone_thresh, size_thresh):

    # remove connected regions with low confidence
    pred = _remove_low_probs(pred, prob_thresh)

    # remove spine false positives
    pred = _remove_spine_fp(pred, image, bone_thresh)

    # remove small connected regions
    pred = _remove_small_objects(pred, size_thresh)

    return pred


def _predict_single_image(model, dataloader, postprocess, prob_thresh,
        bone_thresh, size_thresh):
    pred = np.zeros(dataloader.dataset.image.shape)#Output_shape
    crop_size = dataloader.dataset.crop_size
    #if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)
    model.eval()
    model.cuda()
    with torch.no_grad():
        for _,sample in enumerate(dataloader):
            images, centers = sample
            images = images.cuda()
            output = model(images).sigmoid().cpu().numpy().squeeze(axis=1)
            for i in range(len(centers)):
                center_x, center_y, center_z = centers[i]
                cur_pred_patch = pred[
                    center_x - crop_size // 2:center_x + crop_size // 2,
                    center_y - crop_size // 2:center_y + crop_size // 2,
                    center_z - crop_size // 2:center_z + crop_size // 2
                ]
                pred[
                    center_x - crop_size // 2:center_x + crop_size // 2,
                    center_y - crop_size // 2:center_y + crop_size // 2,
                    center_z - crop_size // 2:center_z + crop_size // 2
                ] = np.where(cur_pred_patch > 0, np.mean((output[i], cur_pred_patch), axis=0), output[i])
    if postprocess:
        pred = _post_process(pred, dataloader.dataset.image, prob_thresh,
            bone_thresh, size_thresh)
    return pred


def _make_submission_files(pred, affine):
    pred_label = label(pred > 0.5).astype(np.int16)
    pred_label[pred_label>0.5]=1
    pred_image = nib.Nifti1Image(pred_label, affine)
    return pred_image
