import torch
import os
import torchvision
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

from torch.utils import data
from argparse import ArgumentParser
import sys
import tqdm
import pandas as pd
import numpy as np
import json
import shutil


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def predict(model, dataloader, device):
    model.eval()

    test_predictions = []
    test_img_paths = []
    for inputs, labels, paths in tqdm.tqdm(dataloader, total=len(dataloader), desc="prediction...", position=0, leave=True):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            preds = model(inputs)
        test_predictions.extend(preds.argmax(dim=1))
        test_img_paths.extend(paths)

    return test_img_paths, test_predictions


def create_submission(test_img_paths, test_predictions, filename_to_submit):
    class_names = ['female', 'male']
    test_predictions_labels = list(map(lambda x: class_names[x], test_predictions))
    submission = dict(zip(test_img_paths, test_predictions_labels))
    with open(filename_to_submit, "w") as json_file:
        json.dump(submission , json_file)
    




def main():
    parser = ArgumentParser()

    parser.add_argument('-t', '--test_path', dest='test_path', type=str, default='./data/test/',
                        help='path to the test data')


    args = parser.parse_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"


    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)


    with open(f"model.pth", "rb") as fp:
       best_state_dict = torch.load(fp, map_location="cpu")
       model.load_state_dict(best_state_dict)

    input_size = 224
    test_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
    ])


    test_dir = 'test_images'
    shutil.copytree(args.test_path, os.path.join(test_dir, 'unknown'))
    test_dataset = ImageFolderWithPaths(test_dir, test_transforms)
    test_dataloader = data.DataLoader(test_dataset, batch_size=8, num_workers=0, pin_memory=True,
                                      shuffle=False, drop_last=False)

    with open(f"model.pth", "rb")  as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_img_paths, test_predictions = predict(model, test_dataloader, device)

    create_submission(test_img_paths, test_predictions, "submit.csv")


if __name__ == '__main__':
    sys.exit(main())
