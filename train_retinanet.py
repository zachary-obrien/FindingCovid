import os
import pandas as pd
from PIL import Image

import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

# from retinanet import coco_eval
# from retinanet import csv_eval

df = pd.read_csv('/media/zac/12TB Drive/covid-detector/unpacked_data/train_image_level.csv')
df = df.dropna()

parsed_df = pd.DataFrame(columns = ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name'])

image_folder = '/media/zac/12TB Drive/covid-detector/extracted_images/jpgs/'
resized_image_folder = '/media/zac/12TB Drive/covid-detector/extracted_images/resized_jpgs/'
num_boxes = 0
for index, row in df.iterrows():
    image_name = image_folder + row['id'].replace("_image", "") + ".jpg"
    resized_image_name = resized_image_folder + row['id'].replace("_image", "") + ".jpg"

    print("Processing df row for:", row['id'])
    print('Is there an original?', os.path.isfile(image_name), ":", image_name)
    print('Is there already a resize?', (not os.path.isfile(resized_image_name)), ":", resized_image_name)

    if os.path.isfile(image_name) and not os.path.isfile(resized_image_name):
        img = Image.open(image_name)

        img_height = img.height
        img_width = img.width

        desired_height = 600
        desired_width = 600

        height_downscale = img_height / desired_height
        width_downscale = img_width / desired_width

        img = img.resize((desired_width, desired_height))

        base_name = os.path.basename(image_name)
        image_name = image_folder + base_name
        print("Resized and placed in:", resized_image_name)
        img.save(resized_image_name)

        if isinstance(row['boxes'], list) and len(row['boxes']) > 0:
            for box in row['boxes']:
                x_min = int(box['x'] / width_downscale)
                y_min = int(box['y'] / height_downscale)
                x_max = int((box['x'] + box['width']) / width_downscale)
                y_max = int((box['y'] + box['height'])  / height_downscale)
                class_name = 'opacity'

                num_boxes = num_boxes + 1
                print("Has boxes", num_boxes)
                if num_boxes > 3:
                    break

                parsed_df = parsed_df.append({'image_name':image_name, 'x_min':int(x_min),
                                  'y_min':int(y_min), 'x_max':int(x_max), 'y_max':int(y_max),
                                  'class_name':class_name},  ignore_index=True)
            exit()
        else:
            parsed_df = parsed_df.append({'image_name':image_name, 'x_min':"",
                            'y_min':"", 'x_max':"", 'y_max':"",
                            'class_name':""},  ignore_index=True)

parsed_df.to_csv('/media/zac/12TB Drive/covid-detector/meta/parsed_df.csv')
print(parsed_df.head())


exit()


from sklearn.model_selection import GroupShuffleSplit

train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, random_state = 42).split(parsed_df, groups=parsed_df['image_name']))

train_df = parsed_df.iloc[train_inds]
test_df = parsed_df.iloc[test_inds]


ANNOTATIONS_FILE = '/media/zac/12TB Drive/covid-detector/meta/annotations.csv'
CLASSES_FILE = '/media/zac/12TB Drive/covid-detector/meta/classes.csv'
VALIDATION_FILE = '/media/zac/12TB Drive/covid-detector/meta/validation.csv'

train_df.to_csv(ANNOTATIONS_FILE, index=False, header=None)
test_df.to_csv(VALIDATION_FILE, index=False, header=None)

classes = set(['opacity'])

with open(CLASSES_FILE, 'w') as f:
  for i, line in enumerate(sorted(classes)):
    f.write('{},{}\n'.format(line,i))


dataset_train = CSVDataset(train_file="/media/zac/12TB Drive/covid-detector/meta/annotations.csv", class_list="/media/zac/12TB Drive/covid-detector/meta/classes.csv",
                           transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
retinanet = retinanet.cuda()
retinanet = torch.nn.DataParallel(retinanet).cuda()
retinanet.training = True
optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
loss_hist = collections.deque(maxlen=500)
retinanet.train()
retinanet.module.freeze_bn()

for epoch_num in range(100):
    retinanet.train()
    retinanet.module.freeze_bn()

    epoch_loss = []

    for iter_num, data in enumerate(dataloader_train):
        try:
            optimizer.zero_grad()

            if torch.cuda.is_available():
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
            else:
                classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            del classification_loss
            del regression_loss
        except Exception as e:
            print(e)
            continue

    scheduler.step(np.mean(epoch_loss))

    torch.save(retinanet.module, './checkpoints/{}_retinanet_{}.pt'.format("csv", epoch_num))

retinanet.eval()

torch.save(retinanet, 'model_final.pt')
torch.save(retinanet.state_dict(), 'model_final.pth')