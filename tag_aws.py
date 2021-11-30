import os
import os.path
import pickle
import argparse
import requests
import random
import utils
import cv2

import logging
import boto3

def rename_attribute(obj, old_name, new_name):
    obj.__dict__[new_name] = obj.__dict__.pop(old_name)


def detect_labels_local_file(photo):
    client = boto3.client('rekognition')

    with open(photo, 'rb') as image:
        response = client.detect_labels(Image={'Bytes': image.read()})

    print('Detected labels in ' + photo)
    for label in response['Labels']:
        print(label['Name'] + ' : ' + str(label['Confidence']))

    return [l['Name'] for l in response['Labels']]

def tag_images(args):

    # images = utils.get_images_in_dir_rec(args.input)
    tmp_faces, img_labels = utils.load_img_labels(args.imgs_root)
    faces = utils.FACES(tmp_faces)
    images = list(faces.dict_by_files.keys())
    random.shuffle(images)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    rekognition_client = boto3.client('rekognition')

    for img in images:

        pkl_path = img + '.pkl'
        if os.path.isfile(pkl_path):
            with open(pkl_path, 'rb') as fid:
                img_label = pickle.load(fid)
        else:
            img_label = utils.IMG_LABELS(utils.get_timestamp(img))
        img_label.path = img

        # if not hasattr(img_label, 'gcloud_labels'):
        #     img_label.gcloud_labels = []
        # if not hasattr(img_label, 'gcloud_objects'):
        #     img_label.gcloud_objects = []
        # if not hasattr(img_label, 'gcloud_landmarks'):
        #     img_label.gcloud_landmarks = []
        # if not hasattr(img_label, 'gcloud_web'):
        #     img_label.gcloud_web = []

        # if len(img_label.gcloud_labels) != 0 or \
        #     len(img_label.gcloud_objects) != 0 or \
        #     len(img_label.gcloud_web) != 0 or \
        #     len(img_label.gcloud_landmarks):
        #     print('{} already tagged'.format(img))
        #     continue

        print('tagging {}'.format(img))

        labels = detect_labels_local_file(img)

        # img_label.gcloud_labels = detect_labels(image)
        # img_label.gcloud_objects = detect_objects(image)
        # img_label.gcloud_web = detect_web(image)
        # img_label.gcloud_landmarks = detect_landmarks(image)

        if 1:
            ocv_img = cv2.imread(img)
            cv2.imshow("image", ocv_img)
            cv2.waitKey(0)

        # with open(pkl_path, 'wb') as fid:
        #     pickle.dump(img_label, fid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help="Input image directory. Recursive processing is supported.")
    parser.add_argument('--imgs_root', type=str, required=True,
                        help="Root directory of your image library.")
    parser.add_argument('--recompute', help='Recompute detections.',
                        action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print('args.input needs to be a valid folder containing images')
        exit()

    print('Tagging images with AWS.')
    tag_images(args)
    print('Done.')

if __name__ == "__main__":
    main()