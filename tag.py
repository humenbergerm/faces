import os
import os.path
import pickle
import argparse
import requests
import random

import utils

api_key = 'acc_11756022bba2b65'
api_secret = '723ff55f9a9081eec4642aab4044243f'


def rename_attribute(obj, old_name, new_name):
    obj.__dict__[new_name] = obj.__dict__.pop(old_name)

def tag_images(args):

    # images = utils.get_images_in_dir_rec(args.input)
    tmp_faces, img_labels = utils.load_img_labels(args.imgs_root)
    faces = utils.FACES(tmp_faces)
    images = list(faces.dict_by_files.keys())
    random.shuffle(images)
    counter = 0

    for img in images:

        if counter >= 2000:
            print('{} images processed'.format(counter))
            return -1

        pkl_path = img + '.pkl'
        if os.path.isfile(pkl_path):
            with open(pkl_path, 'rb') as fid:
                img_label = pickle.load(fid)
        else:
            img_label = utils.IMG_LABELS(utils.get_timestamp(img))
        img_label.path = img

        # if hasattr(img_label, 'imagga'):
        #     rename_attribute(img_label, 'imagga', 'tags')
        #     with open(pkl_path, 'wb') as fid:
        #         pickle.dump(img_label, fid)

        if not hasattr(img_label, 'categories'):
            img_label.categories = []

        if not hasattr(img_label, 'tags'):
            img_label.tags = []

        if len(img_label.tags) != 0 and len(img_label.categories) != 0:
            print('{} already tagged'.format(img))
            continue

        print('tagging {}'.format(img))

        response = requests.post(
            'https://api.imagga.com/v2/uploads',
            auth=(api_key, api_secret),
            files={'image': open(img, 'rb')})
        if response.ok:
            image_upload_id = response.json()['result']['upload_id']
        elif response.reason == 'Forbidden':
            print(response.text)
            return -1
        else:
            print(response.text)
            continue

        if len(img_label.tags) == 0:
            response = requests.get(
                'https://api.imagga.com/v2/tags?image_upload_id=%s' % image_upload_id,
                auth=(api_key, api_secret))
            if response.ok:
                result_list = response.json()['result']['tags']
                img_label.tags = [(t['tag']['en'], t['confidence']) for t in result_list if t['confidence'] > 0]
                counter += 1
                print('ok')
            elif response.reason == 'Forbidden':
                print(response.text)
                return -1
            else:
                print(response.text)
                continue

        # if len(img_label.categories) == 0:
        #     categorizer_id = 'personal_photos'
        #     response = requests.get(
        #         'https://api.imagga.com/v2/categories/%s?image_upload_id=%s' % (categorizer_id, image_upload_id),
        #         auth=(api_key, api_secret))
        #     if response.ok:
        #         result_list = response.json()['result']['categories']
        #         img_label.categories = [(t['name']['en'], t['confidence']) for t in result_list if t['confidence'] > 0]
        #         counter += 1
        #     else:
        #         print(response.text)
        #         return -1

        with open(pkl_path, 'wb') as fid:
            pickle.dump(img_label, fid)


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

    print('Tagging images with imagga.')
    tag_images(args)
    print('Done.')

if __name__ == "__main__":
    main()