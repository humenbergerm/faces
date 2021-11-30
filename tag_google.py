import os
import os.path
import pickle
import argparse
import requests
import random
import utils
import cv2

credential_path = "/Users/mhumenbe/tmp/faces-321809-e11c8ba671f7.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

from google.cloud import vision
import io
client = vision.ImageAnnotatorClient()

def rename_attribute(obj, old_name, new_name):
    obj.__dict__[new_name] = obj.__dict__.pop(old_name)

def detect_labels(image):
    response = client.label_detection(image=image)
    labels = response.label_annotations
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    print('Labels:')
    for label in labels:
        print(label.description)

    return [label.description for label in labels]

def detect_objects(image):
    objects = client.object_localization(
        image=image).localized_object_annotations
    # print('Number of objects found: {}'.format(len(objects)))
    print(list(dict.fromkeys([obj.name for obj in objects])))
    # for object_ in objects:
    #     print('\n{} (confidence: {})'.format(object_.name, object_.score))
    # print('Normalized bounding polygon vertices: ')
    # for vertex in object_.bounding_poly.normalized_vertices:
    #     print(' - ({}, {})'.format(vertex.x, vertex.y))

    return [obj.name for obj in objects]

def detect_landmarks(image):
    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations

    results = []

    print('Landmarks:')

    for landmark in landmarks:
        print(landmark.description)
        results.append(landmark.description)
        for location in landmark.locations:
            lat_lng = location.lat_lng
            print('Latitude {}'.format(lat_lng.latitude))
            print('Longitude {}'.format(lat_lng.longitude))
            results.append(f'lat_{lat_lng.latitude}_lng_{lat_lng.longitude}')

    # if response.error.message != '':
    #     raise Exception(
    #         '{}\nFor more info on error messages, check: '
    #         'https://cloud.google.com/apis/design/errors'.format(
    #             response.error.message))

    return results

def detect_web(image):
    response = client.web_detection(image=image)
    annotations = response.web_detection

    results = []

    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            print('\nBest guess label: {}'.format(label.label))
            results.append(label.label)

    # if annotations.pages_with_matching_images:
    #     print('\n{} Pages with matching images found:'.format(
    #         len(annotations.pages_with_matching_images)))
    #
    #     for page in annotations.pages_with_matching_images:
    #         print('\n\tPage url   : {}'.format(page.url))
    #
    #         if page.full_matching_images:
    #             print('\t{} Full Matches found: '.format(
    #                    len(page.full_matching_images)))
    #
    #             for image in page.full_matching_images:
    #                 print('\t\tImage url  : {}'.format(image.url))
    #
    #         if page.partial_matching_images:
    #             print('\t{} Partial Matches found: '.format(
    #                    len(page.partial_matching_images)))
    #
    #             for image in page.partial_matching_images:
    #                 print('\t\tImage url  : {}'.format(image.url))

    if annotations.web_entities:
        # print('\n{} Web entities found: '.format(
        #     len(annotations.web_entities)))

        # for entity in annotations.web_entities:
        #     print('\n\tScore      : {}'.format(entity.score))
        #     print(u'\tDescription: {}'.format(entity.description))

        for entity in annotations.web_entities:
            print(f'{entity.description} ')
            results.append(entity.description)

    # if annotations.visually_similar_images:
    #     print('\n{} visually similar images found:\n'.format(
    #         len(annotations.visually_similar_images)))
    #
    #     for image in annotations.visually_similar_images:
    #         print('\tImage url    : {}'.format(image.url))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return results

def tag_images(args):

    # images = utils.get_images_in_dir_rec(args.input)
    tmp_faces, img_labels = utils.load_img_labels(args.imgs_root)
    # faces = utils.FACES(tmp_faces)
    images = utils.get_images_in_dir_rec(args.imgs_root)
    # images = list(faces.dict_by_files.keys())
    random.shuffle(images)

    counter = 0
    count_tagged = 0
    untagged = []
    for img in images:
        counter += 1
        print(f'{counter}/{len(images)}')

        pkl_path = img + '.pkl'
        if os.path.isfile(pkl_path):
            with open(pkl_path, 'rb') as fid:
                img_label = pickle.load(fid)
        else:
            img_label = utils.IMG_LABELS(utils.get_timestamp(img))
        img_label.path = img

        if not hasattr(img_label, 'categories'):
            img_label.categories = []
        if not hasattr(img_label, 'tags'):
            img_label.tags = []
        if not hasattr(img_label, 'gcloud_labels'):
            img_label.gcloud_labels = []
        if not hasattr(img_label, 'gcloud_objects'):
            img_label.gcloud_objects = []
        if not hasattr(img_label, 'gcloud_landmarks'):
            img_label.gcloud_landmarks = []
        if not hasattr(img_label, 'gcloud_web'):
            img_label.gcloud_web = []

        if len(img_label.gcloud_labels) != 0 or \
            len(img_label.gcloud_objects) != 0 or \
            len(img_label.gcloud_web) != 0 or \
            len(img_label.gcloud_landmarks) != 0:
            print('{} already tagged'.format(img))
            count_tagged += 1
            continue
        untagged.append(img)

    counter = 0
    print(f'{count_tagged}/{len(images)}')
    for img in untagged:
        counter += 1
        print(f'tagging {img}')
        print(f'tagging {counter}/{len(untagged)}')

        pkl_path = img + '.pkl'
        if os.path.isfile(pkl_path):
            with open(pkl_path, 'rb') as fid:
                img_label = pickle.load(fid)
        else:
            img_label = utils.IMG_LABELS(utils.get_timestamp(img))
        img_label.path = img

        if not hasattr(img_label, 'categories'):
            img_label.categories = []
        if not hasattr(img_label, 'tags'):
            img_label.tags = []
        if not hasattr(img_label, 'gcloud_labels'):
            img_label.gcloud_labels = []
        if not hasattr(img_label, 'gcloud_objects'):
            img_label.gcloud_objects = []
        if not hasattr(img_label, 'gcloud_landmarks'):
            img_label.gcloud_landmarks = []
        if not hasattr(img_label, 'gcloud_web'):
            img_label.gcloud_web = []

        try:
            with io.open(img, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)

            img_label.gcloud_labels = detect_labels(image)
            # img_label.gcloud_objects = detect_objects(image)
            # img_label.gcloud_web = detect_web(image)
            img_label.gcloud_landmarks = detect_landmarks(image)
        except ValueError:
            print('error')

        if 0:
            ocv_img = cv2.imread(img)
            cv2.imshow("image", ocv_img)
            cv2.waitKey(1)

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

    print('Tagging images with Google Cloud.')
    tag_images(args)
    print('Done.')

if __name__ == "__main__":
    main()