import os
import os.path
import pickle
import argparse
import dlib
import cv2

import utils

processed_faces = 0

def detect_faces(args):

    tmp_faces, img_labels = utils.load_img_labels(args.imgs_root)
    faces = utils.FACES(tmp_faces)

    # initialize counters
    total_faces = len(utils.get_images_in_dir_rec(args.input))

    detect_faces_in_folder(args, faces, img_labels, utils.get_images_in_dir_rec(args.input), total_faces)

def detect_faces_in_folder(args, faces, img_labels, files, total_faces):
    global processed_faces

    # opencv face detector
    modelFile = "models/opencv_face_detector_uint8.pb"
    configFile = "models/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    # dlib face detector
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

    # MTCNN
    # from mtcnn.mtcnn import MTCNN
    # detectorMTCNN = MTCNN()

    # Find all the faces and compute 128D face descriptors for each face.
    for n, f in enumerate(files):
        print("Processing file ({}/{}): {}".format(processed_faces, total_faces, f))
        processed_faces = processed_faces + 1

        img_path = os.path.splitext(f)[0] + os.path.splitext(f)[1].lower()
        if img_path in img_labels and not args.recompute:
            print('file already processed, skipping, ...')
            continue

        # img = cv2.imread(f)
        # height, width = img.shape[:2]
        # ws = 600.0 / float(height)
        # img = cv2.resize(img, (int(width * ws), int(height * ws)))

        # locations_mtcnn, descriptors_mtcnn, imagesize = utils.detect_faces_in_image_MTCNN(f, detectorMTCNN, facerec, sp, detector)
        # utils.show_detections_on_image(locations_mtcnn, img, ws, (255, 0, 0))

        locations_cv2, descriptors_cv2, imagesize = utils.detect_faces_in_image_cv2(f, net, facerec, sp, detector)
        print('cv2: {} detection(s) found'.format(len(locations_cv2)))
        # utils.show_detections_on_image(locations_cv2, img, ws, (0, 255, 0))

        locations, descriptors, imagesize = utils.detect_faces_in_image(f, detector, facerec, sp)
        print('dlib: {} detection(s) found'.format(len(locations)))
        # utils.show_detections_on_image(locations, img, ws, (0, 0, 255))

        # merge detections dlib and cv2
        locs, descs = utils.merge_detections(locations, descriptors, locations_cv2, descriptors_cv2)
        print('total: {} detection(s) found'.format(len(locs)))

        # save dets to class "detected"
        timeStamp = utils.get_timestamp(img_path)
        cls = 'detected'
        for l,d in zip(locs, descs):
            face = utils.FACE(l, d, cls, timeStamp, 0)
            face.path = img_path
            faces.add(face)

        faces.store_file_to_img_labels(img_path, timeStamp)

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

    print('Detecting faces in {}'.format(args.input))
    detect_faces(args)
    print('Done.')

if __name__ == "__main__":
    main()