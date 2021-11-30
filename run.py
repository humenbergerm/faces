import os
import os.path
import pickle
import argparse
import dlib
import cv2
from shapely.geometry import Polygon
import numpy as np

import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_root', type=str, required=True,
                        help="Path to images folder.")
    args = parser.parse_args()

    # dlib face detector
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

    # win = dlib.image_window()

    tmp_faces, img_labels = utils.load_img_labels(args.imgs_root)
    faces = utils.FACES(tmp_faces)

    for im in faces.dict_by_files:
        for fi in faces.dict_by_files[im]:
            face = faces.get_face(fi)

            # if hasattr(face, 'shape_dlib68') and hasattr(face, 'desc_dlib68'):
            #     continue

            img = dlib.load_rgb_image(face.path)

            # locations: list of tuples (t,r,b,l)
            # dlib rect: left: int, top: int, right: int, bottom: int
            d = dlib.chip_details(dlib.rectangle(face.loc[3], face.loc[0], face.loc[1], face.loc[2]))
            roi = dlib.extract_image_chip(np.array(img), d)

            # idx tells which sub-detector was used, see https://github.com/davisking/dlib/blob/master/dlib/image_processing/frontal_face_detector.h
            dets, scores, idx = detector.run(roi, 1, 0.3)
            print(scores, idx)
            if len(dets) == 0:
                face.shape_dlib68 = None
                face.desc_dlib68 = None
            else:
                shape = dlib.rectangle(dets[0].left()+face.loc[3], dets[0].top()+face.loc[0], dets[0].right()+face.loc[3], dets[0].bottom()+face.loc[0])
                face.shape_dlib68 = sp(img, shape)
                face.desc_dlib68 = np.array(facerec.compute_face_descriptor(img, face.shape_dlib68))
                # win.clear_overlay()
                # win.set_image(img)
                # win.add_overlay(face.shape_dlib68)
                # dlib.hit_enter_to_continue()

            if not face.path in faces.changed_files:
                faces.changed_files.append(face.path)
        utils.store_to_img_labels(faces, img_labels)


if __name__ == "__main__":
    main()