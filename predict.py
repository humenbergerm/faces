import os.path
import pickle
import argparse
import copy
import subprocess
import cv2

import utils


def predict_faces(args, knn_clf, svm_clf):
    tmp_faces, img_labels = utils.load_img_labels(args.imgs_root)
    faces = utils.FACES(tmp_faces)

    if not args.cls in faces.dict_by_name:
        print('no faces found in this class')
        return False

    files = faces.get_paths(faces.dict_by_name[args.cls])
    total_nr_files = len(files)

    i = 0
    key = 0
    while key != 27 and i < total_nr_files:

        str_count = str(i + 1) + ' / ' + str(total_nr_files)
        print(str_count)

        img_path = files[i]
        opencvImage = cv2.imread(img_path)
        height, width = opencvImage.shape[:2]
        scale = 600.0 / float(height)
        opencvImage = cv2.resize(opencvImage, (int(width * scale), int(height * scale)))
        opencvImage_clean = opencvImage.copy()

        face_idxs = faces.get_face_idxs_by_name_and_file(args.cls, img_path)
        for fi in face_idxs:
            opencvImage = opencvImage_clean.copy()
            face = faces.get_face(fi)
            name = face.name
            names, probs = utils.predict_face_svm(face.desc, svm_clf, print_top=True)
            if name == "unknown" or name == "detected" and name != 'deleted' and name != 'DELETED':
                if probs[0] >= 0.95:
                    name = names[0]
                    print('{} > 0.95'.format(name))
                else:
                    name_knn = utils.predict_knn(knn_clf, face.desc, n=7, thresh=0.3)
                    if name_knn != 'unknown':
                        name = name_knn
                        print('{} has majority in knn search'.format(name))
                    else:
                        name = 'unknown'
                        # faces.rename(fi, name)

            if name != args.cls and name != 'unknown' and name != 'deleted' and name != 'DELETED':
                if args.confirm:
                    utils.show_face_crop(img_path, face.loc)
                    utils.draw_faces_on_image(faces, faces.dict_by_files[img_path], scale, opencvImage, fi)
                    cv2.putText(opencvImage, name, (20, opencvImage.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 1)
                    cv2.imshow("faces", opencvImage)
                    cv2.setMouseCallback("faces", utils.click_face,
                                         (opencvImage_clean, faces, scale, img_labels[img_path], svm_clf))
                    key = cv2.waitKey(0)
                    if len(utils.clicked_idx) == 0:
                        utils.clicked_idx.append(fi)
                        utils.clicked_names = names
                    utils.perform_key_action(args, key, faces, utils.clicked_idx, utils.clicked_names, img_path, knn_clf)
                    utils.clicked_idx = []
                    utils.clicked_names = []

                    if not args.cls in faces.dict_by_name:
                        print('no faces found in this class')
                        return False

                    if key == 46 or key == 47:  # key '.' or key '/'
                        continue
                    elif key == 111:  # 'o'
                        faces.rename(fi, name)
                        faces.set_confirmed(fi, 1)
                else:
                    # move to new class
                    faces.rename(fi, name)
                    if name == 'unknown':
                        faces.set_confirmed(fi, 0)
                    else:
                        faces.set_confirmed(fi, 2)
        i += 1
        files = faces.get_paths(faces.dict_by_name[args.cls])

    faces.store_to_img_labels(args.imgs_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', type=str, required=True,
                        help="Class to predict, such as unknown or detected.")
    parser.add_argument('--knn', type=str, required=True,
                        help="Path to knn model file (e.g. knn.clf).")
    parser.add_argument('--svm', type=str, required=True,
                        help="Path to svm model file (e.g. svm.clf).")
    # parser.add_argument('--db', type=str, required=True,
    #                     help="Path to folder with predicted faces (.csv files).")
    parser.add_argument('--imgs_root', type=str, required=True,
                        help="Root directory of your image library.")
    parser.add_argument('--confirm', help='Each newly found face needs to be confirmed.',
                        action='store_true')
    args = parser.parse_args()

    # if not os.path.isdir(args.db):
    #   utils.mkdir_p(args.db)

    if os.path.isfile(args.knn):
        with open(args.knn, 'rb') as f:
            knn_clf = pickle.load(f)
    else:
        print('args.knn ({}) is not a valid file'.format(args.knn))
        exit()

    if os.path.isfile(args.svm):
        with open(args.svm, 'rb') as f:
            svm_clf = pickle.load(f)
    else:
        print('args.svm ({}) is not a valid file'.format(args.svm))
        exit()

    print('Predicting faces of class {}'.format(args.cls))
    predict_faces(args, knn_clf, svm_clf)

    print('Done.')


if __name__ == "__main__":
    main()
