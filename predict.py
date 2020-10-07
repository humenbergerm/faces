import os.path
import pickle
import argparse
import copy
import subprocess
import cv2

import utils


def predict_faces(args, knn_clf, svm_clf):
    tmp_faces, img_labels = utils.load_img_labels(args.imgs_root)
    faces = utils.FACES(tmp_faces, args.imgs_root)

    name_id = faces.get_name_id(args.cls)

    if not name_id in faces.dict_by_name:
        print('no faces found in this class')
        return False

    files = faces.get_paths(faces.dict_by_name[name_id])
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

        face_idxs = faces.get_face_idxs_by_name_and_file(faces.get_name_id(args.cls), img_path)
        for fi in face_idxs:
            opencvImage = opencvImage_clean.copy()
            face = faces.get_face(fi)
            name = faces.get_real_name(face.name)
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

            # if name != args.cls and name != 'unknown' and name != 'deleted' and name != 'DELETED':
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
                    utils.perform_key_action(args, key, faces, utils.clicked_idx, utils.clicked_names, img_path)
                    utils.clicked_idx = []
                    utils.clicked_names = []

                    if not name_id in faces.dict_by_name:
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
        files = faces.get_paths(faces.dict_by_name[name_id])

    faces.store_to_img_labels(args.imgs_root)


def predict_class(args, knn_clf, svm_clf):
    cls = args.cls
    print('Detecting faces in class {} using knn.'.format(cls))
    preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)

    if preds_per_person.get(cls) == None:
        print('class {} not found'.format(cls))
        exit()

    if 0:
        face_locations = []
        face_encodings = []
        face_path = []
        for p in preds_per_person[cls]:
            face_locations.append(p[0][1])
            face_encodings.append(p[2])
            face_path.append(p[1])

        print('{} members of {}'.format(len(face_locations), cls))
        if len(face_locations) == 0:
            return

        distance_threshold = 0.3

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=3)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

        predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                       zip(knn_clf.predict(face_encodings), face_locations, are_matches)]

        # assumption: predictions has the same length like the class members
        for i, p in enumerate(preds_per_person[cls]):
            preds_per_person[cls][i] = (predictions[i][0], p[0][1]), p[1], p[2], p[3], p[4], p[5]

    ix = 0
    key = 0
    save = []
    save_idx = []
    nr_of_faces = len(preds_per_person[cls])
    save_each_change = True

    while key != 27 and nr_of_faces > 0:

        nr_of_faces = len(preds_per_person[cls])  # because it might have changed (e.g. if a face got deleted)
        if nr_of_faces == 0:
            print('no more faces of class {} found'.format(cls))
            break

        if ix >= nr_of_faces:
            ix = 0
        elif ix < 0:
            ix = nr_of_faces - 1

        name = preds_per_person[cls][ix][0][0]
        print('predicted: {}'.format(name))

        while len(save) > 10:
            save.pop(0)
            save_idx.pop(0)

        faces_files = utils.get_faces_in_files(preds_per_person)

        names, probs = utils.predict_face_svm(preds_per_person[cls][ix][2], svm_clf, print_top=True)
        if name == "unknown" or name == "detected":
            if probs[0] >= 0.95:
                name = names[0]
                print('{} > 0.95'.format(name))
            else:
                name_knn = utils.predict_knn(knn_clf, preds_per_person[cls][ix][2], n=7, thresh=0.3)
                if name_knn != 'unknown':
                    name = name_knn
                    print('{} has majority in knn search'.format(name))
                else:
                    utils.insert_element_preds_per_person(preds_per_person, cls, ix, 'unknown', conf=0)
                    preds_per_person[cls].pop(ix)

        if name != cls and name != 'unknown':
            if args.confirm:
                image_path = preds_per_person[cls][ix][1]
                tmp = preds_per_person[cls][ix]
                utils.show_face_crop(tmp[1], tmp[0][1])
                key, clicked_class, clicked_idx, clicked_names = utils.show_faces_on_image(svm_clf, names, cls, ix,
                                                                                           preds_per_person,
                                                                                           faces_files[image_path],
                                                                                           image_path,
                                                                                           waitkey=True, text=name)
                deleted_elem_of_cls = utils.evaluate_key(args, key, preds_per_person, clicked_class, clicked_idx, save,
                                                         clicked_names, faces_files, save_idx,
                                                         save_each_change=save_each_change)

                if deleted_elem_of_cls > 0 and clicked_idx <= ix and clicked_class == cls:
                    ix -= deleted_elem_of_cls

                if key == 46 or key == 47:  # key '.' or key '/'
                    ix += 1
                # elif key == 44:  # key ','
                #   ix -= 1
                elif key == 98:  # key 'b'
                    if len(save) > 0:
                        preds_per_person = copy.deepcopy(save.pop())
                        ix = save_idx.pop()
                        print("undone last action")
                elif key == 111:  # 'o'
                    # move to new class
                    save.append(copy.deepcopy(preds_per_person))
                    save_idx.append(ix)
                    utils.insert_element_preds_per_person(preds_per_person, cls, ix, name, conf=1)
                    preds_per_person[cls].pop(ix)
                    if save_each_change:
                        utils.export_face_to_csv(args.db, args.imgs_root, preds_per_person, cls)
                        utils.export_face_to_csv(args.db, args.imgs_root, preds_per_person, name)
            else:
                # move to new class
                utils.insert_element_preds_per_person(preds_per_person, cls, ix, name, conf=0)
                preds_per_person[cls].pop(ix)
        else:
            ix += 1

    utils.export_persons_to_csv(preds_per_person, args.imgs_root, args.db)


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
