import os
import os.path
import pickle
import csv
import numpy as np
import dlib
from collections import Counter, OrderedDict
import cv2
from shapely.geometry import Polygon, Point
import copy
import subprocess
from shapely.ops import cascaded_union
from datetime import datetime
from PIL import Image
import piexif


# from matplotlib import pyplot as plt
# import geopandas as gpd
# from descartes import PolygonPatch

def mkdir_p(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_images_in_dir_rec(path_to_dir):
    files = []
    for root, dirnames, filenames in os.walk(os.path.normpath(path_to_dir)):
        for j in filenames:
            if j.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.join(root, j))

    return files


def get_files_in_dir_by_name_rec(path_to_dir, filename):
    files = []
    for root, dirnames, filenames in os.walk(os.path.normpath(path_to_dir)):
        for j in filenames:
            if j == filename:
                files.append(os.path.join(root, j))

    return files


def get_files_in_dir_by_ext_rec(path_to_dir, ext):
    files = []
    for root, dirnames, filenames in os.walk(os.path.normpath(path_to_dir)):
        for j in filenames:
            if j.lower().endswith(ext):
                files.append(os.path.join(root, j))

    return files


def is_valid_timestamp(date):
    if len(date) == 19 \
        and int(date[:4]) > 1900 \
        and int(date[5:7]) in range(1, 13) \
        and int(date[8:10]) in range(1, 32) \
        and int(date[11:13]) in range(0, 24) \
        and int(date[14:16]) in range(0, 60) \
        and int(date[17:19]) in range(0, 60):
        return True
    else:
        return False


def get_files_in_dir(path_to_dir, ext):
    files = []

    entries = os.listdir(os.path.normpath(path_to_dir))
    for f in entries:
        if os.path.isfile(os.path.join(path_to_dir, f)):
            if f.lower().endswith(ext):
                files.append(os.path.join(path_to_dir, f))

    return sorted(files)


def export_persons_to_csv(preds_per_person, imgs_root, preds_per_person_path):
    if len(preds_per_person) == 0:
        print('nothing to export, preds_per_person is empty')
        return
    if not os.path.isdir(preds_per_person_path):
        os.makedirs(preds_per_person_path)
    for p in preds_per_person:
        if len(preds_per_person[p]) == 0:
            print('No faces found in {}. Files will be deleted.'.format(p))
            bin_file = os.path.join(preds_per_person_path, p + '.bin')
            if os.path.isfile(bin_file):
                os.remove(bin_file)
            csv_file = os.path.join(preds_per_person_path, p + '.csv')
            if os.path.isfile(csv_file):
                os.remove(csv_file)
        else:
            export_face_to_csv(preds_per_person_path, imgs_root, preds_per_person, p)


def count_unique_faces(faces):
    counter = 0
    paths = []
    for i in faces:
        if not i[1] in paths:
            counter += 1
            paths.append(i[1])

    return counter


def export_face_to_csv(preds_per_person_path, imgs_root, preds_per_person, face):
    print('exporting {} ({})'.format(face, count_unique_faces(preds_per_person[face])))

    tmp = copy.deepcopy(preds_per_person[face])
    for i, f in enumerate(tmp):
        relpath = os.path.relpath(f[1], imgs_root)
        tmp[i] = (f[0], relpath, f[2], f[3], f[4], f[5])

    # save everything in one pickle file
    pkl_path = os.path.join(preds_per_person_path, face + '.bin')
    with open(pkl_path, "wb") as fp:
        pickle.dump(tmp, fp)

    # open/create csv file
    csv_path = os.path.join(preds_per_person_path, face + '.csv')
    with open(csv_path, "w") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=';')
        header = []
        header.append('name')
        header.append('location in image')
        header.append('image path')
        header.append('flag')
        header.append('timestamp')
        header.append('imagesize')
        filewriter.writerow(header)
        for c in tmp:
            face_row = []
            for i in range(len(c)):
                if i == 0:
                    face_row.append(c[i][0])
                    # face_row.append(c[i][1])
                    face_row.append(
                        str(c[i][1][0]) + ' ' + str(c[i][1][1]) + ' ' + str(c[i][1][2]) + ' ' + str(c[i][1][3]))
                    continue
                if i == 1:
                    face_row.append(os.path.basename(c[i]))
                    continue
                if i == 2:
                    # skip the descriptor because it is saved in a separate file
                    continue
                face_row.append(c[i])
            filewriter.writerow(face_row)


def load_faces_from_csv(preds_per_person_path, imgs_root=''):
    print('Loading the faces from {}.'.format(preds_per_person_path))

    if preds_per_person_path == None:
        print('path to db is needed')
        exit()
    if not os.path.isdir(preds_per_person_path):
        print('path to db is not a valid directory')
        exit()

    preds_per_person = {}
    total = 0

    csv_files = get_files_in_dir(preds_per_person_path, '.csv')
    bin_files = get_files_in_dir(preds_per_person_path, '.bin')
    if len(csv_files) == 0:
        print('no csv files found in {}'.format(preds_per_person_path))
        return preds_per_person
    for f in csv_files:
        name = os.path.splitext(os.path.basename(f))[0]
        descs = pickle.load(open(os.path.join(preds_per_person_path, name + '.bin'), "rb"))
        if 1:
            for i in bin_files:
                filename = os.path.splitext(os.path.basename(i))[0]
                # check if multiple .bin files should be combined
                # bin files can be combined if the 'other' files are exactly one character longer, e.g.: martin.bin and martin1.bin
                if name in filename and len(name) + 1 == len(filename):
                    accept = input('Add ' + i + ' to ' + name + ' (y/n)?')
                    if accept == 'y':
                        tmp = pickle.load(open(i, "rb"))
                        descs += tmp
        with open(f, 'r') as csvfile:
            filereader = csv.reader(csvfile, delimiter=';')
            preds_per_person[name] = []
            for i, row in enumerate(filereader):
                # skip first row (header)
                if i == 0:
                    continue
                if len(row) == 1:
                    row_str = row[0].split(';')
                    face_loc = row_str[0], tuple(map(int, row_str[1].split(' ')))
                    img_name = os.path.basename(row_str[2])
                elif len(row) == 6:
                    face_loc = row[0], tuple(map(int, row[1].split(' ')))
                    img_name = os.path.basename(row[2])
                else:
                    print('invalid csv row format')
                    continue
                if 0:  # use this if the order between the csv and the bin file(s) needs to be restored; e.g. after manual changes in the files
                    found_face = find_face(descs, face_loc, img_name)
                else:
                    found_face = descs[i - 1]
                if len(found_face) != 0:
                    imgname = os.path.basename(found_face[1])
                    new_name = os.path.splitext(imgname)[0] + os.path.splitext(imgname)[1].upper()
                    # new_f = os.path.join(os.path.dirname(f[1]), new_name)
                    new_path = os.path.join(os.path.dirname(found_face[1]), new_name)
                    img_path = os.path.join(imgs_root, new_path)
                    if os.path.isfile(img_path):
                        preds_per_person[name].append(
                            ((name, face_loc[1]), img_path, found_face[2], found_face[3], found_face[4], found_face[5]))
                    else:
                        print('file {} does not exist'.format(found_face[1]))
                else:
                    print('{} not found'.format(img_name))

            total += len(preds_per_person[name])
            print('loaded {} ({})'.format(name, len(preds_per_person[name])))

    print('unknown/known = {}/{}'.format(len(preds_per_person['unknown']),
                                         total - len(preds_per_person['unknown']) - len(preds_per_person['deleted'])))
    return preds_per_person


def get_nr_after_filter(mask, preds_class):
    counter = 0
    for i in preds_class:
        if mask[i[1]][i[0][1]] == 1:
            counter += 1

    return counter


def filter_faces(args, preds_per_person):
    mask = {}
    for p in preds_per_person:
        for n, i in enumerate(preds_per_person[p]):
            loc = i[0][1]
            imgsize = i[5]
            filename = i[1]

            if mask.get(filename) == None:
                mask[filename] = {}

            if filter_face_size(loc, imgsize, float(args.min_size)):
                mask[filename][loc] = 1
            else:
                mask[filename][loc] = 0

    return mask


def get_folders_in_path(path):
    entries = os.listdir(os.path.normpath(path))
    dirs = []
    for f in entries:
        if os.path.isdir(os.path.join(path, f)):
            dirs.append(os.path.join(path, f))

    return dirs


def get_folders_in_dir_rec(path_to_dir):
    dirs = []
    for root, dirnames, filenames in os.walk(os.path.normpath(path_to_dir)):
        for j in dirnames:
            if os.path.isdir(os.path.join(root, j)):
                dirs.append(os.path.join(root, j))

    return dirs


def sort_folders_by_nr_of_images(folder_to_sort):
    dirs = get_folders_in_path(folder_to_sort)

    cnt = Counter()
    for d in dirs:
        num_of_images = len(get_images_in_dir(d))
        cnt[d] = num_of_images

    print("Number of images in folders: {}".format(sum(cnt.values())))

    cnt_ordered = OrderedDict(cnt.most_common())

    counter = 0
    for i in cnt_ordered:
        folder_name = os.path.join(folder_to_sort, str(counter) + "_nr_of_images_" + str(cnt_ordered[i]))
        os.rename(i, folder_name)
        # if (counter == 0):
        #    copytree(folder_name, folder_to_sort)
        counter += 1


def get_images_in_dir(path_to_dir):
    files = []

    entries = os.listdir(os.path.normpath(path_to_dir))
    for f in entries:
        if os.path.isfile(os.path.join(path_to_dir, f)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.join(path_to_dir, f))

    return files


def resizeCV(img, h):
    height, width = img.shape[:2]

    s = h / height
    height *= s
    width *= s

    return cv2.resize(img, (int(width), int(height)))


def filter_face_size(loc, imgsize, thresh=0):
    # loc = (top, right, bottom, left)
    roi_w = loc[1] - loc[3]
    roi_h = loc[2] - loc[0]
    img_h, img_w = imgsize[:-1]

    if roi_w >= thresh * img_w and roi_h >= thresh * img_h:
        return True
    else:
        return False


def is_valid_roi(x, y, w, h, img_shape):
    if x >= 0 and x < img_shape[1] and \
        y >= 0 and y < img_shape[0] and \
        w > 0 and h > 0 and \
        x + w < img_shape[1] and \
        y + h < img_shape[0]:
        return True
    return False


def detect_faces_in_image(img_path, detector, facerec, sp, use_entire_image=False, dets=[]):
    img = dlib.load_rgb_image(img_path)

    if dets == []:
        if use_entire_image:
            dets = [dlib.rectangle(0, 0, img.shape[1] - 1, img.shape[0] - 1)]
        else:
            dets = detector(img, 1)

    # print("Number of faces detected: {}".format(len(dets)))

    locations = []
    descriptors = []

    # process each face we found
    for k, d in enumerate(dets):
        # get the landmarks/parts for the face in box d.
        shape = sp(img, d)

        # face alignment is done automatically by dlib as pre-processing step in compute_face_descriptor
        # aligned_face = dlib.get_face_chip(img, shape)
        # cv2.imshow('aligned', aligned_face)
        # cv2.waitKey(0)

        # locations: list of tuples (t,r,b,l)
        # descriptors: list of float64 np arrays
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        descriptors.append(np.array(face_descriptor))
        locations.append((d.top(), d.right(), d.bottom(), d.left()))

    return locations, descriptors, img.shape


def detect_faces_in_image_cv2(img_path, net, facerec, sp, detector):
    opencvImage = cv2.imread(img_path)
    h, w = opencvImage.shape[:2]
    dim = (300, 300)
    blob = cv2.dnn.blobFromImage(cv2.resize(opencvImage, dim), 1.0, dim, (103.93, 116.77, 123.68))

    net.setInput(blob)
    detections = net.forward()

    dets = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            d = dlib.rectangle(x1, y1, x2, y2)
            dets.append(d)

    locations, descriptors, imagesize = detect_faces_in_image(img_path, detector, facerec, sp, dets=dets)

    return locations, descriptors, imagesize


def initialize_face_data(preds_per_person, cls):
    face_locations = []
    face_encodings = []
    for p in preds_per_person[cls]:
        face_locations.append(p[0][1])
        face_encodings.append(p[2])

    return face_locations, face_encodings


def delete_element_preds_per_person(preds_per_person, cls, ix):
    new_name = 'deleted'
    # add pred in new list
    insert_element_preds_per_person(preds_per_person, cls, ix, new_name)
    # delete pred in current list
    preds_per_person[cls].pop(ix)


def move_class(preds_per_person, cls, new_cls):
    accept = input('All members from class {} will be moved to class {}. (y/n)?'.format(cls, new_cls))
    if accept == 'y':
        for i, p in enumerate(preds_per_person[cls]):
            insert_element_preds_per_person(preds_per_person, cls, i, new_cls, 1)

        preds_per_person[cls] = []


def insert_element_preds_per_person(preds_per_person, cls, ix, new_cls, conf=-1, new_face=None):
    if new_face == None:
        tmp = preds_per_person[cls][ix]
    else:
        tmp = new_face
    if conf == -1:
        conf = tmp[3]
    if preds_per_person.get(new_cls) == None:
        preds_per_person[new_cls] = []
    preds_per_person[new_cls].append(((new_cls, tmp[0][1]), tmp[1], tmp[2], conf, tmp[4], tmp[5]))


def add_new_face(preds_per_person, faces_files, cls, loc, desc, f, timeStamp, imagesize):
    if faces_files.get(
        f) != None:  # if there is no face detected yet, there won't be an entry in face_files -> directly add new face
        for p in faces_files[f]:
            c, i = p
            if np.linalg.norm(desc - preds_per_person[c][i][2]) == 0:
                print('face found in class {}'.format(c))  # if the face was found, skip it
                return
    if preds_per_person.get(cls) == None:
        preds_per_person[cls] = []
    preds_per_person[cls].append([(cls, loc), f, desc, 0, timeStamp, imagesize])
    print('added new face to {}'.format(cls))


def count_preds_status(preds_per_person):
    count_ignored = 0
    count_confirmed = 0
    for i in preds_per_person:
        if i[3] == 1:
            count_confirmed += 1
        if i[3] == 2:
            count_ignored += 1
    count_not_ignored = len(preds_per_person) - count_confirmed - count_ignored

    return count_confirmed, count_ignored, count_not_ignored


def resizeCV(img, w):
    height, width = img.shape[:2]

    s = w / width
    height *= s
    width *= s

    return cv2.resize(img, (int(width), int(height)))


def perform_key_action(args, key, faces, face_indices, names):
    if key == 99:  # key 'c'
        if len(face_indices) != 1:
            print('Too many or zero faces selected.')
            return False
        new_name = guided_input(faces)
        if new_name != "":
            faces.rename(face_indices[0], new_name)
            print("face changed: {} ({})".format(new_name, len(faces.dict_by_name[faces.get_name_id(new_name)])))
    # elif key == 109:  # key 'm'
    #     # new_name = guided_input(preds_per_person)
    #     # if new_name != "":
    #     #   save.append(copy.deepcopy(preds_per_person))
    #     #   if save_idx != None:
    #     #     save_idx.append(ix)
    #     #   move_class(preds_per_person, cls, new_name)
    #     #   if save_each_change:
    #     #     export_face_to_csv(args.db, args.imgs_root, preds_per_person, cls)
    #     #     export_face_to_csv(args.db, args.imgs_root, preds_per_person, new_name)
    #     print("class moved: {} -> {}".format(cls, new_name))
    elif key == 117:  # key 'u'
        if len(face_indices) != 1:
            print('Too many or zero faces selected.')
            return False
        new_name = 'unknown'
        faces.rename(face_indices[0], new_name)
        print("face confirmed: {} ({})".format(new_name, len(faces.dict_by_name[faces.get_name_id(new_name)])))
    elif key == 47:  # key '/'
        if len(face_indices) != 1:
            print('Too many or zero faces selected.')
            return False
        faces.flip_confirmed(face_indices[0])
        name = faces.get_real_face_name(face_indices[0])
        print("face confirmed: {} ({})".format(name, len(faces.dict_by_name[faces.get_name_id(name)])))
    elif key >= 48 and key <= 57:  # keys '0' - '9'
        if len(face_indices) != 1:
            print('Too many or zero faces selected.')
            return False
        new_name = names[key - 48]
        faces.rename(face_indices[0], new_name)
        print("face confirmed: {} ({})".format(new_name, len(faces.dict_by_name[faces.get_name_id(new_name)])))
    elif key == 100:  # key 'd'
        for fi in face_indices:
            faces.rename(fi, 'deleted')
            print("face deleted")
    elif key == 120:  # key 'x'
        for fi in face_indices:
            faces.rename(fi, 'DELETED')
            print("face deleted forever")
    elif key == 116:  # key 't'
        print('t pressed')
        # subprocess.call(["open", "-R", preds_per_person[cls][ix][1]])
    elif key == 97:  # key 'a'
        # To be checked!!!
        # save.append(copy.deepcopy(preds_per_person))
        # if save_idx != None:
        #   save_idx.append(ix)
        # for f in faces_files[preds_per_person[cls][ix][1]]:
        #   del_cls, del_i = f
        #   delete_element_preds_per_person(preds_per_person, del_cls, del_i)
        #   if del_cls == cls and del_i <= ix:
        #     deleted_elem_of_cls += 1
        print('all faces in deleted.')
    elif key == 115:  # key 's'
        # export_persons_to_csv(preds_per_person, args.imgs_root, args.db)
        print('saved')
    elif key == 122:  # key 'z'
        print('z pessed')
        # tmp = preds_per_person[cls][ix]
        # show_face_crop(tmp[1], tmp[0][1])
    return True


def evaluate_key(args, key, preds_per_person, cls, ix, save, names, faces_files, save_idx=None, save_each_change=False):
    deleted_elem_of_cls = 0
    if key == 99:  # key 'c'
        new_name = guided_input(preds_per_person)
        if new_name != "":
            save.append(copy.deepcopy(preds_per_person))
            if save_idx != None:
                save_idx.append(ix)
            # add pred in new list
            if preds_per_person.get(new_name) == None:
                preds_per_person[new_name] = []
            insert_element_preds_per_person(preds_per_person, cls, ix, new_name, 1)
            # delete pred in current list
            preds_per_person[cls].pop(ix)
            deleted_elem_of_cls = 1
            if save_each_change:
                export_face_to_csv(args.db, args.imgs_root, preds_per_person, cls)
                export_face_to_csv(args.db, args.imgs_root, preds_per_person, new_name)
            print("face changed: {} ({})".format(new_name, len(preds_per_person[new_name])))
    elif key == 109:  # key 'm'
        new_name = guided_input(preds_per_person)
        if new_name != "":
            save.append(copy.deepcopy(preds_per_person))
            if save_idx != None:
                save_idx.append(ix)
            move_class(preds_per_person, cls, new_name)
            if save_each_change:
                export_face_to_csv(args.db, args.imgs_root, preds_per_person, cls)
                export_face_to_csv(args.db, args.imgs_root, preds_per_person, new_name)
            print("class moved: {} -> {}".format(cls, new_name))
    elif key == 117:  # key 'u'
        save.append(copy.deepcopy(preds_per_person))
        if save_idx != None:
            save_idx.append(ix)
        new_name = 'unknown'
        # add pred in new list
        if preds_per_person.get(new_name) == None:
            preds_per_person[new_name] = []
        insert_element_preds_per_person(preds_per_person, cls, ix, new_name)
        # delete pred in current list
        preds_per_person[cls].pop(ix)
        deleted_elem_of_cls = 1
        if save_each_change:
            export_face_to_csv(args.db, args.imgs_root, preds_per_person, cls)
            export_face_to_csv(args.db, args.imgs_root, preds_per_person, new_name)
        print("face changed: {} ({})".format(new_name, len(preds_per_person[new_name])))
    elif key == 47:  # key '/'
        save.append(copy.deepcopy(preds_per_person))
        if save_idx != None:
            save_idx.append(ix)
        tmp = preds_per_person[cls][ix]
        if tmp[3] == 0:
            preds_per_person[cls][ix] = tmp[0], tmp[1], tmp[2], 1, tmp[4], tmp[5]
        elif tmp[3] == 1:
            preds_per_person[cls][ix] = tmp[0], tmp[1], tmp[2], 0, tmp[4]
        print("face confirmed: {} ({})".format(tmp[0], len(preds_per_person[cls])))
    elif key >= 48 and key <= 57:  # keys '0' - '9'
        save.append(copy.deepcopy(preds_per_person))
        if save_idx != None:
            save_idx.append(ix)
        new_name = names[key - 48]
        insert_element_preds_per_person(preds_per_person, cls, ix, new_name, 1)
        # delete pred in current list
        preds_per_person[cls].pop(ix)
        deleted_elem_of_cls = 1
        if save_each_change:
            export_face_to_csv(args.db, args.imgs_root, preds_per_person, cls)
            export_face_to_csv(args.db, args.imgs_root, preds_per_person, new_name)
        print("face confirmed: {} ({})".format(new_name, len(preds_per_person[new_name])))
    elif key == 100:  # key 'd'
        save.append(copy.deepcopy(preds_per_person))
        if save_idx != None:
            save_idx.append(ix)
        # delete pred in current list
        delete_element_preds_per_person(preds_per_person, cls, ix)
        deleted_elem_of_cls = 1
        if save_each_change:
            export_face_to_csv(args.db, args.imgs_root, preds_per_person, cls)
            export_face_to_csv(args.db, args.imgs_root, preds_per_person, 'deleted')
        print("face deleted")
    elif key == 116:  # key 't'
        subprocess.call(["open", "-R", preds_per_person[cls][ix][1]])
    elif key == 97:  # key 'a'
        # To be checked!!!
        # save.append(copy.deepcopy(preds_per_person))
        # if save_idx != None:
        #   save_idx.append(ix)
        # for f in faces_files[preds_per_person[cls][ix][1]]:
        #   del_cls, del_i = f
        #   delete_element_preds_per_person(preds_per_person, del_cls, del_i)
        #   if del_cls == cls and del_i <= ix:
        #     deleted_elem_of_cls += 1
        print('all faces in deleted.')
    elif key == 115:  # key 's'
        export_persons_to_csv(preds_per_person, args.imgs_root, args.db)
        print('saved')
    elif key == 122:  # key 'z'
        tmp = preds_per_person[cls][ix]
        show_face_crop(tmp[1], tmp[0][1])
    return deleted_elem_of_cls


def get_rect_from_pts(pts, ws):
    top = min(pts[0][1], pts[1][1])
    left = min(pts[0][0], pts[1][0])
    bottom = max(pts[0][1], pts[1][1])
    right = max(pts[0][0], pts[1][0])
    return (int(top / ws), int(right / ws), int(bottom / ws), int(left / ws))


clicked_cls = ''
clicked_idx = []
clicked_names = []
refPt = []
sp = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")


def click_face(event, x, y, flags, params):
    global refPt
    global clicked_idx
    global clicked_names

    image = params[0]
    faces = params[1]
    scale = params[2]
    img_label = params[3]
    svm_clf = params[4]

    tmp_img = image.copy()

    if event == cv2.EVENT_LBUTTONUP and flags == (cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON):
        pt = Point(x, y)
        for i in faces.dict_by_files[img_label.path]:
            (top, right, bottom, left) = faces.get_loc(i)
            top = int(top * scale)
            right = int(right * scale)
            bottom = int(bottom * scale)
            left = int(left * scale)
            p = Polygon([(left, top), (right, top), (right, bottom), (left, bottom)])
            if p.contains(pt):
                if i in clicked_idx:
                    clicked_idx.pop(clicked_idx.index(i))
                else:
                    clicked_idx.append(i)
                draw_faces_on_image(faces, faces.dict_by_files[img_label.path], scale, tmp_img)
                draw_clicked_faces_on_image(faces, clicked_idx, scale, tmp_img)
                cv2.imshow("faces", tmp_img)
                cv2.waitKey(1)
                if len(clicked_idx) == 1:
                    print('clicked class: {}, clicked index: {}'.format(faces.get_real_face_name(clicked_idx[0]),
                                                                        clicked_idx[0]))
                    clicked_names, probs = predict_face_svm(faces.get_face(clicked_idx[0]).desc, svm_clf)
                    print_top_svm(clicked_names, probs)
                break
    elif event == cv2.EVENT_LBUTTONDOWN and flags == cv2.EVENT_FLAG_LBUTTON:
        refPt = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP and flags == cv2.EVENT_FLAG_LBUTTON:
        refPt.append((x, y))
        p = Polygon([refPt[0], (refPt[1][0], refPt[0][1]), refPt[1], (refPt[0][0], refPt[1][1])])
        if p.area >= 10:
            for i in faces.dict_by_files[img_label.path]:
                (top, right, bottom, left) = faces.get_loc(i)
                top = int(top * scale)
                right = int(right * scale)
                bottom = int(bottom * scale)
                left = int(left * scale)
                if p.contains(Point(left, top)):
                    cv2.rectangle(tmp_img, (left, top), (right, bottom), (255, 128, 0), 1)
                    if not i in clicked_idx:
                        clicked_idx.append(i)
            draw_faces_on_image(faces, faces.dict_by_files[img_label.path], scale, tmp_img)
            draw_clicked_faces_on_image(faces, clicked_idx, scale, tmp_img)
            cv2.imshow("faces", tmp_img)
            cv2.waitKey(1)

            if len(clicked_idx) == 0:
                new_name = faces.get_name_id('unknown')
                if new_name != "":
                    new_loc = get_rect_from_pts(refPt,
                                                scale)  # order in preds_per_person: (top(), right(), bottom(), left())
                    d = dlib.rectangle(new_loc[3], new_loc[0], new_loc[1], new_loc[2])
                    opencvImage = cv2.imread(img_label.path)
                    shape = sp(opencvImage, d)
                    face_descriptor = facerec.compute_face_descriptor(opencvImage, shape)
                    new_desc = np.array(face_descriptor)

                    face = FACE(new_loc, new_desc, new_name, img_label.timestamp, 0)
                    face.path = img_label.path
                    faces.add(face)
                    draw_faces_on_image(faces, faces.dict_by_files[face.path], scale, tmp_img)
                    cv2.imshow("faces", tmp_img)
                    cv2.waitKey(1)
                    print('New unknown face added.')
        refPt = []
    # else:
    #     draw_faces_on_image(faces, faces.dict_by_files[img_label.path], scale, tmp_img)
    #     cv2.imshow("faces", tmp_img)


def click(event, x, y, flags, params):
    global clicked_cls, clicked_idx, clicked_names

    global refPt

    image = params[0]
    preds_per_person = params[1]
    face_files = get_faces_in_files(preds_per_person)
    face_indices = face_files[preds_per_person[clicked_cls][clicked_idx][1]]
    ws = params[3]
    svm_clf = params[4]
    main_face = params[5]
    main_idx = params[6]
    draw_main_face = params[7]

    if flags == (cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON):
        draw_rects(face_indices, preds_per_person, main_face, main_idx, ws, image, draw_main_face=draw_main_face)
        pt = Point(x, y)
        for i in face_indices:
            cls, idx = i
            (top, right, bottom, left) = preds_per_person[cls][idx][0][1]
            top = int(top * ws)
            right = int(right * ws)
            bottom = int(bottom * ws)
            left = int(left * ws)
            p = Polygon([(left, top), (right, top), (right, bottom), (left, bottom)])
            if p.contains(pt):
                cv2.rectangle(image, (left, top), (right, bottom), (255, 128, 0), 1)
                clicked_cls = cls
                clicked_idx = idx
                print('clicked class: {}, clicked index: {}'.format(clicked_cls, clicked_idx))
                print(preds_per_person[clicked_cls][clicked_idx][1])
                clicked_names, probs = predict_face_svm(preds_per_person[clicked_cls][clicked_idx][2], svm_clf)
                break
            # else:
            # cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
            # draw_rects(face_indices, preds_per_person, main_face, main_idx, ws, image)
    elif event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 1)
        p = Polygon([refPt[0], (refPt[1][0], refPt[0][1]), refPt[1], (refPt[0][0], refPt[1][1])])
        if p.area >= 10:
            cv2.imshow("faces", image)
            cv2.waitKey(1)
            # new_name = guided_input(preds_per_person)
            new_name = 'unknown'
            if new_name != "":
                new_loc = get_rect_from_pts(refPt, ws)  # order in preds_per_person: (top(), right(), bottom(), left())
                d = dlib.rectangle(new_loc[3], new_loc[0], new_loc[1], new_loc[2])
                opencvImage = cv2.imread(preds_per_person[main_face][main_idx][1])
                shape = sp(opencvImage, d)
                face_descriptor = facerec.compute_face_descriptor(opencvImage, shape)
                new_desc = np.array(face_descriptor)
                if preds_per_person.get(new_name) == None:
                    preds_per_person[new_name] = []
                preds_per_person[new_name].append(((new_name, new_loc), preds_per_person[main_face][main_idx][1],
                                                   new_desc, 1, preds_per_person[main_face][main_idx][4],
                                                   opencvImage.shape))
                print('New face {} added.'.format(new_name))
        refPt = []

    cv2.imshow("faces", image)


def show_detections_on_image(locations, img_path, waitkey=True):
    opencvImage = cv2.imread(img_path)

    height, width = opencvImage.shape[:2]
    ws = 600.0 / float(height)
    opencvImage = cv2.resize(opencvImage, (int(width * ws), int(height * ws)))

    for l in locations:
        (top, right, bottom, left) = l
        top = int(top * ws)
        right = int(right * ws)
        bottom = int(bottom * ws)
        left = int(left * ws)
        cv2.rectangle(opencvImage, (left, top), (right, bottom), (0, 255, 0), 1)

    cv2.imshow("detections", opencvImage)
    if waitkey:
        return cv2.waitKey(0)
    else:
        return cv2.waitKey(1)


def draw_rect(image, loc, scale, color, name=''):
    (top, right, bottom, left) = loc
    top = int(top * scale)
    right = int(right * scale)
    bottom = int(bottom * scale)
    left = int(left * scale)
    cv2.rectangle(image, (left, top), (right, bottom), color, 1)
    if name != '':
        cv2.putText(image, name, (left + 5, top + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_rects(face_indices, preds_per_person, main_face, main_idx, ws, image, draw_main_face=True):
    for i in face_indices:
        cls, idx = i
        # print(cls)
        if cls == 'unknown':
            color = (0, 0, 255)  # red
        elif cls == 'deleted':
            color = (128, 128, 128)  # gray
        elif cls == 'detected':
            color = (0, 128, 255)
        else:
            color = (0, 255, 0)  # green
        draw_rect(image, preds_per_person[cls][idx][0][1], ws, color)

    if draw_main_face:
        draw_rect(image, preds_per_person[main_face][main_idx][0][1], ws, (255, 0, 0))  # blue


def draw_faces_on_image(faces, face_indices, scale, image):
    for i in face_indices:
        cls = faces.get_real_face_name(i)
        if cls == 'unknown':
            color = (0, 0, 255)  # red
        elif cls.lower() == 'deleted':
            color = (128, 128, 128)  # gray
        elif cls == 'detected':
            color = (0, 128, 255)
        else:
            color = (0, 255, 0)  # green
        draw_rect(image, faces.get_loc(i), scale, color, cls)


def draw_clicked_faces_on_image(faces, face_indices, scale, image):
    for i in face_indices:
        draw_rect(image, faces.get_loc(i), scale, (255, 128, 0))


def show_faces_on_image(svm_clf, names, main_face, main_idx, preds_per_person, face_indices, img_path, waitkey=True,
                        text='', draw_main_face=True):
    # initilize the "clicked face" with the current face (main face)
    global clicked_cls, clicked_idx, clicked_names
    clicked_cls = main_face
    clicked_idx = main_idx
    clicked_names = names

    opencvImage = cv2.imread(img_path)

    height, width = opencvImage.shape[:2]
    ws = 600.0 / float(height)
    opencvImage = cv2.resize(opencvImage, (int(width * ws), int(height * ws)))

    if draw_main_face:
        confirmed = preds_per_person[main_face][main_idx][3]
        if confirmed >= 1:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.putText(opencvImage, main_face, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        if text != None:
            cv2.putText(opencvImage, text, (20, opencvImage.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    draw_rects(face_indices, preds_per_person, main_face, main_idx, ws, opencvImage, draw_main_face=draw_main_face)

    cv2.namedWindow("faces")
    cv2.setMouseCallback("faces", click, (
    opencvImage, preds_per_person, face_indices, ws, svm_clf, main_face, main_idx, draw_main_face))

    cv2.imshow("faces", opencvImage)
    if waitkey:
        return cv2.waitKey(0), clicked_cls, clicked_idx, clicked_names
    else:
        return cv2.waitKey(1), clicked_cls, clicked_idx, clicked_names


# if index is -1: use all elements of predictions, if not, only use one (given by the index)
def show_prediction_labels_on_image(predictions, pil_image, confirmed=None, index=-1, img_path=None, text=None,
                                    force_name=''):
    if img_path != None:
        opencvImage = cv2.imread(img_path)
    else:
        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    height, width = opencvImage.shape[:2]
    ws = 600.0 / float(height)
    opencvImage = cv2.resize(opencvImage, (int(width * ws), int(height * ws)))

    if index != -1:
        name, (top, right, bottom, left) = predictions[index]
        if force_name != '':
            name = force_name
        top = int(top * ws)
        right = int(right * ws)
        bottom = int(bottom * ws)
        left = int(left * ws)
        cv2.rectangle(opencvImage, (left, top), (right, bottom), (0, 255, 0), 1)
        if confirmed >= 1:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.putText(opencvImage, name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        if text != None:
            cv2.putText(opencvImage, text, (20, opencvImage.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    else:
        for name, (top, right, bottom, left) in predictions:
            top = int(top * ws)
            right = int(right * ws)
            bottom = int(bottom * ws)
            left = int(left * ws)
            cv2.rectangle(opencvImage, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(opencvImage, name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            if text != None:
                cv2.putText(opencvImage, text, (20, opencvImage.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 1)

    cv2.imshow("detections", opencvImage)
    return cv2.waitKey(0)


def guided_input(faces):
    options = list(faces.name2name_id.keys())

    user_input = input("Enter new name: ")

    filtered_names = []

    for i in options:
        if user_input in i:
            filtered_names.append(i)

    if len(filtered_names) > 1:
        # Deal with more that one team.
        filtered_names.append('abort')
        print('There is more than one person starting with "{}"'.format(user_input))
        print('Select the correct person from these choices: ')
        for index, name in enumerate(filtered_names):
            print("{}: {}".format(index, name))

        index = input("Enter choice number: ")
        if not index.isdigit():
            print('aborted')
            return ""
        else:
            index = int(index)

        if index == len(filtered_names) - 1:
            new_name = ""
            print('aborted')
        else:
            new_name = filtered_names[index]
            print('Selected person: {}'.format(new_name))
    elif len(filtered_names) == 1:
        # Only one person found
        new_name = filtered_names[0]
    elif len(filtered_names) == 0:
        print('Generate a new person {}?'.format(user_input))
        print('1 ... yes, 0 ... no')
        index = input("Enter choice number: ")
        if not index.isdigit():
            print('aborted')
            index = 0
        else:
            index = int(index)
        if index == 1:
            new_name = user_input
        else:
            new_name = ""

    return new_name


def load_detections(path):
    dets = {}
    files = get_files_in_dir_by_name_rec(path, 'detections.bin')
    for f in files:
        dirname = os.path.dirname(f)
        dets[dirname] = pickle.load(open(f, "rb"))

    return dets


def load_detections_as_single_dict(path):
    dets = {}
    det_file_map = {}
    files = get_files_in_dir_by_name_rec(path, 'detections.bin')
    for f in files:
        print('loading {}'.format(os.path.dirname(f)))
        det_file_map[f] = []
        # dirname = os.path.dirname(f)
        tmp = pickle.load(open(f, "rb"))
        for t in tmp:
            det_file_map[f].append(t)
            dets[t] = tmp[t]

    return dets, det_file_map


def save_detections(dets, det_file_map):
    # dets_per_folder = {}
    for dm in det_file_map:
        dets_to_save = {}
        for d in det_file_map[dm]:
            if dets.get(d) != None:
                dets_to_save[d] = dets[d]

        with open(dm, "wb") as fp:
            pickle.dump(dets_to_save, fp)

        # dirname = os.path.dirname(d)
        # if len(dets_per_folder[dirname]) == 0:
        #   dets_per_folder[dirname][d] = dets[d]
        # dets_per_folder[dirname]

        # outfile = os.path.join(d, 'detections.bin')
        # with open(outfile, "wb") as fp:
        #   pickle.dump(dets[d], fp)


def delete_detections_of_file(dets, filepath):
    if dets.get(filepath) != None:
        dets.pop(filepath)
        print('detections in {} deleted'.format(filepath))
    else:
        print('{} not found in detections'.format(filepath))


def ignore_detections_of_file(dets, filepath):
    if dets.get(filepath) != None:
        dets[filepath] = ([], [])
        print('detections in {} will be ignored'.format(filepath))
    else:
        print('{} not found in detections'.format(filepath))


def predict_face_svm(enc, svm, print_top=True):
    preds = svm.predict_proba(enc.reshape(1, -1))[0]
    sorted = np.argsort(-preds)

    names = []
    probs = []
    for s in range(10):
        ix = sorted[s]
        names.append(svm.classes_[ix])
        probs.append(preds[ix])

    if print_top:
        print_top_svm(names, probs)

    return names, probs


def print_top_svm(names, probs):
    print('\n')
    for i, (n, p) in enumerate(zip(names, probs)):
        print('{}: name: {}, prob: {}'.format(i, n, p))
    print('\n')


def save_face_crop(face_path, img_path, loc):
    margin = 30
    img = cv2.imread(img_path)

    x = loc[3] - margin
    y = loc[0] - margin
    w = loc[1] + margin - x
    h = loc[2] + margin - y
    if is_valid_roi(x, y, w, h, img.shape):
        roi = img[y:y + h, x:x + w]
        cv2.imwrite(face_path, roi)
        return True

    return False


def show_face_crop(img_path, loc):
    margin = 50
    img = cv2.imread(img_path)
    x = loc[3] - margin
    y = loc[0] - margin
    w = loc[1] + margin - x
    h = loc[2] + margin - y
    if not is_valid_roi(x, y, w, h, img.shape):
        x = loc[3]
        y = loc[0]
        w = loc[1] - x
        h = loc[2] - y
    roi = img[y:y + h, x:x + w]
    if roi.shape[1] > 300:
        roi = resizeCV(roi, 300)
    if is_valid_roi(x, y, w, h, img.shape):
        cv2.imshow('roi', roi)
        cv2.waitKey(1)


def save_face_crop_aligned(sp, face_path, img_path, loc):
    img = cv2.imread(img_path)
    # loc = (d.top(), d.right(), d.bottom(), d.left())
    d = dlib.rectangle(loc[3], loc[0], loc[1], loc[2])
    shape = sp(img, d)

    aligned_face = dlib.get_face_chip(img, shape)
    cv2.imwrite(face_path, aligned_face)


def load_faces_from_keywords_csv(faces_csv_path):
    faces_files = {}
    with open(faces_csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for i, line in enumerate(csvreader):
            if i == 0:
                continue
            filename = line[0]
            if len(line) == 2:
                faces_files[filename] = line[1]

    return faces_files


def get_faces_in_files(preds_per_person, folder=None, ignore_unknown=False):
    faces_files = {}
    if folder == None:
        for p in preds_per_person:
            if ignore_unknown:
                if p == 'unknown' or p == 'deleted':
                    continue
            for i, f in enumerate(preds_per_person[p]):
                # if f[0][0] != 'deleted':
                imgname = os.path.basename(f[1])
                new_name = os.path.splitext(imgname)[0] + os.path.splitext(imgname)[1].upper()
                new_f = os.path.join(os.path.dirname(f[1]), new_name)
                if not new_f in faces_files:
                    faces_files[new_f] = []
                faces_files[new_f].append((p, i))
    else:
        if os.path.isdir(folder):
            for p in preds_per_person:
                if ignore_unknown:
                    if p == 'unknown' or p == 'deleted':
                        continue
                for i, f in enumerate(preds_per_person[p]):
                    imgname = os.path.basename(f[1])
                    new_name = os.path.splitext(imgname)[0] + os.path.splitext(imgname)[1].upper()
                    new_f = os.path.join(os.path.dirname(f[1]), new_name)
                    if os.path.dirname(new_f) == folder:
                        if not new_f in faces_files:
                            faces_files[new_f] = []
                        faces_files[new_f].append((p, i))

    return faces_files


def face_intersect(p1, p2):
    if not p1.intersects(p2):
        return False
    else:
        if p1.contains(p2) or p2.contains(p1):
            return True
        else:
            return False


def get_iou(p1, p2):
    polygons = [p1, p2]
    union = cascaded_union(polygons)
    return p1.intersection(p2).area / union.area


# def remove_overlaps(locs1, descs1):
#   locs = locs1.copy()
#   descs = descs1.copy()
#   for j in range(0, len(locs1)-1):
#     l = locs1[j]
#     p2 = Polygon([(l[3], l[0]), (l[1], l[0]), (l[1], l[2]), (l[3], l[2])])
#     for i in range(j+1, len(locs1)):
#       l1 = locs1[i]
#       p1 = Polygon([(l1[3], l1[0]), (l1[1], l1[0]), (l1[1], l1[2]), (l1[3], l1[2])])
#       if face_intersect(p1, p2):
#         locs.pop(j)
#         descs.pop(j)
#
#   return locs, descs

def merge_detections(locs1, descs1, locs2, descs2, return_diff=False):
    locs = locs1.copy()
    descs = descs1.copy()
    l_diff = []
    d_diff = []
    for j in range(0, len(locs2)):
        intersect = False
        l = locs2[j]
        p2 = Polygon([(l[3], l[0]), (l[1], l[0]), (l[1], l[2]), (l[3], l[2])])
        for i in range(0, len(locs1)):
            l1 = locs1[i]
            p1 = Polygon([(l1[3], l1[0]), (l1[1], l1[0]), (l1[1], l1[2]), (l1[3], l1[2])])

            if p1.intersects(p2):
                iou = get_iou(p1, p2)
                # print(iou)
                if iou > 0.5:
                    intersect = True

        if not intersect:
            locs.append(locs2[j])
            descs.append(descs2[j])
            if return_diff:
                l_diff.append(locs2[j])
                d_diff.append(descs2[j])

    if return_diff:
        return l_diff, d_diff
    else:
        return locs, descs


def get_timestamp(f):
    timeStamp = datetime.now()
    no_timestamp = True
    if f.lower().endswith(('.jpg')):
        pil_image = Image.open(f)
        exif = pil_image._getexif()
        if exif != None:
            if exif.get(36868) != None:
                date = exif[36868]
                if is_valid_timestamp(date):
                    timeStamp = datetime.strptime(date, '%Y:%m:%d %H:%M:%S')
                    no_timestamp = False
    return timeStamp


def predict_knn(knn_clf, face_encoding, n=7, thresh=0.3):
    encs_list = [face_encoding]
    closest_distances = knn_clf.kneighbors(encs_list, n_neighbors=n)
    # cls = knn_clf.predict(encs_list)

    cls = knn_clf.classes_[knn_clf._y[closest_distances[1][0][0]]]
    dist = closest_distances[0][0][0]
    if dist <= thresh:
        # print(cls)
        return cls
    else:
        vote = {}
        dist = {}
        for i in range(0, n):
            cls = knn_clf.classes_[knn_clf._y[closest_distances[1][0][i]]]
            if vote.get(cls) == None:
                vote[cls] = 0
                dist[cls] = []
            vote[cls] += 1
            dist[cls].append(closest_distances[0][0][i])
            # print('{}: {}'.format(cls, closest_distances[0][0][i]))

        max_cls = ''
        max_n = 0
        for v in vote:
            if vote[v] > max_n:
                max_n = vote[v]
                max_cls = v
            # print('{}: {}'.format(v, vote[v]))

        if max_n >= n / 2 and dist[max_cls][0] <= 0.5:
            # print(max_cls)
            return max_cls
        # else:
        #   print('no majority found')

    return 'unknown'


def autorotate_and_resize(path, out_path, size):
    quality = 80
    image = Image.open(path)
    try:
        exif = image._getexif()
        exif_dict = piexif.load(path)
    except AttributeError as e:
        print("Could not get exif - Bad image!")
        return False

    (width, height) = image.size
    s = size[0] / width
    height *= s
    width *= s
    image = image.resize((int(width), int(height)), Image.BICUBIC)
    if exif:
        exif_dict['Exif'][40962] = int(width)
        exif_dict['Exif'][40963] = int(height)

    if not exif:
        image.save(out_path, quality=quality)
    else:
        orientation_key = 274  # cf ExifTags
        if orientation_key in exif:
            orientation = exif[orientation_key]
            rotate_values = {
                3: 180,
                6: 270,
                8: 90
            }
            if orientation in rotate_values:
                # Rotate and save the picture
                image = image.rotate(rotate_values[orientation])
                exif_dict['0th'][orientation_key] = 1
                exif_dict['Exif'][40962] = int(height)
                exif_dict['Exif'][40963] = int(width)
            image.save(out_path, quality=quality, exif=piexif.dump(exif_dict))
        else:
            exif_dict['0th'][orientation_key] = 1
            image.save(out_path, quality=quality, exif=piexif.dump(exif_dict))

    return True


class FACE:
    def __init__(self, loc, desc, name, timestamp, confirmed):
        self.loc = loc
        self.desc = desc
        self.name = name
        self.path = ''
        self.timestamp = timestamp
        self.confirmed = confirmed


class FACES:
    def __init__(self, faces, names_path):
        self.faces = []
        self.dict_by_name = {}
        self.dict_by_files = {}
        self.dict_by_folders = {}
        self.changed_files = []

        for f in faces:
            self.add(f, do_not_add_to_changed=True)

        self.get_name_id2names(names_path)

        # self.remove_duplictes()

    def add(self, face, do_not_add_to_changed=False):
        img_path = os.path.splitext(face.path)[0] + os.path.splitext(face.path)[1].lower()
        face.path = img_path
        # check if face already exists
        if face.path in self.dict_by_files:
            for f in self.dict_by_files[face.path]:
                if self.get_loc(f) == face.loc:
                    print('face exists already in {}'.format(face.path))
                    return False
        self.faces.append(face)
        self.add_face_to_dicts(face, len(self.faces)-1)

        if not do_not_add_to_changed:
            if not face.path in self.changed_files:
                self.changed_files.append(face.path)

    def add_face_to_dicts(self, face, idx):
        if not face.path in self.dict_by_files:
            self.dict_by_files[face.path] = []
        self.dict_by_files[face.path].append(idx)

        if not face.name in self.dict_by_name:
            self.dict_by_name[face.name] = []
        self.dict_by_name[face.name].append(idx)

        folder = os.path.dirname(face.path)
        if not folder in self.dict_by_folders:
            self.dict_by_folders[folder] = {}
        if not face.path in self.dict_by_folders[folder]:
            self.dict_by_folders[folder][face.path] = []
        self.dict_by_folders[folder][face.path].append(idx)

    def remove_face_from_dicts(self, face_idx):
        face = self.get_face(face_idx)
        self.dict_by_files[face.path].pop(self.dict_by_files[face.path].index(face_idx))
        if len(self.dict_by_files[face.path]) == 0:
            self.dict_by_files.pop(face.path)
        self.dict_by_name[face.name].pop(self.dict_by_name[face.name].index(face_idx))
        if len(self.dict_by_name[face.name]) == 0:
            self.dict_by_name.pop(face.name)
        folder = os.path.dirname(face.path)
        self.dict_by_folders[folder][face.path].pop(self.dict_by_folders[folder][face.path].index(face_idx))
        if len(self.dict_by_folders[folder][face.path]) == 0:
            self.dict_by_folders[folder].pop(face.path)

    def rename(self, face_idx, name):
        self.remove_face_from_dicts(face_idx)
        self.faces[face_idx].name = self.get_name_id(name)
        self.add_face_to_dicts(self.faces[face_idx], face_idx)

        if not self.faces[face_idx].path in self.changed_files:
            self.changed_files.append(self.faces[face_idx].path)

    def get_loc(self, face_idx):
        return self.faces[face_idx].loc

    def get_desc(self, face_idx):
        return self.faces[face_idx].desc

    def get_face_path(self, face_idx):
        return self.faces[face_idx].path

    def get_real_face_name(self, face_idx):
        name_id = self.faces[face_idx].name
        return self.name_id2name[name_id]

    def get_real_name(self, name_id):
        return self.name_id2name[name_id]

    def get_name_id(self, real_name):
        if not real_name in self.name2name_id:
            name_id = len(self.name_id2name)
            self.name_id2name[name_id] = real_name
            self.name2name_id = dict(map(reversed, self.name_id2name.items()))
            return name_id
        return self.name2name_id[real_name]

    def get_face(self, face_idx):
        return self.faces[face_idx]

    def flip_confirmed(self, face_idx):
        if self.faces[face_idx].confirmed == 0:
            self.faces[face_idx].confirmed = 1
        else:
            self.faces[face_idx].confirmed = 0

    # def get_faces_dicts(self):
    #     self.dict_by_name = {}
    #     self.dict_by_files = {}
    #     self.dict_by_folders = {}
    #
    #     for i,f in enumerate(self.faces):
    #         self.add_face_to_dicts(f, i)

    def get_name_id2names(self, path):
        self.name_id2name = {}
        names_path = os.path.join(path, 'name_mapping.csv')
        with open(names_path, 'r') as csvfile:
            filereader = csv.reader(csvfile, delimiter=';')
            for row in enumerate(filereader):
                self.name_id2name[row[0]] = row[1][1]
        self.name2name_id = dict(map(reversed, self.name_id2name.items()))

    def store_name_id2_names(self, path):
        names_path = os.path.join(path, 'name_mapping.csv')
        with open(names_path, "w") as csvfile:
            filewriter = csv.writer(csvfile, delimiter=';')
            for n in self.name_id2name:
                filewriter.writerow([n, self.name_id2name[n]])

    def store_file_to_img_labels(self, file, timestamp=''):
        if timestamp == '':
            face = self.get_face(self.dict_by_files[file][0])
            timestamp = face.timestamp
        img_labels = IMG_LABELS(timestamp)
        if file in self.dict_by_files:
            for f in self.dict_by_files[file]:
                if self.get_real_face_name(f) != 'DELETED':
                    img_labels.faces.append(self.get_face(f))

        print('Saving changes in {}'.format(file))
        bin_path = file + '.pkl'
        with open(bin_path, 'wb') as fid:
            pickle.dump(img_labels, fid)

    def store_to_img_labels(self, path):
        for to_save in self.changed_files:
            self.store_file_to_img_labels(to_save)

        self.store_name_id2_names(path)
        self.changed_files = []

        print('Stored faces to .pkl files.')

    def remove_duplictes(self):
        for df in self.dict_by_files:
            for f1 in self.dict_by_files[df]:
                l1 = self.get_loc(f1)
                n1 = self.get_real_face_name(f1)
                for f2 in self.dict_by_files[df]:
                    if f1 == f2:
                        continue
                    l2 = self.get_loc(f2)
                    n2 = self.get_real_face_name(f2)
                    if l1 == l2 and n1 != 'DELETED' and n2 != 'DELETED':
                        self.rename(f2, 'DELETED')

class IMG_LABELS:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.path = ''

        self.faces = []

    def get_pkl_file_path(self):
        return self.path + '.pkl'


def load_img_labels(root_path):
    files = get_files_in_dir_by_ext_rec(root_path, '.pkl')

    faces = []
    img_labels = {}
    for f in files:

        # get corresponding image path
        img_path = os.path.splitext(f)[0]
        img_path = os.path.splitext(img_path)[0] + os.path.splitext(img_path)[1].lower()

        if not os.path.exists(img_path):
            print('{} does not exist'.format(img_path))
            continue

        # load pkl file
        with open(f, 'rb') as fid:
            img_label = pickle.load(fid)
        img_label.path = img_path
        img_labels[img_label.path] = img_label

        # add faces from pkl to global face db/list
        for face in img_label.faces:
            face.path = img_path
            face.changed = False
            faces.append(face)

    return faces, img_labels
