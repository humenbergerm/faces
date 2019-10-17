import os.path
import pickle
import argparse
import random
import numpy as np
import copy
#import cv2
import subprocess

import utils

def increment_from(start, preds_per_person, cls, mask, nr_of_faces):
  ix = start + 1
  if ix >= nr_of_faces:
    return 0, True
  while mask[preds_per_person[cls][ix][1]][preds_per_person[cls][ix][0][1]] != 1 and ix < nr_of_faces - 1:
    ix += 1
  if ix >= nr_of_faces - 1:
    if utils.get_nr_after_filter(mask, preds_per_person[cls]) == 0:
      return 0, False
    else:
      # return 0, True # roll over to zero
      return increment_from(-1, preds_per_person, cls, mask, nr_of_faces)
  else:
    return ix, True

def decrement_from(start, preds_per_person, cls, mask, nr_of_faces):
  ix = start - 1
  if ix < 0:
    return nr_of_faces-1, True
  while mask[preds_per_person[cls][ix][1]][preds_per_person[cls][ix][0][1]] != 1 and ix >= 0:
    ix -= 1
  if ix <= 0:
    if utils.get_nr_after_filter(mask, preds_per_person[cls]) == 0:
      return 0, False
    else:
      # return nr_of_faces - 1, True # roll over to the end
      return decrement_from(nr_of_faces, preds_per_person, cls, mask, nr_of_faces)
  else:
    return ix, True

def show_folder(args, svm_clf):

  preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)

  save = []
  key = 0
  i = 0
  while key != 27:

    faces_files = utils.get_faces_in_files(preds_per_person, args.face)

    if i <= 0:
      i = len(faces_files)-1
    if i > len(faces_files)-1:
      i = 0

    image_path = sorted(faces_files.items())[i][0]
    nr_of_faces = len(sorted(faces_files.items())[i][1])
    print(image_path)

    if nr_of_faces != 0:
      cls, ix = sorted(faces_files.items())[i][1][0]

      # names, probs = utils.predict_face_svm(preds_per_person[cls][ix][2], svm_clf)
      names = probs = []

      str_count = str(i + 1) + ' / ' + str(len(faces_files))
      key, clicked_class, clicked_idx, clicked_names = utils.show_faces_on_image(svm_clf, names, cls, ix,
                                                                                 preds_per_person,
                                                                                 faces_files[image_path], image_path,
                                                                                 waitkey=True, text=str_count, draw_main_face=False)
      utils.evaluate_key(args, key, preds_per_person, clicked_class, clicked_idx, save, clicked_names, faces_files)

      if key == 46 or key == 47:  # key '.' or key '/'
        i += 1
      elif key == 44:  # key ','
        i -= 1
      elif key == 114:  # key 'r'
        i = random.randint(0, len(faces_files[i]) - 1)
      elif key == 98:  # key 'b'
        if len(save) > 0:
          preds_per_person = copy.deepcopy(save.pop())
          print("undone last action")
      # else:
      #   faces_files = utils.get_faces_in_files(preds_per_person, args.face)

  utils.export_persons_to_csv(preds_per_person, args.imgs_root, args.db)

def show_class(args, svm_clf):

    preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)
    mask = utils.filter_faces(args, preds_per_person)

    if args.face == 'all':
        classes = preds_per_person
    else:
        classes = [args.face]
        if preds_per_person.get(args.face) == None:
            print('{} not found'.format(classes))
            exit(0)

    for cls in classes:
        nr_of_faces = len(preds_per_person[cls])

        print('{} members of {}'.format(nr_of_faces, cls))
        if nr_of_faces == 0:
            return

        key = 0
        ix = 0
        save = []
        while key != 27 and nr_of_faces > 0:

            nr_of_faces = len(preds_per_person[cls]) # because it might have changed (e.g. if a face got deleted)
            if nr_of_faces == 0:
              print('no more faces of class {} found'.format(cls))
              break

            faces_files = utils.get_faces_in_files(preds_per_person)

            while ix <= nr_of_faces-1 and ix >= 0:
              if mask[preds_per_person[cls][ix][1]][preds_per_person[cls][ix][0][1]] != 1:
                ix += 1
              else:
                break

            if ix >= nr_of_faces:
              ix, ret = increment_from(nr_of_faces-2, preds_per_person, cls, mask, nr_of_faces)
              if not ret:
                break

            elif ix < 0:
              ix, ret = decrement_from(-1, preds_per_person, cls, mask, nr_of_faces)
              if not ret:
                break

            # if mask folder is provided, show only faces within this folder
            if args.mask_folder != None:
                # skip all faces which do not belong to mask_folder
                while (os.path.dirname(preds_per_person[cls][ix][1]) != args.mask_folder and ix < nr_of_faces - 1):
                    ix += 1
                # check if the face at ix belongs to mask_folder, if not, exit
                if os.path.dirname(preds_per_person[cls][ix][1]) != args.mask_folder:
                    print('no more faces of class {} found in {}'.format(cls, args.mask_folder))
                    break

            # # skip all faces which do not meet the filter criteria
            # while mask[preds_per_person[cls][ix][1]][preds_per_person[cls][ix][0][1]] != 1 and ix < nr_of_faces - 1:
            #   ix += 1
            # # check if the face at ix belongs to mask_folder, if not, exit
            # if mask[preds_per_person[cls][ix][1]][preds_per_person[cls][ix][0][1]] != 1 and ix == nr_of_faces - 1:
            #   print('no more faces of class {} found which meet the filter criteria'.format(cls))
            #   break

            while len(save) > 10:
                save.pop(0)

            image_path = preds_per_person[cls][ix][1]
            print(preds_per_person[cls][ix][1])

            names, probs = utils.predict_face_svm(preds_per_person[cls][ix][2], svm_clf)

            str_count = str(ix + 1) + ' / ' + str(utils.get_nr_after_filter(mask, preds_per_person[cls]))
            key, clicked_class, clicked_idx, clicked_names = utils.show_faces_on_image(svm_clf, names, cls, ix, preds_per_person, faces_files[image_path], image_path, waitkey=True, text=str_count)
            deleted_elem_of_cls = utils.evaluate_key(args, key, preds_per_person, clicked_class, clicked_idx, save, clicked_names, faces_files)

            if deleted_elem_of_cls > 0 and clicked_idx <= ix and clicked_class == cls:
              ix -= deleted_elem_of_cls
              # nr_of_faces = len(preds_per_person[cls])

            if key == 46 or key == 47: # key '.' or key '/'
                ix, ret = increment_from(ix, preds_per_person, cls, mask, nr_of_faces)
                if not ret:
                  break
            elif key == 44: # key ','
                ix, ret = decrement_from(ix, preds_per_person, cls, mask, nr_of_faces)
                if not ret:
                  break
            elif key == 114: # key 'r'
                ix = random.randint(0, nr_of_faces-1)
            elif key == 102: #key 'f'
                while (preds_per_person[cls][ix][3] != 0 and ix < nr_of_faces - 1):
                    ix += 1
            elif key == 98: #key 'b'
                if len(save) > 0:
                    preds_per_person = copy.deepcopy(save.pop())
                    print("undone last action")

        utils.export_persons_to_csv(preds_per_person, args.imgs_root, args.db)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--face', type=str, required=True,
                      help="Face to show ('all' shows all faces).")
  parser.add_argument('--svm', type=str, required=True,
                      help="Path to svm model file (e.g. svm.clf).")
  parser.add_argument('--db', type=str, required=True,
                      help="Path to folder with predicted faces (.csv files).")
  parser.add_argument('--imgs_root', type=str, required=True,
                      help="Root directory of your image library.")
  parser.add_argument('--mask_folder', type=str, required=False, default=None,
                      help="Mask folder for faces. Only faces of images within this folder will be shown.")
  parser.add_argument('--min_size', type=str, required=False, default=0,
                      help="Defines the min. size of a face. A face will be accepted if it is larger than min_size * img_height (img_width resp.)")
  args = parser.parse_args()

  if not os.path.isdir(args.db):
      print('args.db is not a valid directory')

  if os.path.isfile(args.svm):
      with open(args.svm, 'rb') as f:
          svm_clf = pickle.load(f)
  else:
      print('args.svm ({}) is not a valid file'.format(args.svm))
      exit()

  if os.path.isdir(args.face):
    print('Showing detections of folder {}'.format(args.face))
    show_folder(args, svm_clf)
  else:
    print('Showing detections of class {}'.format(args.face))
    show_class(args, svm_clf)

  print('Done.')

if __name__ == "__main__":
  main()