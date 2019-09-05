import os.path
import pickle
import argparse
import random
import numpy as np
import copy
#import cv2
import subprocess

import utils

def show_class(args, svm_clf):

    preds_per_person = utils.load_faces_from_csv(args.db)

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

            if ix >= nr_of_faces:
                ix = 0

            elif ix < 0:
                ix = nr_of_faces-1

            # if mask folder is provided, show only faces within this folder
            if args.mask_folder != None:
                # skip all faces which do not belong to mask_folder
                while (os.path.dirname(preds_per_person[cls][ix][1]) != args.mask_folder and ix < nr_of_faces - 1):
                    ix += 1
                # check if the face at ix belongs to mask_folder, if not, exit
                if os.path.dirname(preds_per_person[cls][ix][1]) != args.mask_folder:
                    print('no more faces of class {} found in {}'.format(cls, args.mask_folder))
                    break

            while len(save) > 10:
                save.pop(0)

            image_path = preds_per_person[cls][ix][1]
            print(preds_per_person[cls][ix][1])

            names, probs = utils.predict_face_svm(preds_per_person[cls][ix][2], svm_clf)

            str_count = str(ix + 1) + ' / ' + str(len(preds_per_person[cls]))
            key, clicked_class, clicked_idx, clicked_names = utils.show_faces_on_image(svm_clf, names, cls, ix, preds_per_person, faces_files[image_path], image_path, waitkey=True, text=str_count)
            utils.evaluate_key(args, key, preds_per_person, clicked_class, clicked_idx, save, clicked_names, faces_files)
            if key == 46 or key == 47: # key '.' or key '/'
                ix += 1
            elif key == 44: # key ','
                ix -= 1
            elif key == 114: # key 'r'
                ix = random.randint(0, nr_of_faces-1)
            elif key == 102: #key 'f'
                while (preds_per_person[cls][ix][3] != 0 and ix < nr_of_faces - 1):
                    ix += 1
            elif key == 98: #key 'b'
                if len(save) > 0:
                    preds_per_person = copy.deepcopy(save.pop())
                    print("undone last action")

        utils.export_persons_to_csv(preds_per_person, args.db)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--face', type=str, required=True,
                      help="Face to show ('all' shows all faces).")
  parser.add_argument('--svm', type=str, required=True,
                      help="Path to svm model file (e.g. svm.clf).")
  parser.add_argument('--db', type=str, required=True,
                      help="Path to folder with predicted faces (.csv files).")
  parser.add_argument('--dets', type=str, required=False, default=None,
                      help="Root path of the detections.bin files.")
  parser.add_argument('--mask_folder', type=str, required=False, default=None,
                      help="Mask folder for faces. Only faces of images within this folder will be shown.")
  args = parser.parse_args()

  #TODO: remove dets
  if not os.path.isdir(args.db):
      print('args.db is not a valid directory')

  if os.path.isfile(args.svm):
      with open(args.svm, 'rb') as f:
          svm_clf = pickle.load(f)
  else:
      print('args.svm ({}) is not a valid file'.format(args.svm))
      exit()

  print('Showing detections of class {}'.format(args.face))
  show_class(args, svm_clf)
  print('Done.')

if __name__ == "__main__":
  main()