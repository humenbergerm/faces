import os.path
import pickle
import argparse
import random
import numpy as np
import copy
#import cv2
import subprocess

import utils

def predict_face_svm(enc, svm):
    preds = svm.predict_proba(enc.reshape(1, -1))[0]
    sorted = np.argsort(-preds)

    names = []
    probs = []
    print('\n')
    for s in range(10):
        ix = sorted[s]
        names.append(svm.classes_[ix])
        probs.append(preds[ix])
        print('{}: name: {}, prob: {}'.format(s, names[s], probs[s]))

    print('\n')
    return names, probs

def show_class(args, svm_clf):

    preds_per_person = utils.load_faces_from_csv(args.db)

    dets = {}
    if args.dets != None:
        dets, det_file_map = utils.load_detections_as_single_dict(args.dets)

    if args.face == 'all':
        classes = preds_per_person
    else:
        classes = [args.face]
        if preds_per_person.get(args.face) == None:
            print('{} not found'.format(classes))
            exit(0)

    for cls in classes:
        face_locations, face_encodings = utils.initialize_face_data(preds_per_person, cls)

        print('{} members of {}'.format(len(face_locations), cls))
        if len(face_locations) == 0:
            return

        key = 0
        ix = 0
        save = []
        while key != 27 and len(face_locations) > 0:

            if ix >= len(face_locations):
                ix = 0

            elif ix < 0:
                ix = len(face_locations)-1

            # if mask folder is provided, show only faces within this folder
            if args.mask_folder != None:
                # skip all faces which do not belong to mask_folder
                while (os.path.dirname(preds_per_person[cls][ix][1]) != args.mask_folder and ix < len(face_locations) - 1):
                    ix += 1
                # check if the face at ix belongs to mask_folder, if not, exit
                if os.path.dirname(preds_per_person[cls][ix][1]) != args.mask_folder:
                    print('no more faces of class {} found in {}'.format(cls, args.mask_folder))
                    break

            while len(save) > 100:
                save.pop(0)

            image_path = preds_per_person[cls][ix][1]
            print(preds_per_person[cls][ix][1])

            names, probs = predict_face_svm(face_encodings[ix], svm_clf)
            name = names[0]

            #nr_conf, nr_ignored, nr_not_ignored = utils.count_preds_status(preds_per_person[cls])
            #str_count = 'total: ' + str(len(preds_per_person[cls])) + ', confirmed: ' + str(nr_conf) + ', ignored: ' + str(nr_ignored)
            str_count = str(ix+1) + ' / ' + str(len(preds_per_person[cls]))

            predictions = []
            predictions.append((cls, face_locations[ix]))
            key = utils.show_prediction_labels_on_image(predictions, None, preds_per_person[cls][ix][3], 0, preds_per_person[cls][ix][1], str_count)
            if key == 46: # key '.'
                ix += 1
            elif key == 44: # key ','
                ix -= 1
            elif key == 114: # key 'r'
                ix = random.randint(0, len(face_locations))
                while (preds_per_person[cls][ix][3] != 0):
                    ix = random.randint(0, len(face_locations)-1)
            elif key == 99: # key 'c'
                new_name = utils.guided_input(preds_per_person)
                if new_name != "":
                    save.append(copy.deepcopy(preds_per_person[cls]))
                    # add pred in new list
                    if preds_per_person.get(new_name) == None:
                        preds_per_person[new_name] = []
                    utils.insert_element_preds_per_person(preds_per_person, cls, ix, new_name, 1)
                    # delete pred in current list
                    face_locations, face_encodings = utils.delete_element_preds_per_person(preds_per_person, cls, ix)
                    print("face changed: {} ({})".format(new_name, len(preds_per_person[new_name])))
            elif key == 117: # key 'u'
                save.append(copy.deepcopy(preds_per_person[cls]))
                new_name = 'unknown'
                # add pred in new list
                if preds_per_person.get(new_name) == None:
                    preds_per_person[new_name] = []
                utils.insert_element_preds_per_person(preds_per_person, cls, ix, new_name)
                # delete pred in current list
                face_locations, face_encodings = utils.delete_element_preds_per_person(preds_per_person, cls, ix)
                print("face changed: {} ({})".format(new_name, len(preds_per_person[new_name])))
            elif key == 47:  # key '/'
                save.append(copy.deepcopy(preds_per_person[cls]))
                tmp = preds_per_person[cls][ix]
                if tmp[3] == 0:
                    preds_per_person[cls][ix] = tmp[0], tmp[1], tmp[2], 1, tmp[4]
                elif tmp[3] == 1:
                    preds_per_person[cls][ix] = tmp[0], tmp[1], tmp[2], 0, tmp[4]
                #while (preds_per_person[cls][ix][3] == 0 and not ix == len(face_locations) - 1):
                ix += 1
                print("face confirmed: {} ({})".format(tmp[0], len(preds_per_person[cls])))
            elif key == 102: #key 'f'
                while (preds_per_person[cls][ix][3] != 0 and ix < len(face_locations) - 1):
                    ix += 1
            elif key >= 48 and key <= 57: # keys '0' - '9'
                save.append(copy.deepcopy(preds_per_person[cls]))
                new_name = names[key-48]
                utils.insert_element_preds_per_person(preds_per_person, cls, ix, new_name, 1)
                # delete pred in current list
                face_locations, face_encodings = utils.delete_element_preds_per_person(preds_per_person, cls, ix)
                print("face confirmed: {} ({})".format(new_name, len(preds_per_person[new_name])))
            elif key == 100: # key 'd'
                if 1:
                  save.append(copy.deepcopy(preds_per_person[cls]))
                  new_name = 'deleted'
                  # add pred in new list
                  if preds_per_person.get(new_name) == None:
                    preds_per_person[new_name] = []
                  utils.insert_element_preds_per_person(preds_per_person, cls, ix, new_name)
                  # delete pred in current list
                  face_locations, face_encodings = utils.delete_element_preds_per_person(preds_per_person, cls, ix)
                else:
                  save.append(copy.deepcopy(preds_per_person[cls]))
                  # delete face
                  face_locations, face_encodings = utils.delete_element_preds_per_person(preds_per_person, cls, ix)
                print("face deleted")
            elif key == 116: # key 't'
                subprocess.call(["open", "-R", image_path])
                # image = cv2.imread(image_path)
                # imgtrans = cv2.transpose(image)
                # image = cv2.flip(imgtrans, 1)
                # cv2.imshow('rotated', utils.resizeCV(image, 600))
                # cv2.waitKey(1)
            elif key == 97: # key 'a'
                # delete all faces of this class in the current image
                save.append(copy.deepcopy(preds_per_person[cls]))
                i = 0
                while i < len(preds_per_person[cls]):
                    compare_path = preds_per_person[cls][i][1]
                    if preds_per_person[cls][i][1] == image_path:
                        face_locations, face_encodings = utils.delete_element_preds_per_person(preds_per_person, cls, i)
                    else:
                        i += 1
                # delete detections as well
                if len(dets) != 0:
                    utils.delete_detections_of_file(dets, image_path)
                    print("all faces in {} deleted".format(image_path))
                else:
                    print('detections not deleted from detections.bin')
            elif key == 105: # key 'i'
                # delete all faces of this class in the current image AND set it to be ignored in the future (also for detection)
                save.append(copy.deepcopy(preds_per_person[cls]))
                i = 0
                while i < len(preds_per_person[cls]):
                    compare_path = preds_per_person[cls][i][1]
                    if compare_path == image_path:
                        face_locations, face_encodings = utils.delete_element_preds_per_person(preds_per_person, cls, i)
                    else:
                        i += 1
                # ignore detections in the future
                if len(dets) != 0:
                    utils.ignore_detections_of_file(dets, image_path)
                    #print("all faces in {} deleted and image will be ignored".format(image_path))
                else:
                    print('detections not deleted from detections.bin')
            elif key == 98: #key 'b'
                if len(save) > 0:
                    preds_per_person[cls] = copy.deepcopy(save.pop())
                    face_locations, face_encodings = utils.initialize_face_data(preds_per_person, cls)
                    print("undone last action")
            elif key == 115: #key 's'
                utils.export_persons_to_csv(preds_per_person, args.db)
                if args.dets != None:
                  utils.save_detections(dets, det_file_map)
                print('saved')

        utils.export_persons_to_csv(preds_per_person, args.db)
        if args.dets != None:
          utils.save_detections(dets, det_file_map)

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