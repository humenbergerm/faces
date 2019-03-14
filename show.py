import os.path
import pickle
import argparse
import random
import numpy as np
import copy

import utils

def predict_face_svm(enc, svm):
    preds = svm.predict_proba(enc.reshape(1, -1))[0]
    sorted = np.argsort(-preds)

    names = []
    probs = []
    for s in range(10):
        ix = sorted[s]
        names.append(svm.classes_[ix])
        probs.append(preds[ix])
        print('{}: name: {}, prob: {}'.format(s, names[s], probs[s]))

    print('\n\n')
    return names, probs

def show_class(args, svm_clf):

    preds_per_person = utils.load_faces_from_csv(args.db)

    if args.face == 'all':
        classes = preds_per_person
    else:
        classes = [args.face]

    for cls in classes:
        face_locations, face_encodings = utils.initialize_face_data(preds_per_person, cls)

        print('{} members of {}'.format(len(face_locations), cls))
        if len(face_locations) == 0:
            return

        key = 0
        ix = 0
        save = []
        while key != 27 and len(face_locations) > 0:

            while len(save) > 100:
                save.pop(0)

            if ix >= len(face_locations):
                ix = 0

            elif ix < 0:
                ix = len(face_locations)-1

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
                    tmp = preds_per_person[cls][ix]
                    preds_per_person[new_name].append(((new_name, tmp[0][1]), tmp[1], tmp[2], tmp[3], tmp[4]))
                    # delete pred in current list
                    face_locations, face_encodings = utils.delete_element_preds_per_person(preds_per_person, cls, ix)
                    print("face changed: {} ({})".format(new_name, len(preds_per_person[new_name])))
            elif key == 117: # key 'u'
                save.append(copy.deepcopy(preds_per_person[cls]))
                new_name = 'unknown'
                # add pred in new list
                if preds_per_person.get(new_name) == None:
                    preds_per_person[new_name] = []
                tmp = preds_per_person[cls][ix]
                preds_per_person[new_name].append(((new_name, tmp[0][1]), tmp[1], tmp[2], tmp[3], tmp[4]))
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
                tmp = preds_per_person[cls][ix]
                preds_per_person[new_name].append(((new_name, tmp[0][1]), tmp[1], tmp[2], tmp[3], tmp[4]))
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
                  tmp = preds_per_person[cls][ix]
                  preds_per_person[new_name].append(((new_name, tmp[0][1]), tmp[1], tmp[2], tmp[3], tmp[4]))
                  # delete pred in current list
                  face_locations, face_encodings = utils.delete_element_preds_per_person(preds_per_person, cls, ix)
                else:
                  save.append(copy.deepcopy(preds_per_person[cls]))
                  # delete face
                  face_locations, face_encodings = utils.delete_element_preds_per_person(preds_per_person, cls, ix)
                print("face deleted")
            elif key == 97: # key 'a'
                # delete all faces of this class in the current image
                save.append(copy.deepcopy(preds_per_person[cls]))
                image_path = preds_per_person[cls][ix][1]
                i = 0
                while i < len(preds_per_person[cls]):
                    compare_path = preds_per_person[cls][i][1]
                    if preds_per_person[cls][i][1] == image_path:
                        face_locations, face_encodings = utils.delete_element_preds_per_person(preds_per_person, cls, i)
                    else:
                        i += 1
                print("all faces in {} deleted".format(image_path))

            elif key == 98: #key 'b'
                if len(save) > 0:
                    preds_per_person[cls] = copy.deepcopy(save.pop())
                    face_locations, face_encodings = utils.initialize_face_data(preds_per_person, cls)
                    print("undone last action")
            elif key == 115: #key 's'
                utils.export_persons_to_csv(preds_per_person, args.db)
                print('saved')

        utils.export_persons_to_csv(preds_per_person, args.db)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--face', type=str, required=True,
                             help="Face to show ('all' shows all faces).")
    parser.add_argument('--svm', type=str, required=True,
                        help="Path to svm model file (e.g. svm.clf).")
    parser.add_argument('--db', type=str, required=True,
                             help="Path to folder with predicted faces (.csv files).")
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