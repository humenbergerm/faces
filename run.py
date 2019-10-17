import os
import os.path
import pickle
import argparse
import dlib
import cv2
from shapely.geometry import Polygon

import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True,
                        help="Path to folder with predicted faces (.csv files).")
    args = parser.parse_args()

    preds_per_person = utils.load_faces_from_csv(args.db, args.imgs_root)
    faces_files = utils.get_faces_in_files(preds_per_person)

    to_delete = []
    for n,p in enumerate(preds_per_person['deleted']):
      for f in faces_files[p[1]]:
        cls, i = f
        if cls != 'deleted':
          if p[0][1] == preds_per_person[cls][i][0][1]:
            print(cls)
            to_delete.append(n)

    to_delete = sorted(to_delete)
    to_delete.reverse()
    for i in to_delete:
      print(i)
      if len(preds_per_person['deleted'])-1 >= i:
        preds_per_person['deleted'].pop(i)  # if not, they would exist double, in 'deleted' and in the cluster group

    utils.export_persons_to_csv(preds_per_person, args.imgs_root, args.db)

if __name__ == "__main__":
    main()