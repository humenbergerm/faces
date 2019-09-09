import os
import os.path
# import pickle
import argparse
import dlib
import cv2
import numpy as np

import utils

def cluster_faces_in_class(args):
  preds_per_person = utils.load_faces_from_csv(args.db)
  if preds_per_person.get(args.cls) == None:
    print('Class {} not found.'.format(args.cls))
    return

  descriptors = []
  for i,p in enumerate(preds_per_person[args.cls]):
    descriptors.append(dlib.vector(p[2]))

  # cluster the faces
  print('clustering...')
  all_indices = []
  all_lengths = []

  if 1:
    # chinese whispers
    labels = dlib.chinese_whispers_clustering(descriptors, args.threshold)
    num_classes = len(set(labels))
    print("Number of clusters: {}".format(num_classes))
    for j in range(0, num_classes):
      class_length = len([label for label in labels if label == j])
      if class_length >= args.min_members:
        indices = []
        for i, label in enumerate(labels):
          if label == j:
            indices.append(i)
        all_indices.append(indices)
        all_lengths.append(class_length)
  else:
    #DBSCAN
    from sklearn.cluster import DBSCAN
    clt = DBSCAN(eps=args.threshold, metric="euclidean", n_jobs=4, min_samples=args.min_members)
    clt.fit(descriptors)
    labels = np.unique(clt.labels_)
    num_classes = len(np.where(labels > -1)[0])

    if num_classes > 1: # to be checked!!
      print("Number of clusters: {}".format(num_classes))
      for j in labels:
        idxs = np.where(clt.labels_ == j)[0]
        class_length = len(idxs)
        indices = []
        for i in idxs:
          indices.append(i)
        all_indices.append(indices)
        all_lengths.append(class_length)

  sort_index = np.argsort(np.array(all_lengths))[::-1]

  # Move the clustered faces to individual groups
  print('Moving the clustered faces to the database.')
  to_delete = []
  for i in sort_index[:args.max_clusters]:
    cluster_name = "group_" + str(i)

    # export to folders
    if args.export:
      cluster_path = os.path.join(args.outdir, cluster_name)
      if not os.path.isdir(cluster_path):
        os.makedirs(cluster_path)

    to_delete += all_indices[i]
    for n,index in enumerate(all_indices[i]):
      utils.insert_element_preds_per_person(preds_per_person, args.cls, index, cluster_name)
      if args.export:
        file_name = os.path.join(cluster_path, 'face_' + cluster_name + '_' + str(n) + '.jpg')
        utils.save_face_crop(file_name, preds_per_person[args.cls][index][1], preds_per_person[args.cls][index][0][1])

  to_delete = sorted(to_delete)
  to_delete.reverse()
  for i in to_delete:
    preds_per_person[cls].pop(i) # if not, they would exist double, in 'deleted' and in the cluster group
    # utils.delete_element_preds_per_person(preds_per_person, args.cls, i)

  utils.export_persons_to_csv(preds_per_person, args.db)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cls', type=str, required=True,
                      help="Class to cluster.")
  parser.add_argument('--outdir', type=str, required=True,
                      help="Path to folder with clustered faces.")
  parser.add_argument('--threshold', type=float, default=0.45,
                      help="Threshold for clustering (default=0.45). A larger value decreases the number of resulting clusters.")
  parser.add_argument('--min_members', type=int, default=2,
                      help="Minimum number of members for a cluster to be accepted.")
  parser.add_argument('--max_clusters', type=int, default=10,
                      help="Maximum number of clusters to be exported.")
  parser.add_argument('--db', type=str, required=False,
                      help="Path to folder with predicted faces (.csv files).")
  # parser.add_argument('--recompute', help='Recompute clustering.',
  #                     action='store_true')
  parser.add_argument('--export', help='Export clusters to folders.',
                      action='store_true')
  args = parser.parse_args()

  print('Clustering faces in class {}'.format(args.cls))
  cluster_faces_in_class(args)

  print('Done.')

if __name__ == "__main__":
  main()