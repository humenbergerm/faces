import torch
from tqdm import tqdm
import pretrainedmodels
import pretrainedmodels.utils as utils_pretrained
import argparse
import os, shutil
import utils
import multiprocessing
import dlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import tkinter
from PIL import ImageTk
from PIL import Image
import cv2
from imdirect import imdirect_open
from sklearn.preprocessing import Normalizer
import random

load_img = utils_pretrained.LoadImage()


def _is_valid_img(img_path):
    try:
        load_img(img_path)
        return True
    except Exception:
        return False


def filter_invalid_images(img_paths, num_workers=4, progress=False):
    """Filter invalid images before computing expensive features."""
    with multiprocessing.Pool(num_workers) as p:
        if progress:
            load_works = list(tqdm(
                p.imap(_is_valid_img, img_paths),
                total=len(img_paths),
                desc="Filtering invalid images"))
        else:
            load_works = p.map(_is_valid_img, img_paths)

    img_paths = [
        img_path for img_path, is_loadable in
        zip(img_paths, load_works) if is_loadable
    ]
    return img_paths

def get_model(model_name):
    model = getattr(pretrainedmodels, model_name)(pretrained='imagenet')
    model.eval()
    return model

class ImageLoader():
    def __init__(self, img_paths, model, img_size=224, augment=False):
        self.load_img = utils_pretrained.LoadImage()
        additional_args = {}
        if augment:
            additional_args = {
                'random_crop': True, 'random_hflip': False,
                'random_vflip': False
            }
        self.tf_img = utils_pretrained.TransformImage(
            model, scale=img_size / 256, **additional_args)
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        input_img = self.load_img(self.img_paths[idx])
        input_tensor = self.tf_img(input_img)
        return input_tensor


def image_features(
        img_paths, model_name='resnet50', use_gpu=torch.cuda.is_available(),
        batch_size=32, num_workers=4, progress=False, augment=False):
    """
    Extract deep learning image features from images.

    Args:
        img_paths(list): List of paths of images to extract features from.
        model_name(str, optional): Deep learning model to use for feature
            extraction. Default is resnet50. List of avaiable models are here:
            https://github.com/Cadene/pretrained-models.pytorch
        use_gpu(bool): If gpu is to be used for feature extraction. By default,
            uses cuda if nvidia driver is installed.
        batch_size(int): Batch size to be used for feature extraction.
        num_workers(int): Number of workers to use for image loading.
        progress(bool): If true, enables progressbar.
        augment(bool): If true, images are augmented before passing through
            the model. Useful if you're training a classifier based on these
            features.
    """
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if isinstance(img_paths, str):
        raise ValueError(f'img_paths should be a list of image paths.')

    model = get_model(model_name).to(device)
    dataset = ImageLoader(img_paths, model, augment=augment)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        if progress:
            pbar = tqdm(total=len(img_paths), desc='Computing image features')

        output_features = []
        for batch in dataloader:
            batch = batch.to(device)
            ftrs = model.features(batch).cpu()
            ftrs = ftrs.mean(-1).mean(-1)   # average pool
            output_features.append(ftrs)

            if progress:
                pbar.update(batch.shape[0])

        if progress:
            pbar.close()

    output_features = torch.cat(output_features).numpy()
    return output_features


def topk_retrieved_image(sim, im_fid, topk, map_fnames):
  if im_fid == -1:
    scores = sim
  else:
    scores = sim[im_fid, :]
  pvidxs = np.argsort(-scores)
  sim_list = []
  k = 0
  for s in range(len(pvidxs)):
    if k >= topk:
      break
    else:
      sim_list.append(map_fnames[pvidxs[s]])
      k = k + 1

  return sim_list

class ImageViewer:

  def __init__(self, img_list):
    self.img_list = img_list
    self. index = 0
    self.quit = False

  # process the interaction
  def event_action(self, event):
    print(repr(event))
    event.widget.quit()

  # clicks
  def clicked(self, event):
    self.event_action(event)

  # keys
  def key_press(self, event):
    self.event_action(event)
    if event.keysym == 'Left':
      self.index -= 1
    elif event.keysym == 'Right':
      self.index += 1
    elif event.keysym == 'Escape':
      self.quit = True

  def run(self):
    # set up the gui
    window = tkinter.Tk()
    window.bind("<Button>", self.clicked)
    window.bind("<Key>", self.key_press)

    while not self.quit:

      if self.index < 0:
        self.index = len(self.img_list)-1
      if self.index >= len(self.img_list):
        self.index = 0

      # print(self.index)

      window.title(self.img_list[self.index])
      picture = imdirect_open(self.img_list[self.index])
      picture_width = int(picture.size[0] / 5)
      picture_height = int(picture.size[1] / 5)
      picture = picture.resize((picture_width, picture_height), Image.ANTIALIAS)
      tk_picture = ImageTk.PhotoImage(picture)
      window.geometry("{}x{}+200+200".format(picture_width, picture_height))
      image_widget = tkinter.Label(window, image=tk_picture)
      image_widget.place(x=0, y=0, width=picture_width, height=picture_height)

      # wait for events
      window.mainloop()

    window.destroy()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', type=str, required=True,
                      help="Input image directory. Recursive processing is supported.")
  parser.add_argument('--imgs_root', type=str, required=True,
                      help="Root directory of your image library.")
  # parser.add_argument('--outdir', type=str, required=True,
  #                     help="Output directory.")
  parser.add_argument('--db', type=str, required=True,
                      help="Path to folder with computed features.")
  parser.add_argument('--outdir', type=str, required=True,
                      help="Path to folder with clustered images.")
  parser.add_argument('--recompute', help='Recompute detections.',
                      action='store_true')
  args = parser.parse_args()

  if not os.path.isdir(args.input):
    print('args.input needs to be a valid folder containing images')
    exit()

  descs_path = os.path.join(args.outdir, 'descs.pkl')
  imgslist_path = os.path.join(args.outdir, 'imglist.pkl')

  if os.path.exists(imgslist_path):
    print('Loading image features.')
    with open(imgslist_path, 'rb') as f:
      img_paths = pickle.load(f)
    if os.path.exists(descs_path):
      with open(descs_path, 'rb') as f:
        descriptors = pickle.load(f)
  else:
    print('Computing image features for {}'.format(args.input))
    img_paths = utils.get_images_in_dir_rec(args.input)
    random.shuffle(img_paths)
    img_paths = img_paths[:10000]
    with open(imgslist_path, 'wb') as f:
      pickle.dump(img_paths, f)

    img_features = image_features(filter_invalid_images(img_paths), batch_size=4, progress=True, model_name='resnet50')

    descriptors = []
    for p in img_features:
      descriptors.append(p)

    l2_normalizer = Normalizer()
    descriptors = l2_normalizer.fit_transform(descriptors)

    with open(descs_path, 'wb') as f:
      pickle.dump(descriptors, f)

  #PCA
  # pca = PCA(n_components=128)
  # descriptors = pca.fit_transform(descriptors)
  # tsne = TSNE(n_components=2)
  # descriptors = tsne.fit_transform(descriptors)

  # for i in range(0, 40):
  #   sim = descriptors[i].dot(np.array(descriptors).T)
  #   topklist = topk_retrieved_image(sim, -1, 15, img_paths)
  #   topklist.insert(0, img_paths[i])
  #
  #   iv = ImageViewer(topklist)
  #   iv.run()
  #
  # exit()

  # Clustering
  cluster = KMeans(n_clusters=50, random_state=0).fit(np.array(descriptors))
  # cluster = MeanShift(cluster_all=False).fit(np.array(descriptors))
  # cluster = DBSCAN(eps=0.5, min_samples=3).fit(np.array(descriptors))

  # Copy images renamed by cluster
  # Check if target dir exists
  clustered_imgs = {}
  # try:
  #   os.makedirs(args.outdir)
  # except OSError:
  #   pass
  # Copy with cluster name
  # print("\n")
  for i, m in enumerate(cluster.labels_):
    # print("Copy: {} / {}".format(i, len(cluster.labels_)))
    # dst_path = args.outdir + '/' + str(m) + '/' + os.path.basename(img_paths[i])
    # if not os.path.exists(os.path.dirname(dst_path)):
    #   os.makedirs(os.path.dirname(dst_path))
    # shutil.copy(img_paths[i], dst_path)
    if clustered_imgs.get(m) == None:
      clustered_imgs[m] = []
    clustered_imgs[m].append(img_paths[i])

  for c in clustered_imgs:
    iv = ImageViewer(clustered_imgs[c])
    iv.run()

  print('Done.')

if __name__ == "__main__":
    main()