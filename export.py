import os.path
import argparse
import json
#import subprocess

import utils
import exif

def export_album(args):
  preds_per_person = utils.load_faces_from_csv(args.db)
  if len(preds_per_person) == 0:
    print('no faces loaded')
    exit()

  album_dir = os.path.join(args.outdir, 'faces')
  utils.mkdir_p(album_dir)

  for p in preds_per_person:
    print('exporting {}'.format(p))
    face_dir = os.path.join(album_dir, p)
    utils.mkdir_p(face_dir)
    for f in preds_per_person[p]:
      symlinkname = os.path.join(face_dir, os.path.basename(f[1]))
      if not os.path.islink(symlinkname):
        os.symlink(f[1], symlinkname)

  return album_dir

def export_to_json(args):
  preds_per_person = utils.load_faces_from_csv(args.db)
  if len(preds_per_person) == 0:
    print('no faces loaded')
    exit()

  json_dir = os.path.join(args.db, 'exif_json')
  if not os.path.isdir(json_dir):
    utils.mkdir_p(json_dir)

  for p in preds_per_person:
    if p == 'deleted' or p == 'unknown':
      continue

    print('exporting {}'.format(p))

    for f in preds_per_person[p]:
      # check mask
      if args.mask_folder != None:
        if os.path.dirname(f[1]) != args.mask_folder:
          continue
      if not os.path.isfile(f[1]):
        continue
      json_path = json_dir + f[1][:-3] + 'json'
      if os.path.isfile(json_path):
        continue
      if not os.path.isdir(os.path.dirname(json_path)):
        utils.mkdir_p(os.path.dirname(json_path))

      arg_str = 'exiftool -json "' + f[1] + '" > "' + json_path + '"'
      os.system(arg_str)

def save_to_exif(args):
  face_prefix = 'f '
  # json_dir = os.path.join(args.db, 'exif_json')

  preds_per_person = utils.load_faces_from_csv(args.db)
  if len(preds_per_person) == 0:
    print('no faces loaded')
    exit()

  keywords_files = {}

  for p in preds_per_person:
    #print('exporting {}'.format(p))
    for f in preds_per_person[p]:
      # check mask
      if args.mask_folder != None:
        if os.path.dirname(f[1]) != args.mask_folder:
          continue

      if os.path.isfile(f[1]):
        if keywords_files.get(f[1]) == None:
          keywords_files[f[1]] = []
        if p != 'unknown' and p != 'deleted':
          keywords_files[f[1]].append(face_prefix + p)

  if args.mask_folder == None:
    all_images = utils.get_images_in_dir_rec(args.imgs_root)
  else:
    all_images = utils.get_images_in_dir_rec(args.mask_folder)

  for i,k in enumerate(all_images):
    if args.mask_folder != None:
      if os.path.dirname(k) != args.mask_folder:
        continue
    changed = False
    print('processing exif {}/{} ... {}'.format(i, len(all_images), k))
    ex = exif.ExifEditor(k)

    tag = ex.getTag('ImageDescription')
    if tag != os.path.basename(os.path.dirname(k)):
      ex.setTag('ImageDescription', os.path.basename(os.path.dirname(k)))
    #   exif_image[0]['ImageDescription'] = os.path.basename(os.path.dirname(k))
    #   changed = True
    # if exif_image[0].get('Keywords') == None:
    #   exif_image[0]['Keywords'] = []
    #   changed = True
    # if exif_image[0].get('XPKeywords') != None:
    #   exif_image[0]['XPKeywords'] = []
    #   changed = True
    # if exif_image[0].get('LastKeywordXMP') != None:
    #   exif_image[0]['LastKeywordXMP'] = []
    #   changed = True

    # get face keywords (they start with 'f ')
    kw_faces_exif = []
    kw_others = []
    kws = ex.getKeywords()

    # multiple keywords found
    for kw in kws:
      if kw[:2] == face_prefix:
        kw_faces_exif.append(kw)
      else:
        kw_others.append(kw)

    new_kws = []

    if keywords_files.get(k) == None:
      if args.overwrite:
        changed = True
    else:
      if set(keywords_files[k]) != set(kw_faces_exif):
        new_kws = keywords_files[k]
        if not args.overwrite:
          new_kws = new_kws + kw_others
        changed = True

    if changed:
      ex.setKeywords(new_kws)

    else:
      print('no change in exif found')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--method', type=str, required=True,
                      help="Method of export: 0 ... album, 1 ... exif")
  parser.add_argument('--db', type=str, required=True,
                      help="Path to folder with predicted faces (.csv files).")
  parser.add_argument('--outdir', type=str,
                      help="Output directory.")
  parser.add_argument('--mask_folder', type=str, required=False, default=None,
                      help="Mask folder for faces. Only faces of images within this folder will be processed.")
  parser.add_argument('--imgs_root', type=str, required=True,
                      help="Root directory of your image library.")
  parser.add_argument('--overwrite', help='Overwrite all keywords in EXIF data.', default=False,
                      action='store_true')
  args = parser.parse_args()

  if not os.path.isdir(args.db):
    print('args.db is not a valid directory')

  if args.method == '0':
    if args.outdir == None:
      print('Provide output directory.')
      exit()
    if not os.path.isdir(args.outdir):
      utils.mkdir_p(args.outdir)

    print('Exporting faces as album.')
    album_dir = export_album(args)

    sigal_dir = os.path.join(args.outdir, 'sigal')
    cmd_str = ['sigal ', 'build ', '--config ', 'sigal.conf.py ', '--title ', 'FACES ', album_dir, ' ', sigal_dir]
    # pSigal = subprocess.Popen(cmd_str)
    # pSigal.wait()

    print('To generate a Sigal album use: {}'.format(''.join(str(e) for e in cmd_str)))
    print('Show album with: sigal serve -c sigal.conf.py {}'.format(sigal_dir))
  elif args.method == '1':
    # print('Exporting all exif from the images.')
    # export_to_json(args)
    print('Saving all faces to the images exif data.')
    save_to_exif(args)
    print('Done.')

if __name__ == "__main__":
  main()