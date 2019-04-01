import os.path
import argparse
import json
import subprocess

import utils

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
    print('exporting {}'.format(p))
    for f in preds_per_person[p]:
      if not os.path.isfile(f[1]):
        continue
      json_path = os.path.join(json_dir, f[1].replace('/', '_')[1:-3] + 'json')
      json_path = json_path.replace(' ', '_')
      if os.path.isfile(json_path):
        continue
      else:
        arg_str = 'exiftool -json "' + f[1] + '" > ' + json_path
        os.system(arg_str)

def save_to_exif(args):
  preds_per_person = utils.load_faces_from_csv(args.db)
  if len(preds_per_person) == 0:
    print('no faces loaded')
    exit()

  # keywords_files = {}
  #
  # for p in preds_per_person:
  #   print('exporting {}'.format(p))
  #   for f in preds_per_person[p]:
  #     print(f[1])
  #     if os.path.isfile(f[1]):
  #       if keywords_files.get(f[1]) == None:
  #         keywords_files[f[1]] = ''
  #       keywords_files[f[1]] += p + ' '
  #
  # json_exif = []
  # for k in keywords_files:
  #   exif_image = {}
  #   exif_image['SourceFile'] = k
  #   exif_image['Keywords'] = keywords_files[k]
  #   json_exif.append(exif_image)
  #
  # with open(os.path.join(args.outdir, 'faces.json'), 'w') as fp:
  #   json.dump(json_exif, fp)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--method', type=str, required=True,
                      help="Method of export: 0 ... album, 1 ... exif")
  parser.add_argument('--db', type=str, required=True,
                      help="Path to folder with predicted faces (.csv files).")
  parser.add_argument('--outdir', type=str, required=True,
                      help="Output directory.")
  args = parser.parse_args()

  if not os.path.isdir(args.db):
    print('args.db is not a valid directory')

  if not os.path.isdir(args.outdir):
    utils.mkdir_p(args.outdir)

  if args.method == '0':
    print('Exporting faces as album.')
    album_dir = export_album(args)

    sigal_dir = os.path.join(args.outdir, 'sigal')
    cmd_str = ['sigal ', 'build ', '--config ', 'sigal.conf.py ', '--title ', 'FACES ', album_dir, ' ', sigal_dir]
    # pSigal = subprocess.Popen(cmd_str)
    # pSigal.wait()

    print('To generate a Sigal album use: {}'.format(''.join(str(e) for e in cmd_str)))
    print('Show album with: sigal serve -c sigal.conf.py {}'.format(sigal_dir))
  elif args.method == '1':
    print('Exporting all exif from the images.')
    export_to_json(args)
    print('Saving all faces to the images exif data.')
    save_to_exif(args)
    print('Done.')

if __name__ == "__main__":
  main()