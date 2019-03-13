import os.path
import argparse
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
      if not os.path.isfile(symlinkname):
        os.symlink(f[1], symlinkname)

  return album_dir

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--method', type=str, required=True,
                           help="Method of export: 0 ... album")
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
    pSigal = subprocess.Popen(['sigal', 'build', \
                               '--config', 'sigal.conf.py', \
                               '--title', 'FACES', \
                                album_dir, sigal_dir])
    pSigal.wait()

    print('Show album with: sigal serve -c sigal.conf.py {}'.format(sigal_dir))

    print('Done.')