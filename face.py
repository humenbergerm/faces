import os.path
import argparse
import subprocess
import sys

import utils
import detect
import predict
import cluster
import train
import show
import export

if __name__ == "__main__":

  task = sys.argv[1]
  sys.argv.pop(1)

  if task == 'detect':
    detect.main()
  elif task == 'predict':
    predict.main()
  elif task == 'cluster':
    cluster.main()
  elif task == 'train':
    train.main()
  elif task == 'show':
    show.main()
  elif task == 'export':
    export.main()