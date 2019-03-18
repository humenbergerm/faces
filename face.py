import sys

import detect
import predict
import cluster
import train
import show
import export

def print_help():
  print('Usage: python3 face.py COMMAND\n\n'
        'COMMAND:\n'
        'detect \t\t... detect faces in images\n'
        'cluster \t... group similar faces into clusters\n'
        'train \t\t... train face recognition using faces in folders\n'
        'show \t\t... show face recognition results\n'
        'export \t\t... export face recognition results\n')

if __name__ == "__main__":

  if len(sys.argv) == 1:
    print_help()
    exit()

  task = sys.argv[1]
  sys.argv.pop(1)

  if task == '--help' or task == 'help' or task == '-h':
    print_help()
  elif task == 'detect':
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
  else:
    print_help()