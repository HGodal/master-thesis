import torch
import argparse


def configuration():
    parser = argparse.ArgumentParser()

    parser.add_argument('--DEVICE', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ROOT_DIR', type=str,
                        default='../data/nifti/dataset')
    parser.add_argument('--LOWRES', type=str, default='lowres')
    parser.add_argument('--HIGHRES', type=str, default='highres')
    parser.add_argument('--NAME', type=str, default='pix2pix')
    parser.add_argument('--LR', type=float, default=0.0002)
    parser.add_argument('--BATCH_SIZE', type=int, default=1)
    parser.add_argument('--NUM_WORKERS', type=int, default=2)
    parser.add_argument('--IMAGE_SIZE', type=int, default=256)
    parser.add_argument('--CHANNELS_IMG', type=int, default=1)
    parser.add_argument('--L1_LAMBDA', type=float, default=100)
    parser.add_argument('--NUM_EPOCHS', type=int, default=50)
    parser.add_argument('--LOAD_MODEL', type=bool, default=False)
    parser.add_argument('--SAVE_MODEL', type=bool, default=True)
    parser.add_argument('--CHECKPOINT_DISC', type=str,
                        default='checkpoints/disc.pth.tar')
    parser.add_argument('--CHECKPOINT_GEN', type=str,
                        default='checkpoints/gen.pth.tar')
    parser.add_argument('--NUM_BATCHES', type=int, default=0)
    parser.add_argument('--TRAIN', type=bool, default=True)

    return parser


config = configuration().parse_args()
