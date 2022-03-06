import torch
import argparse
import adsh

from loguru import logger
from data.helper_functions import CocoDetection
import torchvision.transforms as transforms
from data.helper_functions import CutoutPIL
from randaugment import RandAugment


def run():
    args = load_config()
    logger.add('logs/{time}.log', rotation='500 MB', level='INFO')
    logger.info(args)

    # torch.backends.cudnn.benchmark = True

    # Load dataset
    # query_dataloader, _, retrieval_dataloader = load_data(
    #     args.dataset,
    #     args.root,
    #     args.num_query,
    #     args.num_samples,
    #     args.batch_size,
    #     args.num_workers,
    # )

    train_dataset = CocoDetection(root='./',
                                  annFile='../data/coco/annotations/instances_val2017.json',
                                  transform= transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ]))
    
    retrieval_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    for code_length in args.code_length:
        mAP = adsh.train(
            args,
            # query_dataloader,
            retrieval_dataloader,
            code_length,
            args.device,
            args.lr,
            args.max_iter,
            args.max_epoch,
            args.num_samples,
            args.batch_size,
            args.root,
            args.dataset,
            args.gamma,
            args.topk,
        )
        logger.info('[code_length:{}][map:{:.4f}]'.format(code_length, mAP))


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='ADSH_PyTorch')
    parser.add_argument('--dataset',
                        help='Dataset name.')
    parser.add_argument('--root',
                        help='Path of dataset')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size.(default: 64)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--code-length', default='12,24,32,48', type=str,
                        help='Binary hash code length.(default: 48,64,84,108)')
    parser.add_argument('--max-iter', default=50, type=int,
                        help='Number of iterations.(default: 50)')
    parser.add_argument('--max-epoch', default=3, type=int,
                        help='Number of epochs.(default: 3)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num-samples', default=2000, type=int,
                        help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--gamma', default=200, type=float,
                        help='Hyper-parameter.(default: 200)')


    parser.add_argument('--model-name', default='tresnet_l')
    parser.add_argument('--model-path',
                        default='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth',
                        type=str)
    parser.add_argument('--num-classes', default=80)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--image-size', default=448, type=int,
                        metavar='N', help='input image size (default: 448)')

    # ML-Decoder
    parser.add_argument('--use-ml-decoder', default=1, type=int)
    parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
    parser.add_argument('--decoder-embedding', default=768, type=int)
    parser.add_argument('--zsl', default=0, type=int)
    

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))

    return args


if __name__ == '__main__':
    run()
