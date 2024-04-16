import os
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from contextlib import contextmanager
from scipy.signal import fftconvolve
import re
import math
from . import config


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def get_dist(p1, p2):
    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )


def merge_frames(frame_front, frame_back):
    '''merge frames from front and back cameras'''
    frame_back = frame_back[::-1] # vertical mirror
    frame_back = frame_back[:, ::-1] # horizontal mirror
    merged_frame = np.concatenate((frame_front, frame_back), axis=0)
    return merged_frame


def calculate_vector_difference(l1, l2, phi1, phi2):
    '''
    l1 - from ground center to car on first frame (meters)
    l2 - from ground center to car on second frame
    phi - angle between vertical and vector l (rad)
    '''
    x1 = l1 * math.cos(math.pi - phi1)
    y1 = l1 * math.sin(math.pi - phi1)
    x2 = l2 * math.cos(math.pi - phi2)
    y2 = l2 * math.sin(math.pi - phi2)
    # delta = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    delta = abs(y2 - y1)

    return delta


def calculate_vector_difference2(l1, l2, coords1, coords2):
    '''
    calculate k between pixels and meters
    l_pixels, l_meters --> k (meters/pixels)
    find dl in pixels, then convert to meters
    l1, l2 - meters
    coords - coordinates of bboxes
    '''
    x1, y1 = coords1[0], coords1[1]
    x2, y2 = coords2[0], coords2[1]
    xc, yc = (640+config.column_add)//2, 480+config.row_add
    delta_pixels = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    l_pixels_1 = math.sqrt((xc - x1) ** 2 + (yc - y1) ** 2)
    l_pixels_2 = math.sqrt((xc - x2) ** 2 + (yc - y2) ** 2)
    k = (l1 / l_pixels_1 + l2 / l_pixels_2) / 2
    delta = k * delta_pixels
    return delta


def fast_convolution(img, kernel):
    res = fftconvolve(img, kernel, mode='same')
    res[res>255] = 255
    return res.astype(np.uint8)


def offset_calculation(mpu):
    offsets = [[0.0],[0.0],[0.0]]
    buffer_x = []
    buffer_y = []
    buffer_z = []
    for i in range(100):
        # accel_data = mpu.get_gyro_data()
        # buffer_x.append(accel_data['x'])
        # buffer_y.append(accel_data['y'])
        # buffer_z.append(accel_data['z'])
        gyro_data = mpu.get_gyro_data()
                # Угловая скорость
        gx = 1000 * gyro_data['x'] / 32768
        gy = 1000 * gyro_data['y'] / 32768
        gz = 1000 * gyro_data['z'] / 32768
        buffer_x.append(gx)
        buffer_y.append(gy)
        buffer_z.append(gz)
    offsets[0] = sum(buffer_x) / 100
    offsets[1] = sum(buffer_y) / 100
    offsets[2] = sum(buffer_z) / 100

    return offsets

def create_logger(cfg, cfg_path, phase='train', rank=-1):
    # set up logger dir
    dataset = cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_path = os.path.basename(cfg_path).split('.')[0]

    if rank in [-1, 0]:
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        log_file = '{}_{}_{}.log'.format(cfg_path, time_str, phase)
        # set up tensorboard_log_dir
        tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                                  (cfg_path + '_' + time_str)
        final_output_dir = tensorboard_log_dir
        if not tensorboard_log_dir.exists():
            print('=> creating {}'.format(tensorboard_log_dir))
            tensorboard_log_dir.mkdir(parents=True)

        final_log_file = tensorboard_log_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        return logger, str(final_output_dir), str(tensorboard_log_dir)
    else:
        return None, None, None


def select_device(logger=None, device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            if logger:
                logger.info("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        if logger:
            logger.info(f'Using torch {torch.__version__} CPU')

    if logger:
        logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR0,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            #model.parameters(),
            lr=cfg.TRAIN.LR0,
            betas=(cfg.TRAIN.MOMENTUM, 0.999)
        )

    return optimizer


def save_checkpoint(epoch, name, model, optimizer, output_dir, filename, is_best=False):
    model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
    checkpoint = {
            'epoch': epoch,
            'model': name,
            'state_dict': model_state,
            # 'best_state_dict': model.module.state_dict(),
            # 'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }
    torch.save(checkpoint, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in checkpoint:
        torch.save(checkpoint['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
        # elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()