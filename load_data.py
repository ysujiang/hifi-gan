# -*- coding: utf-8 -*-
"""
@Time: 2020/7/23
@author: JiangPeipei
"""
from torch.utils.data import DistributedSampler, DataLoader
# from torch.utils.data import DataLoader
from taco_meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
# from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from env import AttrDict, build_env
import json
from taco_utils.audio import AudioProcessor

def load_datas(a,h):
    training_filelist, validation_filelist = get_dataset_filelist(a)
    print(training_filelist)
    trainset = MelDataset(training_filelist,ap, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss,
                          fine_tuning=False)
    # trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
    #                       h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
    #                       shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss,
    #                       fine_tuning=False)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
    for i, batch in enumerate(train_loader):
        # print(batch)
        x, y, _, y_mel = batch
        print(y_mel.shape)
if __name__ == '__main__':

    json_file = './config_v1.json'
    with open(json_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    a=h
    ap = AudioProcessor(**h.audio)
    # print(ap)
    # print(h)
    # # training_filelist = 'LJSpeech-1.1/training.txt'
    load_datas(a,h)

