import json
import math

import deprecation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from models import QLeNet5_FP, QGCN_Graph_FP
from models.alexnet_fp import QAlexNet_FP
from models.gin_graph_fp import QGIN_Graph_FP
from models.resnet_fp import QResNet_FP
from utils.logging import QLogger


def gen_json(source, name):
    #epochs = []
    qnet = None
    #print(name)
    if 'resnet' in name.lower():
        qnet = QResNet_FP()
    if 'alexnet' in name.lower():
        qnet = QAlexNet_FP()
    modules = [module for module in qnet.modules()
               if (not 'Sequential' in str(type(module)) and
                   not 'ReLU' in str(type(module)) and
                   not 'QAlexNet_FP' in str(type(module)))]
    epochs = {}
    epochs['name'] = []
    epochs['qwl'] = []
    for tuple in source:
        for i in range(0, tuple[1]):
            '''
            layer = []
            for module in modules:
                layer.append({'name': str(type(module)).split('.')[-1][:-1], 'qwl': tuple[0]})
            epochs.append(layer)
            '''
            names = []
            for module in modules:
                names.append(str(type(module)).split('.')[-1][:-1])
            epochs['name'].append(names)
            epochs['qwl'].append([tuple[0]]*len(modules))

    with open("output/logs/{}".format(name), 'w') as f:
        json.dump(epochs, f)


def make_pairwise(it):
    #https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
    it = iter(it)
    while True:
        try:
            yield next(it), next(it)
        except StopIteration:
            # no more elements in the iterator
            return

def activation(madds, idx, modules, h, w, c):
    if idx + 1 < len(modules):
        if 'Tanh' in str(type(modules[idx + 1])):
            madds += h * w * c * 4
        elif 'ReLU' in str(type(modules[idx + 1])):
            madds += h * w * c
    return madds

def pooling2d(modules, module, idx, img_sz, madds_list):
    oh = (img_sz[0] + module.__dict__['padding'] * 2 - module.__dict__['kernel_size']) // module.__dict__[
        'stride'] + 1
    ow = (img_sz[1] + module.__dict__['padding'] * 2 - module.__dict__['kernel_size']) // module.__dict__[
        'stride'] + 1
    madds = oh * ow * img_sz[2]
    madds = activation(madds, idx, modules, oh, ow, img_sz[2])
    madds_list.append(madds)
    img_sz = (oh, ow, img_sz[2])
    return img_sz

def conv2d(modules, module, idx, img_sz, madds_list):
    mk = module.__dict__['kernel_size'][0]
    nk = module.__dict__['kernel_size'][1]
    co = module.__dict__['out_channels']
    ci = module.__dict__['in_channels']
    oh = (img_sz[0] + module.__dict__['padding'][0] * 2 - module.__dict__['kernel_size'][0]) // \
         module.__dict__['stride'][0] + 1
    ow = (img_sz[1] + module.__dict__['padding'][1] * 2 - module.__dict__['kernel_size'][1]) // \
         module.__dict__['stride'][1] + 1
    madds = 2 * img_sz[0] * img_sz[1] * (ci * mk + 1) * co * img_sz[2]
    madds = activation(madds, idx, modules, oh, ow, img_sz[2])
    madds_list.append(madds)
    img_sz = (oh, ow, module.__dict__['out_channels'])
    return img_sz

def linear(modules, module, idx, img_sz, madds_list):
    m = module.__dict__['in_features']
    n = module.__dict__['out_features']
    madds = 2 * m * n - m
    madds = activation(madds, idx, modules, m, n, img_sz[2])
    madds_list.append(madds)
    #TODO img sz doesnt change here? why?
    return img_sz

def batchnorm2d(modules, module, idx, img_sz, madds_list):
    madds = 4 * img_sz[0] * img_sz[1] * img_sz[2]
    madds = activation(madds, idx, modules, img_sz[0], img_sz[1], img_sz[2])
    madds_list.append(madds)
    return img_sz

def gcnconv(modules, module, idx, img_sz, madds_list):
    #Estimate based on X'=D^(-1/2)*A*D^(-1/2)*X*Phi
    #https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
    m = module.__dict__['in_channels']
    n = module.__dict__['out_channels']
    matmul = 2 * m * n - m
    madds = 4 * matmul + m * n * 2
    madds = activation(madds, idx, modules, img_sz[0], img_sz[1], img_sz[2])
    madds_list.append(madds)
    img_sz = (module.__dict__['in_channels'], module.__dict__['out_channels'], 1)
    return img_sz


def marvin_step(tensordimsprod, config, lb= None, res = None, non_zeros = 1.0, wl = 32):
    if lb is None: lb = config.getint('QUANTIZATION', 'MAX_RESOLUTION') # Worst case estimate
    if res is None: res = config.getint('QUANTIZATION', 'MAX_LOOKBACK')
    bisect = 2 * math.log2(wl-8) if wl > 8 else 1.0
    madds_push_down = 3 * tensordimsprod * bisect * res * non_zeros
    madds_push_up = (lb+1) * tensordimsprod + 1
    return madds_push_down+madds_push_up

def analyse(log, config):
    # https://stackoverflow.com/questions/44193270/how-to-calculate-the-output-size-after-convolving-and-pooling-to-the-input-image
    # https://iq.opengenus.org/output-size-of-convolution/
    # https://www.programmersought.com/article/6010108964/
    # https://mediatum.ub.tum.de/doc/625604/625604
    # https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
    # https://machinethink.net/blog/how-fast-is-my-model /
    is_muppet = 'muppet' in log.lower()
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    metrics = None
    with open("output/logs/"+ log, 'r') as jf:
        metrics = json.load(jf)
    qnet = None

    img_sz = None
    if 'resnet' in log.lower():
        img_sz = (32, 32, 3)
        qnet = QResNet_FP()
    if 'alexnet' in log.lower():
        img_sz = (32, 32, 3)
        qnet = QAlexNet_FP()
    if 'lenet' in log.lower():
        img_sz = (28, 28, 1)
        qnet = QLeNet5_FP()
    if 'gcn_graph' in log.lower():
        qnet = QGCN_Graph_FP("ogbg-molhiv") # eval name no effect on analytics, just requiered for correct init
        img_sz = [p for p in qnet.named_parameters()][0][1].shape
        img_sz = (img_sz[0], img_sz[1], 1)
    if 'gin_graph' in log.lower():
        qnet = QGIN_Graph_FP("ogbg-molhiv") # eval name no effect on analytics, just requiered for correct init
        img_sz = [p for p in qnet.named_parameters()][0][1].shape
        img_sz = (img_sz[0], img_sz[1], 1)
    modules = [module for module in qnet.modules()
               if (not 'Sequential' in str(type(module)) and
                   not 'QAlexNet_FP' in str(type(module)) and
                   not 'QLeNet5_FP' in str(type(module)))# and
    ]

    # Find MADDs
    madds_list = []
    tensordimsprods = []
    for idx, module in enumerate(modules):
        #print(madds_idx, str(type(module)) )
        if 'MaxPool2d' in str(type(module)) or 'QAvgPool2d_FP' in str(type(module)):
            if not is_muppet:
                tensordimsprod = img_sz[0] * img_sz[1] * img_sz[2]
                tensordimsprods.append(tensordimsprod)
            img_sz = pooling2d(modules, module, idx, img_sz, madds_list)
        if 'QConv2d_FP' in str(type(module)):
            if not is_muppet:
                tensordimsprod = module.__dict__['kernel_size'][0] * module.__dict__['kernel_size'][1] * \
                                 module.__dict__['out_channels']
                tensordimsprods.append(tensordimsprod)
            img_sz = conv2d(modules, module, idx, img_sz, madds_list)
        if 'QConv2d_FP' in str(type(module)):
            if not is_muppet:
                tensordimsprod = module.__dict__['kernel_size'][0] * module.__dict__['kernel_size'][1] * \
                                 module.__dict__['out_channels']
                tensordimsprods.append(tensordimsprod)
            img_sz = conv2d(modules, module, idx, img_sz, madds_list)
        if 'QLinear_FP' in str(type(module)):
            if not is_muppet:
                tensordimsprod = img_sz[0] * img_sz[1] * img_sz[2]
                tensordimsprods.append(tensordimsprod)
            img_sz = linear(modules, module, idx, img_sz, madds_list)
        if 'QBatchNorm2d_FP' in str(type(module)):
            if not is_muppet:
                tensordimsprod = img_sz[0] * img_sz[1] * img_sz[2]
                tensordimsprods.append(tensordimsprod)
            img_sz = batchnorm2d(modules, module, idx, img_sz, madds_list)
        if 'QGCNConv_FP' in str(type(module)):
            if not is_muppet:
                tensordimsprod = module.__dict__['in_channels']*module.__dict__['out_channels']
                tensordimsprods.append(tensordimsprod)
            img_sz = gcnconv(modules, module, idx, img_sz, madds_list)
        if 'QGINConv_FP' in str(type(module)):
            if not is_muppet:
                #print(module.__dict__.keys(), flush=True)
                tensordimsprod = module.__dict__['in_channels'] * module.__dict__['out_channels']
                tensordimsprods.append(tensordimsprod)

    model_sizes_q = []
    madd_costs_q = []
    model_sizes_float32 = []
    madd_costs_float32 = []

    # Iterate over logfile and weight MADDs for fwd and bwds pass and MARViN with bitwidth
    grad_accum = float(config.getint('MODEL', 'ACCUMULATION_STEPS'))
    accum_overhead = 0
    avg_iter_non_zeros = []
    inf_costs_q, inf_costs_fl32 = 0, 0
    iso_acc = 1.0
    if not is_muppet:
        for i in range(0, len(metrics['val_acc'])):
            if (metrics['val_acc'][i] >= config.getfloat('MODEL', 'TARGET_ACCURACY')):
                iso_acc = i/len(metrics['val_acc'])
                break

    for i, names in enumerate(metrics["name"]):
        iteration_accum_non_zeros = []
        madds_idx = 0
        end_state = (i == (len(metrics["name"])-1))
        for j, name in enumerate(names):
            if "conv" in name.lower() and not "gnn" in name.lower():
                overhead, non_zeros = 0, 1.0
                if not is_muppet:
                    non_zeros = 1.0 - metrics['sp'][i][j]
                    overhead = marvin_step(tensordimsprods[madds_idx], config, metrics['lb'][i][j],
                                           metrics['res'][i][j], non_zeros, metrics['qwl'][i][j])
                    overhead = 32 * overhead * config.getint('LOGGING', 'INTERVAL') / (metrics['lb'][i][j] * grad_accum)
                    accum_overhead += overhead
                iteration_accum_non_zeros.append(non_zeros)
                madd_costs_q.append(non_zeros * (madds_list[madds_idx] * metrics['qwl'][i][j] + overhead))
                madd_costs_float32.append(madds_list[madds_idx] * 32)
                if end_state:
                    inf_costs_q += non_zeros * madds_list[madds_idx] * metrics['qwl'][i][j]
                    inf_costs_fl32 += madds_list[madds_idx] * 32
                madds_idx += 1
            elif "linear" in name.lower() or "fc" in name.lower():
                overhead, non_zeros = 0, 1.0
                if not is_muppet:
                    non_zeros = 1.0 - metrics['sp'][i][j]
                    overhead = marvin_step(tensordimsprods[madds_idx], config, metrics['lb'][i][j],
                                           metrics['res'][i][j], non_zeros, metrics['qwl'][i][j])
                    overhead = 32 * overhead * config.getint('LOGGING', 'INTERVAL') / (metrics['lb'][i][j] * grad_accum)
                    accum_overhead += overhead
                iteration_accum_non_zeros.append(non_zeros)
                madd_costs_q.append(non_zeros * (madds_list[madds_idx] * metrics['qwl'][i][j] + overhead))
                madd_costs_float32.append(madds_list[madds_idx] * 32)
                if end_state:
                    inf_costs_q += non_zeros * madds_list[madds_idx] * metrics['qwl'][i][j]
                    inf_costs_fl32 += madds_list[madds_idx] * 32
                madds_idx += 1
            elif "pool" in name.lower() or "downsample" in name.lower():
                overhead, non_zeros = 0, 1.0
                if not is_muppet:
                    non_zeros = 1.0 - metrics['sp'][i][j]
                    overhead = marvin_step(tensordimsprods[madds_idx], config, metrics['lb'][i][j],
                                           metrics['res'][i][j], non_zeros, metrics['qwl'][i][j])
                    overhead = 32 * overhead * config.getint('LOGGING', 'INTERVAL') / (metrics['lb'][i][j] * grad_accum)
                    accum_overhead += overhead
                iteration_accum_non_zeros.append(non_zeros)
                madd_costs_q.append(non_zeros * (madds_list[madds_idx] * metrics['qwl'][i][j] + overhead))
                madd_costs_float32.append(madds_list[madds_idx] * 32)
                if end_state:
                    inf_costs_q += non_zeros * madds_list[madds_idx] * metrics['qwl'][i][j]
                    inf_costs_fl32 += madds_list[madds_idx] * 32
                madds_idx += 1
            elif "batchnorm" in name.lower() or "bn" in name.lower() and not 'gnn' in name.lower():
                overhead, non_zeros = 0, 1.0
                if not is_muppet:
                    non_zeros = 1.0 - metrics['sp'][i][j]
                    overhead = marvin_step(tensordimsprods[madds_idx], config, metrics['lb'][i][j],
                                           metrics['res'][i][j], non_zeros, metrics['qwl'][i][j])
                    overhead = 32 * overhead * config.getint('LOGGING', 'INTERVAL') / (metrics['lb'][i][j] * grad_accum)
                    accum_overhead += overhead
                iteration_accum_non_zeros.append(non_zeros)
                madd_costs_q.append(non_zeros * (madds_list[madds_idx] * metrics['qwl'][i][j] + overhead))
                madd_costs_float32.append(madds_list[madds_idx] * 32)
                if end_state:
                    inf_costs_q += non_zeros * madds_list[madds_idx] * metrics['qwl'][i][j]
                    inf_costs_fl32 += madds_list[madds_idx] * 32
                madds_idx += 1

        avg_iter_non_zeros.append(sum(iteration_accum_non_zeros)/len(iteration_accum_non_zeros))
        model_sizes_q.append(np.sum(np.array(metrics['qwl'][i]))*avg_iter_non_zeros[-1])
        model_sizes_float32.append(np.sum(np.array([32]*len(metrics['qwl'][i]))))

    inf_su = inf_costs_fl32/inf_costs_q
    final_sparsity = 1.0-avg_iter_non_zeros[-1]
    avg_sparsity =  1.0-sum(avg_iter_non_zeros)/len(avg_iter_non_zeros)
    iso_acc_epoch = int(config.getint('MODEL', 'MAX_EPOCHS')*iso_acc)+1
    bs = 1.0
    if 'cifar' in config['MODEL']['DATASET']:
        bs = 128.0 / float(config.getint('MODEL', 'BATCH_SIZE'))  # Muppet cifar10/100 batch size = 128
    elif 'imagenet' in config['MODEL']['DATASET']:
        bs = 256.0 / float(config.getint('MODEL', 'BATCH_SIZE'))  # Muppet imagenet batch size = 256
    grad_accum_inv = 1 / grad_accum  # Muppet doesnt accumulate
    epochs = 150.0 / float(config.getint('MODEL', 'MAX_EPOCHS'))  # MuPPET Baseline 150 Epochs

    su_own = 2 * sum(madd_costs_float32) / (sum(madd_costs_q) + sum(madd_costs_float32))
    su_own_iso = 2 * sum(madd_costs_float32) / (sum(madd_costs_q) + sum(madd_costs_float32)) /iso_acc
    mem_own = 1 + sum(model_sizes_q) / sum(model_sizes_float32) * 2 # (3 for MuPPET, see their paper variable r there)
    mod_sz = model_sizes_q[-1] / model_sizes_float32[-1]
    su_other = epochs * (2 * sum(madd_costs_float32) / (sum(madd_costs_q) + sum(madd_costs_float32) * grad_accum_inv * bs))
    mod_sz_other = model_sizes_q[-1] / model_sizes_float32[-1]
    overhead_perc_own = accum_overhead/sum(madd_costs_q)

    data = {'madds_q' : madd_costs_q, 'madds' : madd_costs_float32}
    fname = "{}_perfmodel_madds".format(log.split("/")[-1].split(".")[:-1])
    fname = fname.replace('[', '').replace(']', '').replace("'", '')
    with open("{}.json".format(fname), 'w') as outfile:
        json.dump(data, outfile)
    return su_own, mem_own, mod_sz, su_other, mod_sz_other, overhead_perc_own, inf_su, final_sparsity, avg_sparsity, iso_acc_epoch, su_own_iso


def run_analysitcs(config):
    qlogger = QLogger()
    arch_list = ['ResNet', 'AlexNet', 'LeNet', 'Simple_GNN', 'GCN_Graph', 'GIN_Graph']
    clear_name = lambda arch, arch_list: [a for a in arch_list if a.lower() in arch.lower()][0]
    # clear_name = lambda arch: 'ResNet' if 'resnet' in arch else ('AlexNet' if 'alexnet' in arch else '')
    print(config['MODEL']['ARCHITECTURE'])
    arch_name = clear_name(config['MODEL']['ARCHITECTURE'], arch_list)
    dset_name = config['MODEL']['DATASET'].upper()
    if config['MODEL']['DATASET'] in ['cifar10', 'cifar100', 'imagenet', 'emnist', 'fmnist', 'mnist', 'cora', 'ogbg-molhiv', 'ogbg-molpcba']:
        su_own_mv, mem_own_mv, mod_sz_mv, su_mp_mv, mod_sz_mp_mv = None, None, None, None, None
        inf_su, final_sparsity, avg_sparsity, iso_acc_epoch, su_own_iso = None, None, None, None, None
        su_own_mp, mem_own_mp, mod_sz_mp = None, None, None
        overhead_perc_own = None
        if config['MODEL']['ARCHITECTURE'] == 'qalexnet_fp':
            gen_json([(8, 16), (12, 10), (14, 14), (16, 14), (32, 45)],
                     'qalexnet_fp_MuPPET_cifar10.json')  # MuPPET results from paper
            su_own_mp, mem_own_mp, mod_sz_mp, _, _, _, _, _, _, _, _ = analyse('qalexnet_fp_MuPPET_cifar10.json',
                                                                               config)
            su_own_mv, mem_own_mv, mod_sz_mv, su_mp_mv, mod_sz_mp_mv, overhead_perc_own, inf_su, final_sparsity, avg_sparsity, iso_acc_epoch, su_own_iso = analyse(
                qlogger.get_log_name(), config)
        elif config['MODEL']['ARCHITECTURE'] == 'qresnet_fp':
            gen_json([(8, 22), (12, 22), (14, 10), (16, 15), (32, 45)],
                     'qresnet_fp_MuPPET_cifar10.json')  # MuPPET results from paper
            su_own_mp, mem_own_mp, mod_sz_mp, _, _, _, _, _, _, _, _ = analyse('qalexnet_fp_MuPPET_cifar10.json',
                                                                               config)
            su_own_mv, mem_own_mv, mod_sz_mv, su_mp_mv, mod_sz_mp_mv, overhead_perc_own, inf_su, final_sparsity, avg_sparsity, iso_acc_epoch, su_own_iso = analyse(
                qlogger.get_log_name(), config)
        elif config['MODEL']['ARCHITECTURE'] == 'qlenet_fp':
            # LeNet5 not testet by MuPPET
            su_own_mv, mem_own_mv, mod_sz_mv, su_mp_mv, mod_sz_mp_mv, overhead_perc_own, inf_su, final_sparsity, avg_sparsity, iso_acc_epoch, su_own_iso = analyse(
                qlogger.get_log_name(), config)
        elif config['MODEL']['ARCHITECTURE'] == 'qgcn_graph_fp':
            # LeNet5 not testet by MuPPET
            su_own_mv, mem_own_mv, mod_sz_mv, su_mp_mv, mod_sz_mp_mv, overhead_perc_own, inf_su, final_sparsity, avg_sparsity, iso_acc_epoch, su_own_iso = analyse(
                qlogger.get_log_name(), config)
        elif config['MODEL']['ARCHITECTURE'] == 'qgin_graph_fp':
            # LeNet5 not testet by MuPPET
            su_own_mv, mem_own_mv, mod_sz_mv, su_mp_mv, mod_sz_mp_mv, overhead_perc_own, inf_su, final_sparsity, avg_sparsity, iso_acc_epoch, su_own_iso = analyse(
                qlogger.get_log_name(), config)

        print("MARViN {0} vs. MARViNs {1} float32 baseline, Training Speedup (termination at {2} epochs):".format(
            arch_name, dset_name, config.getint('MODEL', 'MAX_EPOCHS')), su_own_mv)
        print(
            "MARViN {0} vs. MARViNs {1} float32 baseline, Training Speedup (termination at iso accuracy, {2} epochs):".format(
                arch_name, dset_name, iso_acc_epoch), su_own_iso)
        print("MARViN {0} vs. MARViNs {1} float32 baseline, Inference Speedup:".format(arch_name, dset_name), inf_su)
        print("MARViN {0} vs. MARViNs {1} float32 baseline, Final and Average Training Sparsity".format(arch_name,
                                                                                                        dset_name),
              final_sparsity, avg_sparsity)
        print("MARViNs Overhead on {0} {1}".format(arch_name, dset_name), overhead_perc_own)
        print("MARViN {0} vs. MARViNs {1} float32 baseline, Memory Footprint:".format(arch_name, dset_name), mem_own_mv)
        print("MARViN {0} vs. MARViNs {1} float32 baseline, Final Model Size:".format(arch_name, dset_name), mod_sz_mv)
        if "resnet" in config['MODEL']['ARCHITECTURE'].lower() or "alexnet" in config['MODEL']['ARCHITECTURE'].lower():
            print("MARViN {0} vs. MuPPETs {1} float32 baseline, Speedup:".format(arch_name, dset_name), su_mp_mv)
            print("MARViN {0} vs. MuPPETs {1} float32 baseline, Final Model Size:".format(arch_name, dset_name),
                  mod_sz_mp_mv)
            print("MuPPET {0} vs. MuPPETs {1} float32 baseline, Speedup:".format(arch_name, dset_name), su_own_mp)
            print("MuPPETs {0} vs. MuPPETs {1} float32 baseline, Memory Footprint:".format(arch_name, dset_name),
                  mem_own_mp)
            print("MuPPET {0} vs. MuPPETs {1} float32 baseline, Final Model Size:".format(arch_name, dset_name),
                  mod_sz_mp)