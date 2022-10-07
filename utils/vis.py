import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.parser import create_arg_parser
import configparser


def shortcuts(name):
    shortcut = 'UNKOWN'
    if "conv" in name.lower(): shortcut = "C"
    elif "linear" in name.lower() or "fc" in name.lower(): shortcut = "L"
    elif "downsample" in name.lower(): shortcut = "D"
    elif "bn" in name.lower(): shortcut = "BN"
    elif "embedding" in name.lower(): shortcut = "EMB"
    elif "eps" in name.lower(): shortcut = "EPS"
    elif "mlp" in name.lower(): shortcut = "MLP"
    return shortcut

def visualize_log(log_name):

    sns.set(rc={'figure.figsize': (11.7, 8.27)})

    results = None
    with open("output/logs/" + log_name, 'r') as jf:
        results = json.load(jf)

    metrics = ["qwl", "lb", "qfl", "res", "sp"]

    new_metrics = {}
    temp_metrics = {}
    if "graph" in log_name:
        kw = 'conv'
        if "gin" in log_name:
            kw = "mlp"

        for metric in metrics:
            new_metrics[metric] = []
            temp_metrics[metric] = np.array(results[metric]).T
        names2 = np.array(results['name']).T
        offset = 0
        for i in range(0, names2.shape[0]):
            if 'conv' in names2[i][0]: offset+=1
        new_names = []
        for i in range(0, names2.shape[0]):
            if 'conv' in names2[i][0] and i + offset < names2.shape[0]:
                new_names.append(names2[i])
                new_names.append(names2[i+offset])
                for metric in metrics:
                    new_metrics[metric].append(temp_metrics[metric][i])
                    new_metrics[metric].append(temp_metrics[metric][i+offset])
            else:
                new_names.append(names2[i])
                for metric in metrics:
                    new_metrics[metric].append(temp_metrics[metric][i])
            if len(new_names) >= names2.shape[0]:
                break

        results["name"] = np.array(new_names).T
        for metric in metrics:
            results[metric] = np.array(new_metrics[metric]).T

    metric_results = []
    names = [shortcuts(name) for name in results["name"][-1]]

    for metric in metrics:

        metric_results = np.array(results[metric]).T

        ax = plt.subplot()
        cm = ax.pcolormesh(metric_results, cmap='coolwarm')
        y_pos = range(0, len(names))
        y_pos = [i + 0.5 for i in y_pos]
        plt.yticks(y_pos, names, va='center')
        plt.xlabel("iteration")
        plt.ylabel("layer")
        plt.title(log_name.split(".")[0].replace('_', ' '), loc='center')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="20%", pad=0.3)

        plt.colorbar(cm, cax=cax)
        plt.xlabel(metric)

        plt.savefig("output/vis/" + '{}_{}.png'.format(log_name, metric))
        plt.close()

    visualize_acc_lrs(log_name, results["val_acc"], results["lr"], results["val_loss"])
    return


def visualize_acc_lrs(log, accs, lrs, loss):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.tight_layout()

    ax1.plot(accs)
    # ax1.set_title(log.replace('_', ' '))
    ax1.set_xlabel("# iterations")
    ax1.set_ylabel("validation accuracy[%]")
    ax1.set_ylim([0, 1])
    # ax1.margins(0.2)

    ax2.plot(loss)
    # ax3.set_title(log.replace('_', ' '))
    ax2.set_xlabel("# iterations")
    ax2.set_ylabel("validation loss")
    # ax3.margins(0.2)

    ax3.plot(lrs)
    # ax2.set_title(log.replace('_', ' '))
    ax3.set_xlabel("# iterations")
    ax3.set_ylabel("learning rate")
    # ax2.margins(0.2)

    plt.savefig("output/vis/" + '{}_{}.png'.format(log, "lr_acc_loss"), bbox_inches='tight')
    plt.close()
