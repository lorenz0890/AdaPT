import json
import sys
from utils.parser import create_arg_parser
import configparser


class QLogger:
    def __init__(self):
        arg_parser = create_arg_parser()
        args = arg_parser.parse_args(sys.argv[1:])
        config = configparser.ConfigParser()
        config.read(args.cfg)
        self.cfg = config
        self.val_acc = []
        self.lr = []
        self.val_loss = []
        self.qwl = []
        self.qfl = []
        self.perm = []
        self.res = []
        self.lb = []
        self.name = []
        self.sp = []
        self.metrics = {'val_acc': self.val_acc, 'lr': self.lr,
                        'val_loss': self.val_loss, 'qwl': self.qwl, 'qfl': self.qfl, 'perm': self.perm, 'res': self.res,
                        'lb': self.lb, 'name': self.name, 'sp': self.sp}

    def get_log_name(self):
        rb = 'rb0'
        if 'QUANTIZATION' in self.cfg.keys():
            rb = 'rb{}'.format(self.cfg['QUANTIZATION']['RESULTS_BUFFER'])
        name = "{}_{}_{}_{}.json".format(self.cfg['MODEL']['ARCHITECTURE'], rb, self.cfg['OPTIMIZER']['NAME'],
                                         self.cfg['MODEL']['DATASET'])
        return name

    def qlog(self, key, value):
        if key in self.metrics.keys():
            self.metrics[key].append(value)
        elif key == 'qmap':
            bias = []
            for i, name in enumerate(value["name"]):
                if "bias" in name:
                    bias.append(i)
            for subkey in value.keys():
                if subkey in self.metrics:
                    values = [metric_value for i, metric_value in enumerate(value[subkey]) if i not in bias]
                    #values = values[::2]
                    self.metrics[subkey].append(values)

        return

    def save(self):
        with open("output/logs/{}".format(self.get_log_name()), 'w') as f:
            f.write(json.dumps(self.metrics))
        return