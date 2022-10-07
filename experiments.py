import configparser
import sys

import pytorch_lightning as pl

from utils.analytics import run_analysitcs
from utils.cv import CV
from utils.factory import model_factory
from utils.parser import create_arg_parser

arg_parser = create_arg_parser()
args = arg_parser.parse_args(sys.argv[1:])
print("Command line arguments:")
print(args)

config = configparser.ConfigParser()
config.read(args.cfg)
sections = config.sections()
for section in sections:
    print(section)
    print(dict(config[section]))

# Initialize trainer and CV
trainer = pl.Trainer(gpus=config.getint('SYSTEM', 'GPUS'), max_epochs=config.getint('MODEL', 'MAX_EPOCHS'),
                     progress_bar_refresh_rate=20)
cv = CV(trainer=trainer, config=config, n_splits=config.getint('MODEL', 'KCV'), stratify=config.getboolean('MODEL', 'STRATIFY'))

# Run K-fold cross-validation experiments:

model, model_args, image_data = model_factory(config)

best_model, best_acc = cv.fit(model, image_data)

print('best model test accuracy', best_acc)

run_analysitcs(config)

exit(0)
