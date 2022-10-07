import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Marvin Framework")
    parser.add_argument('--cfg', type=str, help='configuration file (path)')
    return parser
