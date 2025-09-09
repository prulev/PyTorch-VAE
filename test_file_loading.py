import os
import yaml
import argparse

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

root = config['preprocess_params']['raw_root']
folder = config['preprocess_params']['raw_folder']
filename = config['preprocess_params']['file_name']
path = os.path.join(root, folder, filename)
print(path)