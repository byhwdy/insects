import sys
sys.path.append('/paddle/insects')

import test

import yaml


with open('configs/test.yml', 'r') as f:
	person = yaml.load(f, yaml.Loader)

person.run()

print(yaml)