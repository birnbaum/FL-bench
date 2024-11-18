#!/bin/bash

python main.py --config-name cifar10 method=floco +floco.num_endpoints=3 +floco.tau=50
python main.py --config-name cifar10 method=floco +floco.num_endpoints=3 +floco.tau=101
python main.py --config-name cifar10 method=floco +floco.num_endpoints=3 +floco.tau=101 +floco.pers_epoch=1
python main.py --config-name cifar10 method=floco +floco.num_endpoints=3 +floco.tau=101 +floco.pers_epoch=5
python main.py --config-name cifar10 method=fedavg
python main.py --config-name cifar10 method=ditto +ditto.pers_epoch=1
python main.py --config-name cifar10 method=ditto +ditto.pers_epoch=5