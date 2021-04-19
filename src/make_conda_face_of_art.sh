#!/bin/zsh
conda create -n foa_env python=3.5 anaconda
source activate foa_env
conda install -c menpo opencv
conda install -c menpo menpo
conda install -c menpo menpofit
pip3 install tensorflow-gpu
