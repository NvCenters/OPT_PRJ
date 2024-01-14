#!/bin/bash

env_name="OPT"
conda create --name $env_name
source activate $env_name
git clone https://github.com/CERN/TIGRE.git
cd TIGRE/Python/  
pip install -r requirements.txt --user  
pip install . --user