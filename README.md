# SKFSR: SKip-connected Feature-reuse Super-Resolution
A deep-learning model combines guidance of physics and data to learn observation functions accurately and efficiently, which conducts a linearization and preidiction of high-dimensional nonlinear dynamical systems.

## Introduction
- This repo provides code for the paper ["Fractal-constrained Deep Learning for Super-resolution of Turbulence with Zero or Few Label Data"](under review) by Jiaxin Wu, Min Luo, Boo Cheong Khoo and Pengzhi Lin.
- This study proposes a deep-learning model that constrained by fractal physics and utilizes zero / few data, for the high-resolution reconstruction of turbulent flows.

## Structure
    │  LICENSE
    │  README.md
    ├─ Fractal_script              
    │  │  SKFSR_main.py  (main program)
    │  └─ ML_SKFSR_TorchModel.py  (main network)
    ├─ data         
    └─ utils
        │  loadSnapshotsObjects(...).py  (visualization tools)
        │  ML_plot.py  (visualization utilities)
        │  generator.py  (tool for generating sierpinski foam)
        │  fdgpu.py  (tensorized differential computation for fractal dimensions)
        └─ ML_utils.py  (general utilities)

## Common usage
- Runing SKFSR_main.py with arguments 
- e.g., python SKFSR_main.py -m Super_SKFSR_DB_unsp_single -c Channel -b 32 -s 4 -e 1000 --load False
- Some basic denotes have been made in SKFSR_main.py

## General hyperparameters
- "case" for selecting numerical cases
- "method" for defining numerical approaches
template: (1) binarization scheme + (2) training strategy + (3) snapshot quantity, 
(1) binarization scheme: TB (truncated binarinzation) / DB (differential binarization)
(2) training strategy: unsp / semi / sup
(3) snapshot quantity: single (1) / few (30) / standard (120)

e.g., DB_unsp_single

       
## Datasets
Cases C1a-e are generated by IFS are studied including: 
* C1a: 
    Please use "np.eye" function instead
* C1b: 
    koch126_512_9.npy
* C1c:
    koch146_512_9.npy
* C1d:
    koch15_512_7.npy
* C1e:
    sier_512_7.npy

Cases C2a&b are downloaded from Johns Hopkins Turbulence Databases (https://turbulence.pha.jhu.edu/) including: 

* C2a: channel turbulence
    channel_u_512X128p_4to24s_dt0.13.npy
* C2b: isotropic turbulence
    iso_velocity_pressure_256x256_2to8.npy

Please download turbulence data from [] and put them into the "data" folder

## Requirements
The model is built in Python environment using TensorFlow v2 backend, basically using packages of:
* Python 3.x  
* torch 
* sklearn
* numpy
