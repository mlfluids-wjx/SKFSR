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
Four numerical cases with discrete/continuous spectra on Eulerian/Lagrangrian descriptions are studied including: 
* fixed-point attractor 

    generated using ODEPACK from the open-source Python package SciPy, (attractor500x32x32_10s.npy)
* Duffing oscillator 

    generated using ODEPACK from the open-source Python package SciPy, (duffing600x32x32_30s.npy)
* fluid flow past a cylinder 

    obtained from Kutz, et al (http://dmdbook.com/), interpolated into 192×96 grids (Cylinder_interpolate192x96y_order_C.npy)
* channel turbulence

    downloaded from Johns Hopkins Turbulence Databases (https://turbulence.pha.jhu.edu/), of 256×64 grids (channel_u_256X64p_4to24s_dt0.13.npy)

## Requirements
The model is built in Python environment using TensorFlow v2 backend, using packages of:
* Python 3.x  

* torch 

* sklearn

* numpy
