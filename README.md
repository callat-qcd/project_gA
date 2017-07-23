# project_gA

This project is for the analysis code and data files for our calculation of gA with MDWF on HISQ

# Setup for Python environment
## Download Anaconda and install 
Download [Anaconda](https://www.continuum.io/downloads) and follow installation instructions.

## Create Python environment with Anaconda
```bash
conda create --name callat_ga python=3 conda
source activate callat_ga
pip install gvar
pip install lsqfit```

Key libraries from [gplepage GitHub](https://github.com/gplepage).
* `gvar` version 8.0
* `lsqfit` version 8.1

Exit conda environment with
```bash
source deactivate
```

## Open Jupyter notebook
```bash
jupyter notebook ga_workbook.ipynb
```
