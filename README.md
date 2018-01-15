# project g<sub>A</sub>

This project is for the analysis code and data files for our calculation of g<sub>A</sub> with MDWF on HISQ, and contains the following:
* `setup.py`: Setup file for Python environment
* `ga_workbook.ipynb`: Jupyter notebook for chiral-continuum extrapolation analysis used in the final analysis
* `data`: Directory of data
  * `github_ga_v1.csv`: Bootstrapped correlation function analysis results in csv format
    * Correlator data is made easily accessible from Jupyter with `pandas` and summarized in a dataframe
  * `hisq_params.csv`: a/w<sub>0</sub> and α<sub>s</sub> for HISQ ensembles used for this work in csv format
    * HISQ parameters are displayed in `pandas` dataframe
* `callat_ga_lib`: Library for extrapolation
  * correlator data formatting for `lsqfit`
  * fit function definitions
  * systematic error breakdown definitions
  * matplotlib routines

# Setup for Python environment
## Download Anaconda and install
Download [Anaconda](https://www.continuum.io/downloads) and follow installation instructions.

## Create Python environment with Anaconda
```bash
conda create --name callat_ga python=3 anaconda
source activate callat_ga
python setup.py install
```

Key libraries from [gplepage GitHub](https://github.com/gplepage).
* `gvar` version 8.3.2
* `lsqfit` version 9.1.3

Exit conda environment with
```bash
source deactivate
```

## Open Jupyter notebook
```bash
jupyter notebook ga_workbook.ipynb
```

<figure style="float:right">
    <img
    src="./data/callat_logo.png"
    width="100"
    alt="CalLat logo"
    /img>
</figure>


## Tested with the following Python Setup
python version: 3.6.1 |Anaconda 4.4.0 (x86_64)| (default, May 11 2017, 13:04:09)
[GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)]
pandas version: 0.20.1
numpy  version: 1.12.1
scipy  version: 0.19.0
mpl    version: 2.0.2
lsqfit version: 9.1.3
gvar   version: 8.3.2

and

python version: 2.7.13 (default, Jul 29 2017, 11:08:07)
[GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.42)]
pandas version: 0.20.3
numpy  version: 1.13.1
scipy  version: 0.19.0
mpl    version: 2.0.2
lsqfit version: 9.1.3
gvar   version: 8.2.2
