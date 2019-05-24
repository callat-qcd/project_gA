<figure style="float:right">
    <img
    src="./data/callat_logo.png"
    width="100"
    alt="CalLat logo"
    align="right"
    /img>
</figure>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1241374.svg)](https://doi.org/10.5281/zenodo.1241374)

# project g<sub>A</sub>

This project performs the chiral, continuum, infinite volume extrapolation of the g<sub>A</sub> values computed with MDWF on HISQ lattice action, as described in [Nature **558**, 91–94 (2018)](https://www.nature.com/articles/s41586-018-0161-8) or [arXiv:1805.12130](https://arxiv.org/abs/1805.12130).  To perform the extrapolation, we have created a Jupyter notebook and an accompanying Python library:
* `ga_workbook.ipynb`: Jupyter notebook for chiral-continuum extrapolation analysis used in the final analysis
* `callat_ga_lib`: Library for extrapolation
  * correlator data formatting for `lsqfit`
  * fit function definitions
  * systematic error breakdown definitions
  * matplotlib routines
The bootstrap results of our correlation function analysis are contained in the `data` folder along with other input parameters from the HISQ ensembles needed in the analysis:
* `data`: Directory of data
  * `github_ga_v2.csv`: Bootstrapped correlation function analysis results in csv format
    * Correlator data is made easily accessible from Jupyter with `pandas` and summarized in a dataframe
  * `hisq_params.csv`: a/w<sub>0</sub> and α<sub>s</sub> for HISQ ensembles used for this work in csv format
    * HISQ parameters are displayed in `pandas` dataframe

In addition, the raw correlation functions computed for this project are included in `correlation_functions`:
* `correlation_functions`: Directory of data
  * `callat_gA.h5`
We provide a sample correlation function fitter that performs the same analysis performed for our project in `sample_corr_fit`.  This sample fitter uses [`iminuit`](https://iminuit.readthedocs.io/en/latest/) v1.1.1 (our main analysis was performed with `lsqfit`):
* `sample_corr_fit`
  * `fh_fit.py`: main library for performing fit
  * `ga_sample_corr_fitter.ipynb`: Jupyter notebook that uses the library
  * `fit_params.py`: an input file generated through our Bayes constrained fit to pre-condition the frequentist least squares minimization.


# Setup for Python environment
## Download Anaconda and install
Download [Anaconda](https://www.continuum.io/downloads) and follow installation instructions.

## Create Python environment with Anaconda
```bash
conda create --name pyqcd3 python=3 anaconda
source activate pyqcd3
```

Key libraries from [gplepage GitHub](https://github.com/gplepage).
* `gvar` version 8.3.2 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.816304.svg)](https://doi.org/10.5281/zenodo.816304)
* `lsqfit` version 9.1.3 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.494988.svg)](https://doi.org/10.5281/zenodo.494988)

Exit conda environment with
```bash
source deactivate
```

## Open Jupyter notebook
```bash
jupyter notebook ga_workbook.ipynb
```

## ga_workbook.ipynb Tested with the following Python Setup
```
python version: 3.6.1 |Anaconda 4.4.0 (x86_64)| (default, May 11 2017, 13:04:09)
[GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)]
pandas version: 0.20.1
numpy  version: 1.12.1
scipy  version: 0.19.0
mpl    version: 2.0.2
lsqfit version: 9.1.3
gvar   version: 8.3.2
```

and

```
python version: 2.7.13 (default, Jul 29 2017, 11:08:07)
[GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.42)]
pandas version: 0.20.3
numpy  version: 1.13.1
scipy  version: 0.19.0
mpl    version: 2.0.2
lsqfit version: 9.1.3
gvar   version: 8.2.2
```

## ga_sample_corr_fitter.ipynb Tested with the following Python Setup
```
python version: 3.6.1 |Anaconda 4.4.0 (x86_64)| (default, May 11 2017, 13:04:09)
[GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)]
numpy  version: 1.12.1
scipy  version: 0.19.0
mpl    version: 2.0.2
iminuit version: 1.1.1
```

```
python  version: 2.7.14 (default, Sep 25 2017, 09:53:22)
[GCC 4.2.1 Compatible Apple LLVM 9.0.0 (clang-900.0.37)]
numpy   version: 1.14.2
scipy   version: 1.0.1
mpl    version: 2.0.2
iminuit version: 1.1.1
```



## Copyright Notice

project_gA Copyright (c) 2018, The Regents of the University of California (UC), through Lawrence Berkeley National Laboratory, and the UC Berkeley campus (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Innovation & Partnerships Office at  IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit other to do so.

| | | |
|:---:|:---:|:---:|
| [<img src='./data/incite_logo.png' width='275'/>](http://www.doeleadershipcomputing.org/)  | [<img src='./data/olcf_logo.png' width='275'/>](https://www.olcf.ornl.gov/) | [<img src='./data/llnl_logo.png' width='275' />](https://hpc.llnl.gov/) |

| | |
|:---:|:---:|
| [<img src='./data/scidac_logo.png' width='416.5' />](http://www.scidac.gov/) | [<img src='./data/doe_oos_highres.jpg' width='416.5'/>](https://science.energy.gov) |
