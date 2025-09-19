# jz_visualizations
#### This repository features the following files `plotly_vis.py` and `vispy_vis.py`, in which exist utility functions for plotting using Plotly and VisPy, respectively.

- #### The notebook file `examples/plotly_vis_test_usage.ipynb` highlights some example use cases for using the `agg_plot()` and `correlation_plot()` functions.
- #### The notebook file `examples/vispy_vis_test_usage.ipynb` highlights some example use cases for using the Vis Class and associated methods for plotting images, lines, and scatterplots.
- #### These are by no means exhaustive, but just exemplify some of the ways that the tools can be flexibly used.

#### The Plotly code relies on the use of the following packages,<br>and this code was developed using the following versions:
- `python = 3.12`
- `plotly = 6.0.1`
- `numpy = 2.2.4`
- `pandas = 2.2.3`
- `scipy = 1.15.2`
- `statsmodels = 0.14.4`
- `matplotlib = 3.10.1`

#### The VisPy code relies on and was developed using:
- `vispy = 0.15.2`
- `pyqt = 5.15.11`


# _Installation instructions_

### Directly from GitHub (*non-editable version*)
***
1. Open terminal and either activate an existing environment (`source activate myenv`) or create a new one (`conda create -n jzvis python=3.12`) and then activate it.
- Optional: If creating a new env, install all binary packages using conda _before pip installing the package_. If the dependencies that this relies on are not present, they will automatically be downloaded with pip.
- Optional 2b: If you are planning to use jupyter notebooks, make sure to install that now too with `conda install -c conda-forge jupyter`.
2. Install package to env with `pip install git+https://github.com/joezaki/jz_visualizations.git`.
3. Verify installation with `python -c "from jz_visualizations.vispy_vis import Vis; print('Vis imported successfully.')"`.

### From local cloned copy (*editable version*)
***
1. Open terminal and navigate to the folder where you would like jz_visualizations to live (`cd /path/to/folder`).
2. Clone the repository into this folder with `git clone https://https://github.com/joezaki/jz_visualizations.git`.
3. Navigate into `jz_visualizations` with `cd jz_visualizations`.
4. Activate an existing environment (`source activate myenv`) or create a new one (`conda create -n jzvis`) and then activate it.
- Optional: If creating a new env, install all binary packages using conda _before pip installing the package_.
- Optional 2a: If creating a new env, you can use the provided `environment.yaml` file here, with `conda env create -f environment.yaml -n jzvis`.
- Optional 2b: If you are planning to use jupyter notebooks, make sure to install that now too with `conda install -c conda-forge jupyter`.
5. Install package to env with `pip install -e .`. This ensures that any changes that are made to the repo will become immediately available.
6. Verify installation with: `python - <<'PY'`, then next line `import jz_visualizations`, then `print("package imported successfully")`, then `PY`.<br>If the package imported successfully, the print statement will be printed.