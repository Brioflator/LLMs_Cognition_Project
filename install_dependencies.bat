@echo off
echo Installing/Updating Jupyter and Widget Dependencies...
echo.

REM Update pip first
python -m pip install --upgrade pip

REM Install/update jupyter and ipywidgets
python -m pip install --upgrade jupyterlab notebook ipywidgets

REM Enable ipywidgets extension
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter labextension install @jupyter-widgets/jupyterlab-manager

REM Update h5py to match HDF5 version
python -m pip install --upgrade h5py

REM Install remaining requirements
python -m pip install -r requirements.txt

echo.
echo Installation complete!
echo Please restart your Jupyter kernel for changes to take effect.
pause

