# Porting Python code to GPUs - training material

The `getting_started_with_numba_and_cupy.ipynb` Jupyter notebook was created as a training resource as part of an NCEO project investigating the advantages of different tools designed for running code on GPUs. Examples of code were obtained from NCEO scientists and the applications were ported to GPUs on the NEODAAS MAGEO system. Using Numba and CuPy, accelerations of between x700 and x1800 were achieved. The notebook provides a gentle introduction to Numba and CuPy using some simple examples to demonstrate the basic methods. It also shows how the methods might be applied to real life problems and how Numba and CuPy can be used together. Timings are compared for CPU and GPU to demonstrate the benefits of the methods. Recommended resources for further learning are included at the end of the notebook.

This resource may be used by researchers looking for an introduction to Numba and CuPy. It has been designed to run on MAGEOHub. It can be run on any machine with access to a CUDA-enabled NVIDIA GPU but may need some additional setting up outside of MAGEOHub.

MAGEOHub is operated by the NERC Earth Observation Data Acquisition and Analysis Service (NEODAAS) and access is available to researchers eligible for NERC funding. For information on how to access and information on other services NEODAAS provides please see: https://www.neodaas.ac.uk/


## Running the notebook on MAGEOHub

To run the notebook you can log onto MAGEOHub and open a terminal window, then clone this repository:
```
git clone https://github.com/NEODAAS/Porting-Python-code-to-GPUs.git
```
You should now have a folder called 'Porting-Python-code-to-GPUs' and within the folder the notebook 'getting_started_with_numba_and_cupy.ipynb', which you can open by clicking on it. You will also find another notebook in this folder called 'nanpercentile_functions.ipynb'. This notebook is not designed to be run directly, but instead is imported within the main getting started notebook.

You can use the default Python 3 environment to run the notebook. It will take about an hour to work through all the examples. The notebook starts with some simple introductory examples and then moves on to some more advanced examples. You may wish to skip the final example, 'Accelerating Numpy's nanpercentile function', which is more advanced.


[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]<br />This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://i.creativecommons.org/l/by-nc/4.0/88x31.png