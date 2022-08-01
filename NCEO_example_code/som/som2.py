import random
import sys
import xarray as xr
import logging
import numpy as np
from numba import jit

class Progress(object):

    def __init__(self,label,silent=False):
        self.label = label
        self.last_progress_frac = None
        self.silent = silent

    def report(self,msg,progress_frac):
        if self.silent:
            return
        if self.last_progress_frac is None or (progress_frac - self.last_progress_frac) >= 0.01:
            self.last_progress_frac = progress_frac
            i = int(100*progress_frac)
            if i > 100:
                i = 100
            si = i // 2
            sys.stdout.write("\r%s %s %-05s %s" % (self.label,msg,str(i)+"%","#"*si))
            sys.stdout.flush()

    def complete(self,msg):
        if self.silent:
            return
        sys.stdout.write("\n%s %s\n" % (self.label,msg))
        sys.stdout.flush()

@jit(nopython=True)
def find_bmu(values,weights):
    inarr = np.expand_dims(values, axis=1)
    sqdiffs = (weights - inarr) ** 2
    sumsdiffs = np.sum(sqdiffs, axis=0)
    return np.argmin(sumsdiffs)

@jit(nopython=True)
def iterate(nr_instances,instances,instance_mask,weights,gridwidth,gridheight,neighbour_limit,learn_rate):
    for i in range(nr_instances):
        if instance_mask[i]:
            winner = find_bmu(instances[i,:],weights)
            wx = winner % gridwidth
            wy = winner // gridwidth
            update_network(weights, gridwidth, gridheight, wx, wy, instances[i, :], neighbour_limit,
                           learn_rate)

@jit(nopython=True)
def update_network(weights, gridwidth, gridheight, wx, wy, values, neighbour_limit, learn_rate):
    for x in range(max(0, wx - neighbour_limit), min(gridwidth, wx + neighbour_limit + 1)):
        for y in range(max(0, wy - neighbour_limit), min(gridheight, wy + neighbour_limit + 1)):
            index = x + (y * gridwidth)
            weights[:, index] -= learn_rate * (weights[:, index] - values)

@jit(nopython=True)
def compute_scores(nr_instances,instance_mask,instances,weights,gridwidth):
    scores = np.zeros(shape=(nr_instances, 2))
    for i in range(nr_instances):
        if instance_mask[i]:
            bmu = find_bmu(instances[i, :],weights)
            wx = bmu % gridwidth
            wy = bmu // gridwidth
        else:
            wx = np.nan
            wy = np.nan
        scores[i, :] = np.array([wx,wy])

    return scores

class SelfOrganisingMap(object):

    """
    Train Self Organising Map (SOM) with cells arranged in a 2-dimensional rectangular layout

    Parameters
    ----------
    iters : int
        the number of training iterations to use when training the SOM
    gridwidth : int
        number of cells across the grid
    gridheight : int
        number of cells down the grid
    initial_neighbourhood : int
        the initial neighbourhood size

    Keyword Parameters
    ------------------
    verbose : bool
        whether to print progress messages
    seed : int
        random seed - set to produce repeatable results
    """

    def __init__(self, gridwidth, gridheight, iters, initial_neighbourhood, verbose=False, seed=None):
        self.gridheight = gridheight
        self.gridwidth = gridwidth
        self.iters = iters
        self.initial_neighbourhood = initial_neighbourhood
        self.verbose = verbose
        self.rng = random.Random()
        if seed:
            self.rng.seed(seed)
        self.learn_rate_initial = 0.5
        self.learn_rate_final = 0.05

    def get_weights(self,outputIndex):
        return self.weights[:,outputIndex].tolist()

    def fit_transform(self, instances):
        self.neighbour_limit = 0
        self.nr_inputs = instances.shape[1]
        self.nr_instances = instances.shape[0]
        self.instance_mask = ~np.any(np.isnan(instances), axis=1)

        self.nr_outputs = self.gridwidth * self.gridheight
        self.nr_weights = self.nr_outputs * self.nr_inputs

        self.weights = np.zeros((self.nr_inputs, self.nr_outputs))
        for row in range(0, self.nr_inputs):
            for col in range(0, self.nr_outputs):
                self.weights[row, col] = self.rng.random()

        p = Progress("SOM",silent=not self.verbose)
        progress_frac = 0.0
        p.report("Starting", progress_frac)
        iteration = 0
        while iteration < self.iters:
            learn_rate = (1.0 - float(iteration) / float(self.iters)) \
                         * (self.learn_rate_initial - self.learn_rate_final) + self.learn_rate_final
            neighbour_limit = self.initial_neighbourhood - int(
                (float(iteration) / float((self.iters + 1))) * self.initial_neighbourhood)
            logging.debug("iter=%d (of %d) / learning-rate=%f / neighbourhood=%d"%(iteration, self.iters,
                                                                                   learn_rate,
                                                                                   neighbour_limit))

            iterate(self.nr_instances,instances,self.instance_mask,self.weights,self.gridwidth,self.gridheight,neighbour_limit,learn_rate)

            iteration += 1
            progress_frac = iteration/self.iters
            p.report("Training neighbourhood=%d"%(neighbour_limit), progress_frac)

        p.complete("SOM Training Complete")

        return compute_scores(self.nr_instances,self.instance_mask,instances,self.weights,self.gridwidth)

    def coords(self, output):
        return (output % self.gridwidth, output // self.gridwidth)

    def get_output(self, x, y):
        return x + (y*self.gridwidth)


if __name__ == '__main__':
    # SOM training parameters
    # we would like to be able to run gridsize=100, iters=100
    gridsize = 8
    gridheight = 8
    iters = 10

    initial_neighbourhood = min(2,int(gridsize/3))
    da = xr.open_dataset("sla_c3s_clim.nc")["sla_c3s"] # sea level anomalies averaged by month-of-year, lat and lon cell

    stack_dims = ("lat","lon")
    stack_sizes = (da.shape[1],da.shape[2])

    # each (lat,lon) position becomes an independent case
    # flatten lat and lon dimensions and transpose to arrange by (ncases, time)
    # where ncases = nlat*nlon
    instances = da.stack(case=stack_dims).transpose("case", "month").values

    # run SOM to reduce time dimension from 12 to 2
    s = SelfOrganisingMap(gridsize, gridsize, iters, initial_neighbourhood, seed=1, verbose=True)
    import time
    start_time = time.time()
    scores = s.fit_transform(instances)
    end_time = time.time()
    print("Elapsed time: %d seconds" % (int(end_time-start_time)))

    # restore lat/lon dimensions and output
    a = scores.reshape(stack_sizes + (2,))
    new_dims = stack_dims + ("som_axis",)
    output = xr.DataArray(data=a, dims=new_dims, name="monthly_sla_som")
    output.to_netcdf("som.nc")