import random
import sys
import xarray as xr
import logging
import numpy as np
import numpy.random


class Progress(object):

    def __init__(self, label, silent=False):
        self.label = label
        self.last_progress_frac = None
        self.silent = silent

    def report(self, msg, progress_frac):
        if self.silent:
            return
        if self.last_progress_frac is None or (progress_frac - self.last_progress_frac) >= 0.01:
            self.last_progress_frac = progress_frac
            i = int(100 * progress_frac)
            if i > 100:
                i = 100
            si = i // 2
            sys.stdout.write("\r%s %s %-05s %s" % (self.label, msg, str(i) + "%", "#" * si))
            sys.stdout.flush()

    def complete(self, msg):
        if self.silent:
            return
        sys.stdout.write("\n%s %s\n" % (self.label, msg))
        sys.stdout.flush()


def find_bmu(instances, weights):
    sqdiffs = (instances[:, :, None] - np.transpose(weights)) ** 2
    sumsqdiffs = sqdiffs.sum(axis=1)
    return sumsqdiffs.argmin(axis=1)


def train_batch(instances, weights, learn_rate, neighbourhood_lookup):
    # winners(#instances) holds the index of the closest weight for each instance
    winners = find_bmu(instances, weights)
    # now find the neighbours of each winner that are also activated by each instance
    # nhoods(#activations,2) holds the instance index and the weight index for each activation
    nwinners = neighbourhood_lookup[winners, :]
    nhoods = np.argwhere(nwinners)

    # get the indices
    weight_indices = nhoods[:, 1]
    instance_indices = nhoods[:, 0]
    fractions = nwinners[instance_indices, weight_indices]
    # print(fractions.shape,np.min(fractions),np.max(fractions))

    # get the updates
    updates = -learn_rate * fractions[:, None] * (weights[weight_indices, :] - instances[instance_indices])

    # aggregate the updates for each weight
    numerator = np.zeros(shape=weights.shape)
    np.add.at(numerator, weight_indices, updates)
    denominator = np.zeros(shape=weights.shape[:1])[:, None]
    np.add.at(denominator, weight_indices, 1)
    denominator = np.where(numerator == 0, 1, denominator)  # fix annoying divide by zero warning
    weight_updates = numerator / denominator

    # update the weights
    weights += weight_updates


def compute_scores(instances, weights, gridwidth, minibatch_size):
    index = 0
    nr_instances = instances.shape[0]
    batch_size = nr_instances if not minibatch_size else minibatch_size
    bmus = np.zeros(shape=(nr_instances,), dtype=int)
    while index < nr_instances:
        last_index = min(index + batch_size, nr_instances)
        bmus[index:last_index] = find_bmu(instances[index:last_index], weights)
        index += batch_size
    scores = np.vstack([bmus % gridwidth, bmus // gridwidth])
    return np.transpose(scores)


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

    def __init__(self, gridwidth, gridheight, iters, initial_neighbourhood=None, verbose=False, seed=None,
                 minibatch_size=None):
        self.gridheight = gridheight
        self.gridwidth = gridwidth
        self.nr_outputs = self.gridwidth * self.gridheight
        self.iters = iters
        self.minibatch_size = minibatch_size

        self.initial_neighbourhood = initial_neighbourhood if initial_neighbourhood else int(gridsize / 3)
        self.verbose = verbose
        self.seed = seed
        self.rng = random.Random()
        if seed:
            self.rng.seed(seed)

        self.learn_rate_initial = 0.01
        self.learn_rate_final = 0.001
        self.neighbourhood_lookup = np.zeros(shape=(self.initial_neighbourhood + 1, self.nr_outputs, self.nr_outputs))

        # for each neighbourhood size 0,1,...initial_neighbourhood
        # build a lookup table where the fraction at neighbourhood_lookup[n,o1,o2]
        # indicates if (and how much) weight at index o2 is a neighbour of the weight at index o1 in neighbourhood size n
        # use 1 and 0 for a binary mask, or between -1.0 and 1.0 for a varying mask
        for neighbourhood in range(0, self.initial_neighbourhood + 1):
            nsq = neighbourhood ** 2
            for i in range(self.nr_outputs):
                ix, iy = self.coords(i)
                for j in range(self.nr_outputs):
                    jx, jy = self.coords(j)
                    sqdist = (jx - ix) ** 2 + (jy - iy) ** 2
                    self.neighbourhood_lookup[neighbourhood, i, j] = 1 if sqdist <= nsq else 0

    def get_weights(self, outputIndex):
        return self.weights[:, outputIndex].tolist()

    def fit_transform(self, original_instances):

        # mask out instances containing NaNs and remove them
        instance_mask = ~np.any(np.isnan(original_instances), axis=1)
        nr_original_instances = original_instances.shape[0]
        valid_instances = original_instances[instance_mask, :]

        # randomly re-shuffle the remaining instances.
        # TODO consider reshuffling after every iteration
        instances = np.copy(valid_instances)
        rng = np.random.default_rng(seed=self.seed)
        rng.shuffle(instances)

        nr_inputs = instances.shape[1]
        nr_instances = instances.shape[0]

        weights = np.zeros((self.nr_outputs, nr_inputs))
        for output_idx in range(0, self.nr_outputs):
            weights[output_idx, :] = instances[self.rng.choice(range(0, nr_instances)), :]

        p = Progress("SOM", silent=not self.verbose)
        progress_frac = 0.0
        p.report("Starting", progress_frac)

        for iteration in range(self.iters):
            # reduce the learning rate and neighbourhood size linearly as training progresses
            learn_rate = self.learn_rate_initial - (self.learn_rate_initial - self.learn_rate_final) * (
                        (iteration + 1) / self.iters)
            neighbour_limit = round(self.initial_neighbourhood * (1 - (iteration + 1) / self.iters))
            neighbourhood_mask = self.neighbourhood_lookup[neighbour_limit, :, :]
            batch_size = nr_instances if not minibatch_size else minibatch_size

            index = 0
            while index < nr_instances:
                last_index = min(index + batch_size, nr_instances)
                train_batch(instances[index:last_index, :], weights, learn_rate, neighbourhood_mask)
                index += batch_size

            progress_frac = iteration / self.iters
            p.report("Training neighbourhood=%d" % (neighbour_limit), progress_frac)

        p.complete("SOM Training Complete")

        # compute final scores from the trained weights
        valid_scores = compute_scores(valid_instances, weights, self.gridwidth, self.minibatch_size)

        # restore the results into the same order as the input array
        scores = np.zeros(shape=(nr_original_instances, 2))
        scores[:, :] = np.nan
        scores[instance_mask, :] = valid_scores
        return scores

    def coords(self, output):
        return (output % self.gridwidth, output // self.gridwidth)

    def get_output(self, x, y):
        return x + (y * self.gridwidth)


if __name__ == '__main__':
    # SOM training parameters
    gridsize = 16
    gridheight = 16
    iters = 10
    minibatch_size = 1000  # the max number of instances passed in each call to train_batch

    logging.basicConfig(level=logging.DEBUG)

    da = xr.open_dataset("sla_c3s_clim.nc")[
        "sla_c3s"]  # sea level anomalies averaged by month-of-year, lat and lon cell

    stack_dims = ("lat", "lon")
    stack_sizes = (da.shape[1], da.shape[2])

    # each (lat,lon) position becomes an independent case
    # flatten lat and lon dimensions and transpose to arrange by (ncases, time)
    # where ncases = nlat*nlon
    instances = da.stack(case=stack_dims).transpose("case", "month").values

    # run SOM to reduce time dimension from 12 to 2
    s = SelfOrganisingMap(gridsize, gridsize, iters, seed=1, verbose=True, minibatch_size=minibatch_size)

    import time

    start_time = time.time()
    scores = s.fit_transform(instances)
    end_time = time.time()
    print("Elapsed time: %d seconds" % (int(end_time - start_time)))

    # restore lat/lon dimensions and output
    a = scores.reshape(stack_sizes + (2,))
    new_dims = stack_dims + ("som_axis",)
    output = xr.DataArray(data=a, dims=new_dims, name="monthly_sla_som")
    output.to_netcdf("som.nc")
