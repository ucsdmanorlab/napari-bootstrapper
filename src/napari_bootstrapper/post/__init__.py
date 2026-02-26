from pprint import pprint

from .cc import cc_from_affinities
from .mws import mwatershed_from_affinities

# from .ws import watershed_from_affinities

DEFAULT_SEG_PARAMS = {
    # "watershed": {
    #     "sigma": None,
    #     "noise_eps": None,
    #     "bias": None,
    #     "threshold": 0.5,
    #     "min_seed_distance": 10,
    #     "fragments_in_xy": True,
    # },
    "mutex watershed": {
        "filter_fragments": 0.1,
        "sigma": None,
        "noise_eps": 0.001,
        "bias": [-0.4, -0.4, -0.4, -0.7, -0.7, -0.7, -0.7, -0.7, -0.7],
        "min_seed_distance": None,
        "seed_eps": None,
        "neighborhood": [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [-2, 0, 0],
            [0, -9, 0],
            [0, 0, -9],
            [-3, 0, 0],
            [0, -27, 0],
            [0, 0, -27],
        ],
        "strides": [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [2, 9, 9],
            [2, 9, 9],
            [2, 9, 9],
            [3, 27, 27],
            [3, 27, 27],
            [3, 27, 27],
        ],
        "randomized_strides": True,
        "remove_debris": 64,
    },
    "connected components": {
        "sigma": None,
        "noise_eps": None,
        "bias": 0.0,
        "threshold": 0.5,
    },
}


def segment_affs(affs, method="mutex watershed", params=DEFAULT_SEG_PARAMS):
    print(
        f"Segmenting affs of shape {affs.shape} with method {method} and params:"
    )
    pprint(params[method])
    # if method == "watershed":
    #     return watershed_from_affinities(affs[:3], **params[method])
    if method == "mutex watershed":
        return mwatershed_from_affinities(affs, **params[method])
    elif method == "connected components":
        return cc_from_affinities(affs, **params[method])
    else:
        raise ValueError("Invalid segmentation method")
