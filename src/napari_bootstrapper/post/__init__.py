from pprint import pprint

from .cc import cc_from_affinities
from .mws import mwatershed_from_affinities

DEFAULT_SEG_PARAMS = {
    "mutex watershed": {
        "bias": (-0.4, -0.7),
        "filter_fragments": 0.1,
        "sigma": None,
        "noise_eps": 0.001,
        "min_seed_distance": None,
        "seed_eps": None,
        "randomized_strides": True,
        "remove_debris": 64,
    },
    "connected components": {
        "bias": 0.0,
        "sigma": None,
        "noise_eps": None,
        "threshold": 0.5,
    },
}


def segment_affs(affs, method="mutex watershed", params=DEFAULT_SEG_PARAMS):
    print(
        f"Segmenting affs of shape {affs.shape} with method {method} and params:"
    )
    pprint(params[method])
    if method == "mutex watershed":
        return mwatershed_from_affinities(affs, **params[method])
    elif method == "connected components":
        return cc_from_affinities(affs, **params[method])
    else:
        raise ValueError("Invalid segmentation method")
