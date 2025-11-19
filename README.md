# cellpose-tools
This repository contains command line tools that allows one to run [Cellpose](https://github.com/MouseLand/cellpose) distributed over a Dask cluster. These are used mostly by the [Cellpose Nextflow pipeline](https://github.com/JaneliaSciComp/nf-cellpose).

# Setup the environment
```
mamba env create -n cellpose-tools -f conda-env.yml
pip install -e .
```

# Running distributed cellpose
```
python -m tools.main_distributed_cellpose \
    -i <input-image-or-container> -o <output-image-or-container> \
    --dask-scheduler <tcp://x.x.x.x:port> 
```

If no cluster is available you have 2 options:
* Use a local client that can still chunk the image into smaller blocks and pass these to a local Dask client that has a number of local workers equal to the argument specified by `--local-dask-workers <nworkers>`.
* Don't use any distribution and simply run Cellpose eval method on the entire image - if no `--local-dask-workers` is present or if `--local-dask-workers 0`

# Image preprocessing
The tool allows for dynamic configuration of preprocessing algorithms, but so far we only support a gaussian filtering applied to the image before running the cellpose segmentation.
The parameters for these preprocessing steps can be defined in a YAML file like this:
```
unsharp:
    sigma_one: 1.0
    weight: 0.1
    iterations: 5
    sigma_two: 0.1
```
