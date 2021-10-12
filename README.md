# Consistent Post-Reconstruction for Progressive Photon Mapping
Authors: [Hajin Choi](https://cglab.gist.ac.kr/people), [Bochang Moon](https://cglab.gist.ac.kr/people/bochang.html)

![](main.png)

This is an official implementation of the paper "*Consistent Post-Reconstruction for Progressive Photon Mapping*", which is presented in Pacific Graphics 2021. This repository includes training/test codes with four test images needed to reproduce the results in the paper. For re-training using other images, please refer to the paper and [scripts/config.py](scripts/config.py) to see how the input images should be consist of. We recommend the [Mitsuba renderer](https://github.com/mitsuba-renderer/mitsuba) for generating the SPPM images.

## Setup

### Tested environments
- OS: Ubuntu 20.04
- CPU: AMD Ryzen Threadripper 3990X
- GPU: NVIDIA RTX 3090

### Prerequisites
- NVIDIA graphic driver (tested by version >= 460)
- Docker
- nvidia-docker
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

### Installation
1. Clone this repository
2. Inside the cloned directory, build a docker image by  
`docker build . -t IMAGE_NAME`
3. Create a container by  
`docker run -d -it -v $PWD:/consistent-recon --gpus all IMAGE_NAME`

### Configurations
- For training, set following things in [scripts/config.py](scripts/config.py)
  ```python
  # Mode select (TRAINING, TEST)
  config["mode"] = "TRAINING"
  # Number of epochs
  config['epochs'] = 100
  # Prefix for dataset directory. Note that it will be used to find the refereces too (e.g., dataset_refereces). Same for test.
  config["datasetPrefix"] = "dataset"
  # Directory of training dataset
  config["trainDatasetDirectory"] = "dataset_train"
  # List of scenes (e.g., dataset_train/box)
  config["trainScenes"] = ['box', 'sponza']
  ```
- For test, set following things in [scripts/config.py](scripts/config.py)
  ```python
  # Mode select (TRAINING, TEST)
  config["mode"] = "TEST"
  # Epoch of checkpoint to load
  config["loadEpoch"] = "100"
  # Directory of test dataset
  config["testDatasetDirectory"] = "dataset_test"
  # List of scenes (e.g., dataset_test/box)
  config["testScenes"] = ['bookshelf', 'breakfast-room', 'pool', 'water']
  ```
 - Others are optional and can be left at their default values.
 - Outputs of the test will be stored in `/consistent-recon/output`

---

## Run

 1. Attach to the container by a docker command e.g.,  
   `docker exec -w /consistent-recon -it CONTAINER_ID bash`
 2. Run the code by `python scripts/main.py`
    - If you meet a **weird** memory increase along the epochs, then run the script with `TCMalloc`   
      `LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python scripts/main.py`

---

## Citation

<!-- ```
@article{Choi21,
  author = {},
  title = {},
  year = {},
  issue_data = {},
  volume = {},
  number = {},
  journal = {},
  month = Oct,
  articleno = {},
  numpages = {}
}
``` -->

## License

All source codes are released under a [BSD License](LICENSE).

## Credits

We used exr-related operations using the script [scripts/exr.py](scripts/exr.py) from [KPCN](https://jannovak.info/publications/KPCN/index.html).
