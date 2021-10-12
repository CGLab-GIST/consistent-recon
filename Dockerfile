FROM tensorflow/tensorflow:2.4.1-gpu

RUN apt-get update --fix-missing
RUN apt-get install -y libtcmalloc-minimal4 libopenexr-dev
RUN pip install OpenEXR psutil parmap tqdm pytz tqdm scipy

WORKDIR /consistent-recon