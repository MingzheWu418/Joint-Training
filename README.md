# Relational Inference
Joint training model for spatial relation inference. 

$ ./tree-md .
# Project tree

.
 * [KETI_oneweek](./KETI_oneweek) contains the KETI-Oneweek dataset with 204 sensors in an office building, each with 130,000 readings.
 * [colocation](./colocation)
   * [main.py](./colocation/main.py) is the file for model training
   * [stn.yaml](./colocation/stn.yaml) fixes hyper-parameters for STN, as well as which dataset to be used.
   * [10_rooms.json](./colocation/10_rooms.json) and [10_rooms_soda.json](./colocation/10_rooms_soda.json) store hyper-parameters for the Genetic Algorithm baseline on different datasets.
   * [Data.py](./colocation/Data.py) is responsible for reading in datasets, such as Soda or KETI-Oneweek.
   * [rawdata](./colocation/rawdata/) contains our other dataset Soda
     * [metadata](./colocation/rawdata/metadata)
       * [Soda](./colocation/rawdata/metadata/Soda) is the directory that stores all sensor readings.
 * [README.md](./README.md)
