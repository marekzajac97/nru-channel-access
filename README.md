# NR-U SimPy Simulator

`nru.py` is a discrete-event simulator of channel access for NR-U base stations (gNBs) written in Python using the [SimPy](https://simpy.readthedocs.io/en/latest/) library. The simulator implements the listen before talk (LBT) competitive channel access mechanism under the assumption that NR-U transmissions may begin only at slot boundaries. The simulator supports any number of transmitting stations and configurable transmission parameters. It outputs the following metrics:

- collision probability -- ratio between collided and all attempted transmissions,
- channel efficiency -- ratio of successful transmission duration and simulation time,
- Jain's fairness index -- calculated over per-station airtime.

## Usage

Simply run `nru.py` in a Python environment. 

The following parameters set in the script may be of particular interest:

```python
SYNCHRONIZATION_SLOT_DURATION = 1000  # synchronization slot duration [us]
RS_SIGNALS = False                    # if True use reservation signals else gap 
PARTIAL_ENDING_SUBFRAMES = False      # if True, random last slot duration (1-14 OFDM symbols)
[...]
CW_MIN = 15 # minimum contention window value
CW_MAX = 63 # maximum contention window value
[...]
SIM_TIME = 100   # simulation time
SEED = 42        # seed for PRNG
NR_OF_GNBS = 10  # number of transmitting gNBs
```

## Research

The simulator was used to conduct the research presented in the following papers:

- M. Zając and S. Szott, "[Resolving 5G NR-U Contention for Gap-Based Channel Access in Shared Sub-7 GHz Bands](https://ieeexplore.ieee.org/abstract/document/9673740)," in IEEE Access, vol. 10, pp. 4031-4047, 2022, doi: 10.1109/ACCESS.2022.3141193.

