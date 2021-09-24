import random
import simpy
import math
import csv
import statistics
import numpy as np
import os
from enum import Enum
import time

class Gap(Enum):
    BEFORE = 1
    AFTER = 2
    DURING = 3
    AFTER_WITH_CCA = 4
    INSIDE = 5

DEBUG = False

DETER_PERIOD = 16                     # Time which a node is required to wait at the start of prioritization period in microseconds
OBSERVATION_SLOT_DURATION = 9         # observation slot length in microseconds
SYNCHRONIZATION_SLOT_DURATION = 1000  # synchronization slot length in microseconds
MAX_SYNC_SLOT_DESYNC = 1000           # max random delay between sync slots of each gNB in microseconds (0 to make all gNBs synced)
MIN_SYNC_SLOT_DESYNC = 0              # same as above but minimum between other stations
RS_SIGNALS = False                    # if True use reservation signals before transmission. Use gap otherwise
GAP_PERIOD = Gap.AFTER_WITH_CCA               # insert backoff 'before', 'during', 'after', 'after_cca' backoff procedure.
PARTIAL_ENDING_SUBFRAMES = False      # make last slot duration random between 1 and 14 OFDM slots
SKIP_NEXT_SLOT_BOUNDARY = False
SKIP_NEXT_TXOP = False

CCA_TX_SWITCH_TIME = 0

# set of parameters only applicaple with GAP_PERIOD set to DURING
BACKOFF_SLOTS_SPLIT = 'fixed'       # 'fixed' or 'variable'
BACKOFF_SLOTS_TO_LEAVE = 7          # how many slots from the backoff procedure leave to count after the gap. A fixed number of slots or percentage of backoff.

# Channel access class 1
# M = 1                               # fixed number of observation slots in prioritization period
# CW_MIN = 3                          # minimum contention window size
# CW_MAX = 7                          # maximum contention window size
# MCOT = 2                            # maximum channel occupancy time

# Channel access class 2
# M = 1
# CW_MIN = 7
# CW_MAX = 15
# MCOT = 3

# Channel access class 3
M = 3
CW_MIN = 15
CW_MAX = 63
MCOT = 8

# Channel access class 4
# M = 7
# CW_MIN = 15
# CW_MAX = 1023
# MCOT = 8

def random_sample(max, number, min_distance=0):
    """
    Returns number of elements from 0 to max, with a minimum distance
    """
    samples = random.sample(range(max-(number-1)*(min_distance-1)), number)
    indices = sorted(range(len(samples)), key=lambda i: samples[i])
    ranks = sorted(indices, key=lambda i: indices[i])
    return [sample + (min_distance-1)*rank for sample, rank in zip(samples, ranks)]


class Transmission(object):
    def __init__(self, start, airtime, rs_time=0):
        self.start = start                          # transmission start (including RS)
        self.end = start + rs_time + airtime        # transmission end
        self.airtime = airtime                      # time spent on sending data
        self.rs_time = rs_time                      # time spent on sending reservation signal before data
        self.collided = False                       # true if transmission colided with another one


class Channel(object):
    def __init__(self, env):
        self.ongoing_transmisions = list()
        self.sensing_processes = list()
        self.env = env

    def transmit(self, transmission):
        """Occupy channel for the duration of reservation signal and transmission duration, check for collison at the end"""
        self.ongoing_transmisions.append(transmission)
        for process in self.sensing_processes:
            if process.is_alive:
                process.interrupt()
        yield self.env.timeout(transmission.rs_time)
        yield self.env.timeout(transmission.airtime)
        self.check_collision(transmission)
        self.ongoing_transmisions.remove(transmission)

    def check_collision(self, transmission):
        """Check if a given data transmission start/end times overlapped with any of the ongoing transmissions"""
        for t in self.ongoing_transmisions:
            if (t.end > (transmission.start + transmission.rs_time) and
                    (t.start + t.rs_time) < transmission.end and transmission is not t):
                transmission.collided = True
                t.collided = True

    def time_until_free(self):
        """Return time left for the channel to become idle"""
        max_time = 0
        for t in self.ongoing_transmisions:
            time_left = t.end - self.env.now
            if time_left > max_time:
                max_time = time_left
        return max_time


class Gnb(object):
    def __init__(self, env, id, channel, desync):
        self.env = env
        self.channel = channel
        self.id = id
        self.cw = CW_MIN                 # current contention window size
        self.N = None                    # backoff counter
        self.next_sync_slot_boundary = 0  # nex synchronization slot boundary
        self.successful_trans = 0        # number of successful transmissions
        self.total_trans = 0             # total number of transmissions
        self.total_airtime = 0           # time spent on transmiting data (including failed transmissions)
        self.succ_airtime = 0            # time spent on transmiting data (only successful transmissions)
        self.desync = desync
        self.skip = None

        self.env.process(self.sync_slot_counter())
        self.env.process(self.run())

    def _log(self, output):
        return "{:.0f}|gNB-{}\t: {}".format(self.env.now, self.id, output)

    def log(self, output, fail=False, success=False):
        if DEBUG:
            if fail:
                print("\033[91m" + self._log(output) + "\033[0m")
            elif success:
                print("\033[92m" + self._log(output) + "\033[0m")
            elif self.id == 0:
                print("\033[35m" + self._log(output) + "\033[0m")  # highlight one gNB
            else:
                print(self._log(output))     

    def sync_slot_counter(self):
        """Process responsible for keeping the next sync slot boundary timestamp"""
        self.next_sync_slot_boundary = self.desync
        self.log("selected random sync slot offset equal to {} us".format(self.desync))
        yield self.env.timeout(self.desync)  # randomly desync tx starting points
        while True:
            self.next_sync_slot_boundary += SYNCHRONIZATION_SLOT_DURATION
            yield self.env.timeout(SYNCHRONIZATION_SLOT_DURATION)

    def wait_for_idle_channel(self):
        """Wait until the channel is sensed idle"""
        waiting_time = self.channel.time_until_free()
        while waiting_time != 0:
            self.log("is sensing channel busy (for at least {} us)".format(waiting_time))
            yield self.env.timeout(waiting_time)
            waiting_time = self.channel.time_until_free()  # in case a new transmission started check again

    def sense_channel(self, slots_to_wait):
        """Wait for the duration of slots_to_wait x OBSERVATION_SLOT_DURATION. Return remaining slots (0 if procedure was successful)"""
        try:
            while slots_to_wait > 0:
                yield self.env.timeout(OBSERVATION_SLOT_DURATION)
                slots_to_wait -= 1
        except simpy.Interrupt:
            pass  # if the procedure was interrupted by the start of a new transmission retrun remaining slots
            # slots_to_wait -= 1  # simulate propagation delay  
        return slots_to_wait

    def wait_prioritization_period(self):
        """Wait initial 16 us + m x OBSERVATION_SLOT_DURATION us"""
        m = M
        while m > 0:  # if m == 0 continue to backoff
            yield self.env.process(self.wait_for_idle_channel())
            self.log("Channel is now idle, waiting the duration of the deter period ({:.0f} us)".format(DETER_PERIOD))
            yield self.env.timeout(DETER_PERIOD)
            if self.channel.time_until_free() == 0:
                self.log("Checking the channel after deter period: IDLE - wait {} observation slots".format(M))
            else:
                self.log("Checking the channel after deter period: BUSY - wait for idle channel")
                continue  # start the whole proces over again

            sensing_proc = self.env.process(self.sense_channel(M))
            self.channel.sensing_processes.append(sensing_proc)  # let the channel know that there is a process sensing it
            m = yield sensing_proc
            self.channel.sensing_processes.remove(sensing_proc)
            if m != 0:
                self.log("Channel BUSY - prioritization period failed.")

    def wait_gap_period(self):
        """Wait gap period"""
        backoff_time = self.N * OBSERVATION_SLOT_DURATION  # time that will be taken for backoff
        time_to_next_sync_slot = self.next_sync_slot_boundary - self.env.now  # calculate time needed for gap

        gap_length = time_to_next_sync_slot - backoff_time
        while gap_length < 0:  # less than 0 means it's impossible to transmit in the next slot because backoff is too long
            gap_length += SYNCHRONIZATION_SLOT_DURATION  # check if possible to transsmit in the slot after the next slot and repeat

        self.log("calculating and waiting the gap period ({:.0f} us)".format(gap_length))
        if GAP_PERIOD != Gap.INSIDE:
            yield self.env.timeout(gap_length)
        else:
            self.log("waiting first half of the gap period ({:.0f} us)".format(gap_length / 2))
            yield self.env.timeout(gap_length / 2)
            self.log("(re)starting backoff procedure in the middle of the gap (slots to wait: {})".format(self.N))
            yield self.env.process(self.wait_random_backoff())
            if self.N == 0:
                self.log("waiting second half of the gap period ({:.0f} us)".format(gap_length / 2))
                yield self.env.timeout(gap_length / 2)

    def wait_random_backoff(self):
        """Wait random number of slots N x OBSERVATION_SLOT_DURATION us"""
        if self.channel.time_until_free() > 0:  # if channel is busy at the start of backoff (e.g. after gap period channel is busy) imidiately return
            return

        if not RS_SIGNALS and GAP_PERIOD == Gap.DURING:
            if BACKOFF_SLOTS_SPLIT == 'fixed':
                backoff_slots_left = BACKOFF_SLOTS_TO_LEAVE
            elif BACKOFF_SLOTS_SPLIT == 'variable':
                backoff_slots_left = int(math.ceil(BACKOFF_SLOTS_TO_LEAVE * self.N))

            slots_to_wait = self.N - backoff_slots_left
            slots_to_wait = slots_to_wait if slots_to_wait >= 0 else 0  # if backoff is longer than BACKOFF_SLOTS_TO_LEAVE, count these slots after gap
            self.log("will wait {} slots before stopping backoff".format(slots_to_wait))
        else:
            slots_to_wait = self.N
        sensing_proc = self.env.process(self.sense_channel(slots_to_wait))
        self.channel.sensing_processes.append(sensing_proc)
        remaining_slots = yield sensing_proc
        self.channel.sensing_processes.remove(sensing_proc)

        if not RS_SIGNALS and GAP_PERIOD == Gap.DURING and remaining_slots == 0:  # redo backoff for additional backoff_slots_left
            self.log("stopping backoff and inserting gap now")
            yield self.env.process(self.wait_gap_period())
            self.log("waiting remaining backoff slots ({}) after gap".format(self.N - slots_to_wait))
            if self.channel.time_until_free() > 0:  # cca at the beggining of the backoff
                self.N = self.N - slots_to_wait
                return
            sensing_proc = self.env.process(self.sense_channel(self.N - slots_to_wait))
            self.channel.sensing_processes.append(sensing_proc)
            self.N = yield sensing_proc
            self.channel.sensing_processes.remove(sensing_proc)
        elif not RS_SIGNALS and GAP_PERIOD == Gap.DURING and remaining_slots > 0:
            self.N = remaining_slots + self.N - slots_to_wait
        else:
            self.N = remaining_slots

    def run(self):
        """Main process. Genrate new transmission, wait for channel to become idle and begin transmission"""
        while True:
            self.log("begins new transmission procedure")

            self.N = random.randint(0, self.cw)  # draw a random backoff
            self.log("has drawn a random backoff counter = {}".format(self.N))
            while True:
                yield self.env.process(self.wait_prioritization_period())
                self.log("prioritization period has finished")
                if not RS_SIGNALS and GAP_PERIOD == Gap.BEFORE or GAP_PERIOD == Gap.INSIDE:  # if RS signals not used, use gap BEFORE backoff procedure
                    yield self.env.process(self.wait_gap_period())
                if GAP_PERIOD != Gap.INSIDE:  # do not wait backoff in case it was already done inside wait_gap_period
                    self.log("(re)starting backoff procedure (slots to wait: {})".format(self.N))
                    if self.N == 0 and GAP_PERIOD == Gap.BEFORE and self.channel.time_until_free() > 0:
                        self.log("Remaining backoff slots is 0 but channel is busy, aborting transmission")
                        continue
                    else:
                        yield self.env.process(self.wait_random_backoff())
                if self.N == 0:
                    yield self.env.timeout(CCA_TX_SWITCH_TIME)  # simulate short switching from sensing to TX
                    break
                else:
                    self.log("Channel BUSY - backoff is frozen. Remaining slots: {}".format(self.N))

            if not RS_SIGNALS and (GAP_PERIOD == Gap.AFTER or GAP_PERIOD == Gap.AFTER_WITH_CCA):  
                yield self.env.process(self.wait_gap_period())  # when RS signals are not used, use gap AFTER backoff procedure and transmit imidiately
            if not RS_SIGNALS and (GAP_PERIOD == Gap.AFTER_WITH_CCA or GAP_PERIOD == Gap.INSIDE):
                if self.channel.time_until_free() > 0:
                    self.log("Channel BUSY after gap period, aborting transmission")
                    continue

            if (SKIP_NEXT_SLOT_BOUNDARY and self.skip == self.env.now) or (SKIP_NEXT_TXOP and self.skip):
                self.skip = None
                self.log("SKIPPING SLOT (will restart transmission procedure after {:.0f} us)".format(SYNCHRONIZATION_SLOT_DURATION))
                yield self.env.timeout(SYNCHRONIZATION_SLOT_DURATION)
                continue

            if PARTIAL_ENDING_SUBFRAMES:
                last_slot = random.choice([3, 6, 9, 10, 11, 12, 14])
                trans_time = (MCOT * 1e3 - SYNCHRONIZATION_SLOT_DURATION) + (SYNCHRONIZATION_SLOT_DURATION/14) * last_slot
            else:
                trans_time = MCOT * 1e3  # if gap in use = full MCOT to transmit data
            if RS_SIGNALS:
                time_to_next_sync_slot = self.next_sync_slot_boundary - self.env.now  # calculate time needed for reservation signal
                trans_time = (trans_time - time_to_next_sync_slot)  # if RS in use = the rest of MCOT to transmit data
                transmission = Transmission(self.env.now, trans_time, time_to_next_sync_slot)
            else:
                transmission = Transmission(self.env.now, trans_time, 0)

            self.log("is now occupying the channel for the next {:.0f} us (RS={:.0f} us)".format(transmission.end - transmission.start,
                                                                                            transmission.rs_time))
            yield self.env.process(self.channel.transmit(transmission))
            self.log("frees the channel")
            if not transmission.collided:
                self.cw = CW_MIN
                self.log("transmission was successful. Current CW={}".format(self.cw), success=True)
                self.successful_trans += 1
                self.succ_airtime += trans_time
                if SKIP_NEXT_SLOT_BOUNDARY or SKIP_NEXT_TXOP:
                    self.skip = self.next_sync_slot_boundary
            else:
                if self.cw < CW_MAX:
                    self.cw = ((self.cw + 1) * 2) - 1
                self.log("transmission resulted in a collision. Current CW={}".format(self.cw), fail=True)

            self.total_trans += 1
            self.total_airtime += trans_time


def run_simulation(sim_time, nr_of_gnbs, seed, desyncs=None):
    """Run simulation. Return a list with results."""
    random.seed(seed)

    env = simpy.Environment()
    channel = Channel(env)

    if desyncs is None:
        ## random desync offsets, but every value is at least MIN_SYNC_SLOT_DESYNC as far from any other value
        desyncs = random_sample(MAX_SYNC_SLOT_DESYNC - MIN_SYNC_SLOT_DESYNC, nr_of_gnbs, MIN_SYNC_SLOT_DESYNC)
        ## random offset from a set (set contains values with step of MIN_SYNC_SLOT_DESYNC)
        # desync_set = list(np.linspace(0, MAX_SYNC_SLOT_DESYNC, num=int(MAX_SYNC_SLOT_DESYNC/MIN_SYNC_SLOT_DESYNC)+1))[:-1]
        # desyncs = [random.choice(desync_set) for _ in range(0, nr_of_gnbs)]
        print(desyncs)

    gnbs = [Gnb(env, i, channel, desyncs[i]) for i in range(nr_of_gnbs)]

    env.run(until=(sim_time*1e6))

    results = list()

    for gnb in gnbs:
        results.append({'id': gnb.id,
                        'succ': gnb.successful_trans,
                        'fail': gnb.total_trans - gnb.successful_trans,
                        'trans': gnb.total_trans,
                        'pc': 1 - gnb.successful_trans / gnb.total_trans if gnb.total_trans > 0 else None,
                        'occ': gnb.total_airtime,
                        'eff': gnb.succ_airtime})
    return results


def dump_csv(parameters, results, filename=None):
    filename = filename if filename else 'results.csv'
    write_header = True
    if os.path.isfile(filename):
        write_header = False
    with open(filename, mode='a') as csv_file:
        results.update(parameters)
        fieldnames = results.keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(results)


def process_resutls(results, sim_time, seed, nr_of_gnbs):
    occupancy_total = 0
    trans_total = 0
    fail_total = 0
    succ_total = 0
    efficient_airtime = 0

    for result in results:
        occupancy_total += result['occ']
        trans_total += result['trans']
        fail_total += result['fail']
        succ_total += result['succ']
        efficient_airtime += result['eff']
    ret = {}
    ret['fail'] = fail_total
    ret['succ'] = succ_total
    ret['trans'] = trans_total
    ret['pc'] = fail_total / trans_total
    ret['occ'] = occupancy_total
    ret['eff'] = efficient_airtime / (sim_time*1e6)
    # calculate Jain's fairness index
    sum_sq = 0
    n = len(results)
    for result in results:
        sum_sq += result['eff']**2
    ret['jfi'] = efficient_airtime**2 / (n * sum_sq)

    parameters = {'sim_time': sim_time,
                  'seed': seed,
                  'N_sta': nr_of_gnbs,
                  'rs': RS_SIGNALS,
                  'gap': GAP_PERIOD if "N/A" else RS_SIGNALS,
                  'cw_min': CW_MIN,
                  'cw_max': CW_MAX,
                  'sync': SYNCHRONIZATION_SLOT_DURATION,
                  'partial': PARTIAL_ENDING_SUBFRAMES,
                  'mcot': MCOT}

    dump_csv(parameters, ret)

    return ret


if __name__ == "__main__":
    SIM_TIME = 100
    SEED = 42
    NR_OF_GNBS = 10
    start_time = time.time()
    results = run_simulation(sim_time=SIM_TIME, nr_of_gnbs=NR_OF_GNBS, seed=SEED)
    end_time = time.time()
    processed = process_resutls(results, SIM_TIME, SEED, NR_OF_GNBS)


    for result in results:
        print("------------------------------------")
        print(result['id'])
        print('Collisions: {}/{} ({}%)'.format(result['fail'],
                                              result['trans'],
                                              result['pc'] * 100 if result['pc'] is not None else 'N/A'))
        print('Total airtime: {} ms'.format(result['occ'] / 1e3))
        print('Channel efficiency: {:.2f}'.format(result['occ'] / processed['occ']))

    print('====================================')
    print('Total collision probability: {:.4f}'.format(processed['pc']))
    print('Total channel efficiency: {:.4f}'.format(processed['eff']))
    print("Jain's fairness index: {:.4f}".format(processed['jfi']))
    print('====================================')
    print("--- Simulation ran for %s seconds ---" % (end_time - start_time))