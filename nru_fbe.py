#!/usr/bin/env python3

import random
import simpy
import csv
import os
import time

DEBUG = False

OBSERVATION_SLOT_DURATION = 9            # observation slot length in microseconds
FIXED_FRAME_PERIOD = 1000                # fixed frame period in microseconds
IDLE_PERIOD = 100                        # idle period
MCOT = FIXED_FRAME_PERIOD - IDLE_PERIOD  # using full FIXED_FRAME_PERIOD to transmit data
MAX_SYNC_SLOT_DESYNC = 1000              # max random delay between fixed frame period of each gNB in microseconds (0 to make all gNBs synced)
MIN_SYNC_SLOT_DESYNC = 0                 # same as above but minimum between other stations


TXOP_BACKOFF = False    # custom implementation of backoff which instead of observation slots counts number of skipped TX opportunities
S_CW_MIN = 15           # number of skipped TX opportunities is drawn from 0 to S_CW_MIN
S_CW_MAX = 63

"""
'muting' proposal from 3GPP TSG-RAN WG1 Meeting #99 R1-1912449 Reno, USA, November 18th – 22nd, 2019

Alternatively, each gNB chooses a random number M before accessing the unlicensed channel. Then, the maximum
number of the fixed frame periods that gNB can grab before entering “muting” periods is set to M. If gNB grabs the
channel for M fixed frame periods, it chooses a random number N and will not access the channel for N fixed frame
periods. During such muting periods, neighbour gNBs that have failed to grab the channel due to the biased CCA timing
can get channel access opportunities so that fairness among multiple gNBs can be improved as shown in Figure 5.
"""

MUTING = True
MUTING_TXOP_ONLY = False   # special mode in which N will only be decremented when a gNB is allowed to transmit i.e. senses a free channel
N_MAX = 15                 # N - number of skipped fixed frame periods in a row, drawn from 0 to N_MAX
M_MAX = 3                  # M - number of allowed transmission in a row, drawn from 0 to M_MAX

CCA_TX_SWITCH_TIME = 5

if TXOP_BACKOFF and MUTING:
    raise ValueError("Cannot use both methods at once")


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
    
    def is_busy(self):
        for t in self.ongoing_transmisions:
            if t.start == self.env.now: # just started
                return False
            if t.end == self.env.now: # just ended
                return False
            else:
                return True
        return False

class Gnb(object):
    def __init__(self, env, id, channel, desync):
        self.env = env
        self.channel = channel
        self.id = id
        self.s_cw = S_CW_MIN             # current contention window size
        self.next_sync_slot_boundry = 0  # nex synchronization slot boundry
        self.successful_trans = 0        # number of successful transmissions
        self.total_trans = 0             # total number of transmissions
        self.total_airtime = 0           # time spent on transmiting data (including failed transmissions)
        self.succ_airtime = 0            # time spent on transmiting data (only successful transmissions)
        self.delay_total = 0             # used for calculating mean transmission delay
        self.desync = desync             # ixed frame period time offset
        self.skip_count = 0              # 'txop backoff' implementation
        self.muting_m = 0                # M (muting implementation)
        self.muting_n = 0                # N (muting implementation)

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
        """Process responsible for keeping the next fixed frame period boundry timestamp"""
        self.next_sync_slot_boundry = self.desync
        self.log("selected random fixed frame period offset equal to {} us".format(self.desync))
        yield self.env.timeout(self.desync)  # randomly desync tx starting points
        while True:
            self.next_sync_slot_boundry += FIXED_FRAME_PERIOD
            yield self.env.timeout(FIXED_FRAME_PERIOD)

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


    def wait_idle_period(self):
        """Wait gap period"""
        cca_time = OBSERVATION_SLOT_DURATION
        time_to_next_sync_slot = self.next_sync_slot_boundry - self.env.now  # calculate time needed for gap

        gap_length = time_to_next_sync_slot - cca_time
        while gap_length < 0:  # less than 0 means it's impossible to transmit in the next slot because backoff is too long
            gap_length += FIXED_FRAME_PERIOD  # check if possible to transsmit in the slot after the next slot and repeat

        self.log("calculating and waiting for the idle period ({:.0f} us)".format(gap_length))
        yield self.env.timeout(gap_length)
    
    def wait_remaining_time_to_fixed_frame_period(self):
        """"Waits until next Fixed Frame Period"""
        time_to_next_sync_slot = self.next_sync_slot_boundry - self.env.now
        yield self.env.timeout(time_to_next_sync_slot)

    def cca(self):
        """"Performs Clear Channel assesment during a single observation slot"""
        if self.channel.is_busy():
            self.log("Channel BUSY - CCA failed.")
            return False
        sensing_proc = self.env.process(self.sense_channel(1)) # single Observation Slot. ETSI 4.2.7.3.1.4
        self.channel.sensing_processes.append(sensing_proc)  # let the channel know that there is a process sensing it
        m = yield sensing_proc
        self.channel.sensing_processes.remove(sensing_proc)
        if m != 0: # CCA interupted by transmission
            self.log("Channel BUSY - CCA failed.")
            return False
        return True
    
    def mute(self):
        self.muting_n = random.randint(0, N_MAX)
        self.log("MUTING ENABLED, (N={})".format(self.muting_n))
    
    def unmute(self):
        self.muting_m = random.randint(1, M_MAX)
        self.log("MUTING DISABLED, drawn a random number of allowed transmissions in a row M={}".format(self.muting_m))

    def blocked_by_muting(self):
        self.muting_n -= 1
        self.log("MUTING IS ENABLED CAN'T PROCEED, (N={}) (will restart transmission procedure after {:.0f} us)".format(self.muting_n, FIXED_FRAME_PERIOD))
        yield self.env.timeout(FIXED_FRAME_PERIOD)
        if (self.muting_n == 0):
            self.unmute()

    def run(self):
        """Main process. Genrate new transmission, wait for channel to become idle and begin transmission"""
        if MUTING:
            self.unmute()
        while True:
            self.log("begins new transmission procedure")

            proc_start = self.env.now

            if TXOP_BACKOFF:
                self.skip_count = random.randint(0, self.s_cw)  # draw a random 'txop backoff'
                self.log("has drawn a random TX skipping counter = {}".format(self.skip_count))
            while True:
                yield self.env.process(self.wait_idle_period()) # wait for the beginning of fixed frame period (minus time for CCA right before it)
                can_transmit = yield self.env.process(self.cca()) # perform CCA (single observation slot in FBE)
                if can_transmit:
                    if TXOP_BACKOFF and self.skip_count > 0:
                        self.skip_count -= 1
                        self.log("SKIPPING FRAME PERIOD (skip count={}) (will restart transmission procedure after {:.0f} us)".format(self.skip_count, FIXED_FRAME_PERIOD))
                        yield self.env.timeout(FIXED_FRAME_PERIOD)
                        continue
                    if MUTING and self.muting_n > 0:
                        yield self.env.process(self.blocked_by_muting())
                        continue
                    break
                else:
                    if not MUTING_TXOP_ONLY and self.muting_n > 0: 
                        yield self.env.process(self.blocked_by_muting()) # decrement N even if cannot transmit
                    yield self.env.process(self.wait_remaining_time_to_fixed_frame_period()) # transmission not possible wait to next fixed frame period

            yield self.env.timeout(CCA_TX_SWITCH_TIME)

            trans_time = MCOT
            transmission = Transmission(self.env.now, trans_time, 0)
            
            proc_end = self.env.now

            self.log("is now occupying the channel for the next {:.0f} us".format(transmission.end - transmission.start))
            yield self.env.process(self.channel.transmit(transmission))
            self.log("frees the channel")
            if not transmission.collided:
                self.s_cw = S_CW_MIN
                self.log("transmission was successful. Current S_CW={}".format(self.s_cw), success=True)
                self.successful_trans += 1
                self.succ_airtime += trans_time
            else:
                if self.s_cw < S_CW_MAX:
                    self.s_cw = ((self.s_cw + 1) * 2) - 1
                self.log("transmission resulted in a collision. Current S_CW={}".format(self.s_cw), fail=True)
            
            if MUTING:
                self.muting_m -= 1
                self.log("remaining M transmissions allowed: {}".format(self.muting_m))
                if self.muting_m <= 0: # can be -1 if drawn N=0
                    self.mute()

            self.delay_total += (proc_end - proc_start)

            self.total_trans += 1
            self.total_airtime += trans_time


def run_simulation(sim_time, nr_of_gnbs, seed, desyncs=None):
    """Run simulation. Return a list with results."""
    random.seed(seed)

    env = simpy.Environment()
    channel = Channel(env)

    if desyncs is None:
        ## random desync offsets, but every value is at least MIN_SYNC_SLOT_DESYNC as far from any other value
        desyncs = random_sample(MAX_SYNC_SLOT_DESYNC - MIN_SYNC_SLOT_DESYNC, nr_of_gnbs, MIN_SYNC_SLOT_DESYNC) # MAKE DESYNCED
        # desyncs = [0]*nr_of_gnbs # MAKE SYNCED
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
                        'eff': gnb.succ_airtime,
                        'delay': gnb.delay_total})
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


def process_results(results, sim_time, seed, nr_of_gnbs):
    occupancy_total = 0
    trans_total = 0
    fail_total = 0
    succ_total = 0
    efficient_airtime = 0
    delay_total = 0

    for result in results:
        occupancy_total += result['occ']
        trans_total += result['trans']
        fail_total += result['fail']
        succ_total += result['succ']
        efficient_airtime += result['eff']
        delay_total += result['delay']
    ret = {}
    ret['fail'] = fail_total
    ret['succ'] = succ_total
    ret['trans'] = trans_total
    ret['pc'] = fail_total / trans_total
    ret['occ'] = occupancy_total
    ret['eff'] = efficient_airtime / (sim_time*1e6)
    ret['delay'] = delay_total
    # calculate Jain's fairnes index
    sum_sq = 0
    n = len(results)
    for result in results:
        sum_sq += result['eff']**2
    ret['jfi'] = efficient_airtime**2 / (n * sum_sq)

    parameters = {'sim_time': sim_time,
                  'seed': seed,
                  'N_sta': nr_of_gnbs,
                  'sync': FIXED_FRAME_PERIOD,
                  'mcot': MCOT}

    dump_csv(parameters, ret)

    return ret


if __name__ == "__main__":
    SIM_TIME = 0.5
    SEED = 42
    NR_OF_GNBS = 2

    start_time = time.time()
    results = run_simulation(sim_time=SIM_TIME, nr_of_gnbs=NR_OF_GNBS, seed=SEED)
    end_time = time.time()
    processed = process_results(results, SIM_TIME, SEED, NR_OF_GNBS)


    for result in results:
        print("------------------------------------")
        print(result['id'])
        print('Collisions: {}/{} ({}%)'.format(result['fail'],
                                            result['trans'],
                                            result['pc'] * 100 if result['pc'] is not None else 'N/A'))
        print('Total airtime: {} ms'.format(result['occ'] / 1e3))
        print('Mean transmission delay: {} ms'.format(result['delay'] / 1e3 / result['trans'] if result['trans'] != 0 else 'N/A'))
        print('Channel efficiency: {:.2f}'.format(result['occ'] / processed['occ']))

    print('====================================')
    print('Total collision probability: {:.4f}'.format(processed['pc']))
    print('Total channel efficiency: {:.4f}'.format(processed['eff']))
    print("Jain's fairness index: {:.4f}".format(processed['jfi']))
    print("Mean of mean delays: {:.4f}".format(processed['delay'] / 1e3 / processed['trans']))
    print('====================================')
    print("--- Simulation ran for %s seconds ---" % (end_time - start_time))