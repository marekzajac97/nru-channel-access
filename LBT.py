import random
import simpy
import math
import numpy as np

DEBUG = False
gnbs = None

DETER_PERIOD = 16                     # Time which a node is required to wait at the start of prioritization period in microseconds
OBSERVATION_SLOT_DURATION = 9         # observation slot length in microseconds
SYNCHRONIZATION_SLOT_DURATION = 1000  # synchronization slot length in microseconds
MAX_SYNC_SLOT_DESYNC = 1000           # max random delay between sync slots of each gNB in microseconds (0 to make all gNBs synced)
MIN_SYNC_SLOT_DESYNC = 60              # same as above but minimum between other stations
RS_SIGNALS = False                    # if True use reservation signals before transmission. Use gap otherwise
GAP_PERIOD = 'after_cca'              # insert backoff 'before', 'during', 'after', 'after_cca' backoff procedure.
PARTIAL_ENDING_SUBFRAMES = True       # make last slot duration random between 1 and 14 OFDM slots
SKIP_NEXT_SLOT_BOUNDRY = True

BACKOFF_INSIDE = False                # wait backoff in the middle of the gap, set this on True only with GAP_PERIOD == 'before' !!!

# set of parameters only applicaple with GAP_PERIOD set to 'during'
BACKOFF_SLOTS_SPLIT = 'fixed'      # 'fixed' or 'variable'
BACKOFF_SLOTS_TO_LEAVE = 7          # how many slots from the backoff procedure leave to count after the gap. A fixed number of slots or percentage of backoff.

# Channel access class 1
# M = 1                               # fixed number of observation slots in prioritization period
# CW_MIN = 3                          # minimum contention window size
# CW_MAX = 7                          # maximum
# MCOT = 2

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

# no backoff
CW_MIN = 0
CW_MAX = 0


def log(output):
    if DEBUG:
        if 'gNB 0' in output:
            print("\033[35m" + output + "\033[0m")  # highlight one gNB
        else:
            print(output)


def log_fail(output):
    if DEBUG:
        print("\033[91m" + output + "\033[0m")


def log_success(output):
    if DEBUG:
        print("\033[92m" + output + "\033[0m")


def ranks(sample):
    """
    Return the ranks of each element in an integer sample.
    """
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])


def sample_with_min_distance(n, k, d):
    """
    Sample of k elements from range(n), with a minimum distance d.
    """
    sample = random.sample(range(n-(k-1)*(d-1)), k)
    return [s + (d-1)*r for s, r in zip(sample, ranks(sample))]


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
        self.next_sync_slot_boundry = 0  # nex synchronization slot boundry
        self.successful_trans = 0        # number of successful transmissions
        self.total_trans = 0             # total number of transmissions
        self.total_airtime = 0           # time spent on transmiting data (including failed transmissions)
        self.succ_airtime = 0            # time spent on transmiting data (only successful transmissions)
        self.desync = desync
        self.skip = None

        self.env.process(self.sync_slot_counter())
        self.env.process(self.run())

    def sync_slot_counter(self):
        """Process responsible for keeping the next sync slot boundry timestamp"""
        # desync = 100*int(self.id[-1])
        # self.desync = random.randint(0, MAX_SYNC_SLOT_DESYNC)
        self.next_sync_slot_boundry = self.desync
        log("{:.0f}:\t {} selected random sync slot offset equal to {} us".format(self.env.now, self.id, self.desync))
        yield self.env.timeout(self.desync)  # randomly desync tx starting points
        while True:
            self.next_sync_slot_boundry += SYNCHRONIZATION_SLOT_DURATION
            # if 'gNB 0' in self.id:
            #     log_fail("{:.0f}:\t {} SYNC SLOT BOUNDRY NOW".format(self.env.now, self.id))
            yield self.env.timeout(SYNCHRONIZATION_SLOT_DURATION)

    def wait_for_idle_channel(self):
        """Wait until the channel is sensed idle"""
        waiting_time = self.channel.time_until_free()
        while waiting_time != 0:
            log("{:.0f}:\t {} is sensing channel busy (for at least {} us)".format(self.env.now, self.id, waiting_time))
            yield self.env.timeout(waiting_time)
            waiting_time = self.channel.time_until_free()  # in case a new transmission started check again

    def sense_channel(self, slots_to_wait):
        """Wait for the duration of slots_to_wait x OBSERVATION_SLOT_DURATION. Return remaining slots (0 if procedure was successful)"""
        try:
            while slots_to_wait > 0:
                yield self.env.timeout(OBSERVATION_SLOT_DURATION)
                slots_to_wait -= 1
        except simpy.Interrupt:
            pass
            # slots_to_wait -= 1  # simulate propagation delay
            # pass  # if the procedure was interrupted by the start of a new transmission retrun remaining slots
        return slots_to_wait

    def wait_prioritization_period(self):
        """Wait initial 16 us + m x OBSERVATION_SLOT_DURATION us"""
        m = M
        while m > 0:  # if m == 0 continue to backoff
            yield self.env.process(self.wait_for_idle_channel())
            log("{:.0f}:\t {} Channel is now idle, waiting the duration of the deter period ({:.0f} us)".format(self.env.now, self.id, DETER_PERIOD))
            yield self.env.timeout(DETER_PERIOD)
            if self.channel.time_until_free() == 0:
                log("{:.0f}:\t {} Checking the channel after deter period: IDLE - wait {} observation slots".format(self.env.now, self.id, M))
            else:
                log("{:.0f}:\t {} Checking the channel after deter period: BUSY - wait for idle channel".format(self.env.now, self.id))
                continue  # start the whole proces over again

            sensing_proc = self.env.process(self.sense_channel(M))
            self.channel.sensing_processes.append(sensing_proc)  # let the channel know that there is a process sensing it
            m = yield sensing_proc
            self.channel.sensing_processes.remove(sensing_proc)
            if m != 0:
                log("{:.0f}:\t {} Channel BUSY - prioritezation period failed.".format(self.env.now, self.id))

    def wait_gap_period(self):
        """Wait gap period"""
        backoff_time = self.N * OBSERVATION_SLOT_DURATION  # time that will be taken for backoff
        time_to_next_sync_slot = self.next_sync_slot_boundry - self.env.now  # calculate time needed for gap

        gap_length = time_to_next_sync_slot - backoff_time
        while gap_length < 0:  # less than 0 means it's impossible to transmit in the next slot because backoff is too long
            gap_length += SYNCHRONIZATION_SLOT_DURATION  # check if possible to transsmit in the slot after the next slot and repeat

        log("{:.0f}:\t {} calculating and waiting the gap period ({:.0f} us)".format(self.env.now, self.id, gap_length))
        if BACKOFF_INSIDE:
            log("{:.0f}:\t {} waiting first half of the gap period ({:.0f} us)".format(self.env.now, self.id, gap_length / 2))
            yield self.env.timeout(gap_length / 2)
            log("{:.0f}:\t {} (re)starting backoff procedure in the middle of the gap (slots to wait: {})".format(self.env.now, self.id, self.N))
            yield self.env.process(self.wait_random_backoff())
            if self.N == 0:
                log("{:.0f}:\t {} waiting second half of the gap period ({:.0f} us)".format(self.env.now, self.id, gap_length / 2))
                yield self.env.timeout(gap_length / 2)
        else:
            yield self.env.timeout(gap_length)

    def wait_random_backoff(self):
        """Wait random number of slots N x OBSERVATION_SLOT_DURATION us"""
        if self.channel.time_until_free() > 0:  # if channel is busy at the start of backoff (e.g. after gap period channel is busy) imidiately return
            return

        if not RS_SIGNALS and GAP_PERIOD == 'during':
            if BACKOFF_SLOTS_SPLIT == 'fixed':
                backoff_slots_left = BACKOFF_SLOTS_TO_LEAVE
            elif BACKOFF_SLOTS_SPLIT == 'variable':
                backoff_slots_left = int(math.ceil(BACKOFF_SLOTS_TO_LEAVE * self.N))

            slots_to_wait = self.N - backoff_slots_left
            slots_to_wait = slots_to_wait if slots_to_wait >= 0 else 0  # if backoff is longer than BACKOFF_SLOTS_TO_LEAVE, count these slots after gap
            log("{:.0f}:\t {} will wait {} slots before stopping backoff".format(self.env.now, self.id, slots_to_wait))
        else:
            slots_to_wait = self.N
        sensing_proc = self.env.process(self.sense_channel(slots_to_wait))
        self.channel.sensing_processes.append(sensing_proc)
        remaining_slots = yield sensing_proc
        self.channel.sensing_processes.remove(sensing_proc)

        if not RS_SIGNALS and GAP_PERIOD == 'during' and remaining_slots == 0:  # redo backoff for additional backoff_slots_left
            log("{:.0f}:\t {} stopping backoff and inserting gap now".format(self.env.now, self.id))
            yield self.env.process(self.wait_gap_period())
            log("{:.0f}:\t {} waiting remaining backoff slots ({}) after gap".format(self.env.now, self.id, self.N - slots_to_wait))
            if self.channel.time_until_free() > 0:  # cca at the beggining of the backoff
                self.N = self.N - slots_to_wait
                return
            sensing_proc = self.env.process(self.sense_channel(self.N - slots_to_wait))
            self.channel.sensing_processes.append(sensing_proc)
            self.N = yield sensing_proc
            self.channel.sensing_processes.remove(sensing_proc)
        elif not RS_SIGNALS and GAP_PERIOD == 'during' and remaining_slots > 0:
            self.N = remaining_slots + self.N - slots_to_wait
        else:
            self.N = remaining_slots

    def run(self):
        """Main process. Genrate new transmission, wait for channel to become idle and begin transmission"""
        while True:
            log("{:.0f}:\t {} begins new transmission procedure".format(self.env.now, self.id))

            self.N = random.randint(0, self.cw)  # draw a random backoff
            log("{:.0f}:\t {} gNB has drawn a random backoff counter = {}".format(self.env.now, self.id, self.N))
            while True:
                yield self.env.process(self.wait_prioritization_period())
                log("{:.0f}:\t {} prioritization period has finished".format(self.env.now, self.id))
                if not RS_SIGNALS and GAP_PERIOD == 'before':  # if RS signals not used, use gap BEFORE backoff procedure
                    yield self.env.process(self.wait_gap_period())
                if not BACKOFF_INSIDE:  # do not wait backoff as it was already waited inside gap
                    log("{:.0f}:\t {} (re)starting backoff procedure (slots to wait: {})".format(self.env.now, self.id, self.N))
                    yield self.env.process(self.wait_random_backoff())
                if self.N == 0:
                    break
                else:
                    log("{:.0f}:\t {} Channel BUSY - backoff is frozen. Remaining slots: {}".format(self.env.now, self.id, self.N))

            if not RS_SIGNALS and GAP_PERIOD == 'after':  # if RS signals not used, use gap AFTER backoff procedure and transmit imidiately
                yield self.env.process(self.wait_gap_period())
            elif not RS_SIGNALS and GAP_PERIOD == 'after_cca':  # transmit after gap AND successful CCA
                yield self.env.process(self.wait_gap_period())
                if self.channel.time_until_free() > 0:
                    log("{:.0f}:\t {} Channel BUSY after gap period, aborting transmission".format(self.env.now, self.id))
                    continue
            elif BACKOFF_INSIDE:
                if self.channel.time_until_free() > 0:
                    log("{:.0f}:\t {} Channel BUSY after gap period, aborting transmission".format(self.env.now, self.id))
                    continue

            if SKIP_NEXT_SLOT_BOUNDRY and self.skip == self.env.now:
                self.skip = None
                log("{:.0f}:\t {} SKIPPING SLOT (will restart transmission procedure after {:.0f} us)".format(self.env.now, self.id, SYNCHRONIZATION_SLOT_DURATION))
                yield self.env.timeout(SYNCHRONIZATION_SLOT_DURATION)
                continue

            if PARTIAL_ENDING_SUBFRAMES:
                last_slot = random.randint(1, 14)
                trans_time = (MCOT * 1e3 - SYNCHRONIZATION_SLOT_DURATION) + (SYNCHRONIZATION_SLOT_DURATION/14) * last_slot
            else:
                trans_time = MCOT * 1e3  # if gap in use = full MCOT to transmit data
            if RS_SIGNALS:
                time_to_next_sync_slot = self.next_sync_slot_boundry - self.env.now  # calculate time needed for reservation signal
                trans_time = (trans_time - time_to_next_sync_slot)  # if RS in use = the rest of MCOT to transmit data
                transmission = Transmission(self.env.now, trans_time, time_to_next_sync_slot)
            else:
                transmission = Transmission(self.env.now, trans_time, 0)

            log("{:.0f}:\t {} is now occupying the channel for the next {:.0f} us (RS={:.0f} us)".format(self.env.now,
                                                                                                         self.id,
                                                                                                         transmission.end - transmission.start,
                                                                                                         transmission.rs_time))
            yield self.env.process(self.channel.transmit(transmission))
            log("{:.0f}:\t {} frees the channel".format(self.env.now, self.id))
            if not transmission.collided:
                self.cw = CW_MIN
                log_success("{:.0f}:\t {} transmission was successful. Current CW={}".format(self.env.now, self.id, self.cw))
                self.successful_trans += 1
                self.succ_airtime += trans_time
                if SKIP_NEXT_SLOT_BOUNDRY:
                    self.skip = self.next_sync_slot_boundry
            else:
                if self.cw < CW_MAX:
                    self.cw = ((self.cw + 1) * 2) - 1
                log_fail("{:.0f}:\t {} transmission resulted in a collision. Current CW={}".format(self.env.now, self.id, self.cw))

            self.total_trans += 1
            self.total_airtime += trans_time


def run_simulation(sim_time, nr_of_gnbs, seed, desyncs=None):
    """Run simulation. Return a list with results."""
    random.seed(seed)

    env = simpy.Environment()
    channel = Channel(env)

    if desyncs is None:
        # ## random desync offsets, but every value is at least MIN_SYNC_SLOT_DESYNC as far from any other value
        desyncs = sample_with_min_distance(MAX_SYNC_SLOT_DESYNC - MIN_SYNC_SLOT_DESYNC, nr_of_gnbs, MIN_SYNC_SLOT_DESYNC)
        # ## random offset from a set (set contains values with step of MIN_SYNC_SLOT_DESYNC)
        # desync_set = list(np.linspace(0, MAX_SYNC_SLOT_DESYNC, num=int(MAX_SYNC_SLOT_DESYNC/MIN_SYNC_SLOT_DESYNC)+1))[:-1]
        # desyncs = [random.choice(desync_set) for _ in range(0, nr_of_gnbs)]
        print(desyncs)

    gnbs = [Gnb(env, 'gNB {}'.format(i), channel, desyncs[i]) for i in range(nr_of_gnbs)]

    env.run(until=(sim_time*1e6))

    results = list()

    for gnb in gnbs:
        results.append({'gnb_id': gnb.id,
                        'successful_transmissions': gnb.successful_trans,
                        'failed_transmissions': gnb.total_trans - gnb.successful_trans,
                        'total_transmissions': gnb.total_trans,
                        'collision_probability': 1 - gnb.successful_trans / gnb.total_trans if gnb.total_trans > 0 else None,
                        'airtime': gnb.total_airtime,
                        'efficient_airtime': gnb.succ_airtime})
    return results


if __name__ == "__main__":

    SIM_TIME = 100

    total_t = 0
    fail_t = 0
    total_airtime = 0
    efficient_airtime = 0

    results = run_simulation(sim_time=SIM_TIME, nr_of_gnbs=2, seed=4)

    for result in results:
        total_airtime += result['airtime']
        total_t += result['total_transmissions']
        fail_t += result['failed_transmissions']
        efficient_airtime += result['efficient_airtime']
    for result in results:
        print("------------------------------------")
        print(result['gnb_id'])
        print('Collsions: {}/{} ({}%)'.format(result['failed_transmissions'],
                                              result['total_transmissions'],
                                              result['collision_probability'] * 100 if result['collision_probability'] is not None else 'N/A'))
        print('Total channel occupancy time: {} ms'.format(result['airtime'] / 1e3))
        print('Normalized airtime: {:.2f}'.format(result['airtime'] / total_airtime))

    print('====================================')
    print('Total colission probablility: {:.4f}'.format(fail_t / total_t))
    print('Total channel efficiency: {:.4f}'.format(efficient_airtime / (SIM_TIME*1e6)))

    # calculate Jain's fairnes index
    sum_sq = 0
    n = len(results)
    for result in results:
        sum_sq += result['efficient_airtime']**2

    jain_index = efficient_airtime**2 / (n * sum_sq)
    print("Jain's fairnes index: {:.4f}".format(jain_index))
