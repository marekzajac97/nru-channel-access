import random
import simpy

DEBUG = False

DETER_PERIOD = 16                     # Time which a node is required to wait at the start of prioritization period in microseconds
OBSERVATION_SLOT_DURATION = 9         # observation slot length in microseconds
SYNCHRONIZATION_SLOT_DURATION = 1000  # synchronization slot length in microseconds
MAX_SYNC_SLOT_DESYNC = 1000           # max random delay between sync slots of each gNB in microseconds (0 to make all gNBs synced)
RS_SIGNALS = True                    # if True use reservation signals before transmission. Use gap otherwise
GAP_PERIOD = 'during'                 # insert backoff 'before', 'during','during' 'after', 'after_cca' backoff procedure.

# set of parameters only applicaple with GAP_PERIOD set to 'during'
BACKOFF_SLOTS_SPLIT = 'variable'      # 'fixed' or 'variable'
BACKOFF_SLOTS_TO_LEAVE = 0.5          # how many slots from the backoff procedure leave to count after the gap. A fixed number of slots or percentage of backoff.

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
    def __init__(self, env, id, channel):
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

        self.env.process(self.sync_slot_counter())
        self.env.process(self.run())

    def sync_slot_counter(self):
        """Process responsible for keeping the next sync slot boundry timestamp"""
        desync = random.randint(0, MAX_SYNC_SLOT_DESYNC)
        self.next_sync_slot_boundry = desync
        log("{:.0f}:\t {} selected random sync slot offset equal to {} us".format(self.env.now, self.id, desync))
        yield self.env.timeout(desync)  # randomly desync tx starting points
        while True:
            self.next_sync_slot_boundry += SYNCHRONIZATION_SLOT_DURATION
            # log_fail("{:.0f}:\t {} SYNC SLOT BOUNDRY NOW".format(self.env.now, self.id))
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
            slots_to_wait -= 1  # simulate propagation delay
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
        yield self.env.timeout(gap_length)

    def wait_random_backoff(self):
        """Wait random number of slots N x OBSERVATION_SLOT_DURATION us"""
        if self.channel.time_until_free() > 0:  # if channel is busy at the start of backoff (e.g. after gap period channel is busy) imidiately return
            return

        if not RS_SIGNALS and GAP_PERIOD == 'during':
            if BACKOFF_SLOTS_SPLIT == 'fixed':
                backoff_slots_to_left = BACKOFF_SLOTS_TO_LEAVE
            elif BACKOFF_SLOTS_SPLIT == 'variable':
                backoff_slots_to_left = round(BACKOFF_SLOTS_TO_LEAVE * self.N)

            slots_to_wait = self.N - backoff_slots_to_left
            slots_to_wait = slots_to_wait if slots_to_wait >= 0 else 0
        else:
            slots_to_wait = self.N
        sensing_proc = self.env.process(self.sense_channel(self.N))
        self.channel.sensing_processes.append(sensing_proc)
        self.N = yield sensing_proc
        self.channel.sensing_processes.remove(sensing_proc)

        if not RS_SIGNALS and GAP_PERIOD == 'during':  # redo backoff for additional backoff_slots_to_left
            log("{:.0f}:\t {} stopping backoff and inserting gap now".format(self.env.now, self.id))
            self.N += backoff_slots_to_left
            yield self.env.process(self.wait_gap_period())
            if self.channel.time_until_free() > 0:
                return
            log("{:.0f}:\t {} waiting remaining backoff slots ({}) after gap".format(self.env.now, self.id, backoff_slots_to_left))
            sensing_proc = self.env.process(self.sense_channel(self.N))
            self.channel.sensing_processes.append(sensing_proc)
            self.N = yield sensing_proc
            self.channel.sensing_processes.remove(sensing_proc)

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

            if RS_SIGNALS:
                time_to_next_sync_slot = self.next_sync_slot_boundry - self.env.now  # calculate time needed for reservation signal
                time_to_next_sync_slot = 0
                trans_time = (MCOT * 1e3 - time_to_next_sync_slot)  # if RS in use = the rest of MCOT to transmit data
                transmission = Transmission(self.env.now, trans_time, time_to_next_sync_slot)
            else:
                trans_time = MCOT * 1e3  # if gap in use = full MCOT to transmit data
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
                log(str(self.succ_airtime))
            else:
                if self.cw < CW_MAX:
                    self.cw = ((self.cw + 1) * 2) - 1
                log_fail("{:.0f}:\t {} transmission resulted in a collision. Current CW={}".format(self.env.now, self.id, self.cw))

            self.total_trans += 1
            self.total_airtime += trans_time


def run_simulation(sim_time, nr_of_gnbs, seed):
    """Run simulation. Return a list with results."""
    random.seed(seed)

    env = simpy.Environment()
    channel = Channel(env)
    gnbs = [Gnb(env, 'gNB {}'.format(i), channel) for i in range(nr_of_gnbs)]

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

    results = run_simulation(sim_time=SIM_TIME, nr_of_gnbs=2, seed=49)

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
        sum_sq += result['airtime']**2

    jain_index = total_airtime**2 / (n * sum_sq)
    print("Jain's fairnes index: {:.4f}".format(jain_index))
