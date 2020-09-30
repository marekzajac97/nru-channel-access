import random
import simpy


RANDOM_SEED = 42
TRANSMITION_TIME = 8 	 # Mean tranmission time in miliseconds
SIM_TIME = 100           # Simulation time in seconds
GNB_NUMBER = 2           # Number of gNBs/nodes

DETER_PERIOD = 16        # Time which a node is required to wait at the start of prioritization period (16 us)
SLOT_DURATION = 9   	 # observation slot length in nanoseconds

# Channel access class 1
# M = 1 					 # fixed number of observation slots in prioritization period
# CW_MIN = 3				 # minimum contention window size
# CW_MAX = 7			     # maximum

# Channel access class 2
# M = 1
# CW_MIN = 7
# CW_MAX = 15

# Channel access class 3
# M = 2
# CW_MIN = 15
# CW_MAX = 63

# Channel access class 4
M = 7
CW_MIN = 15
CW_MAX = 1023


def log(output):
	pass
	# print(output)

def log_fail(output):
	pass
	# print("\033[91m" + output + "\033[0m")

def log_success(output):
	pass
	# print("\033[92m" + output + "\033[0m")

class Transmission(object):
	def __init__(self, start, end, airtime):
		self.start = start
		self.end = end
		self.airtime = airtime
		self.collided = False

class Channel(object):
	def __init__(self, env):
		self.ongoing_transmisions = []
		self.sensing_processes = []
		self.env = env

	def occupy(self, transmission):
		self.ongoing_transmisions.append(transmission)
		for process in self.sensing_processes:
			if process.is_alive:
				process.interrupt()
		yield self.env.timeout(transmission.airtime)
		self.ongoing_transmisions.remove(transmission)

	def check_collision(self, transmission):
		"""Check if a given transmission start/end times overlapped with any of the ongoing transmissions"""
		for t in self.ongoing_transmisions:
			if t.end > transmission.start and t.start < transmission.end:
			 	transmission.collided = True
			 	t.collided = True

	def time_until_free(self):
		"""Return time left for the channel to become idle"""
		max_time = 0
		for t in self.ongoing_transmisions:
			# if self.env.now != t.start:  # don't count transmissions which had just started (workaround for case when a few nodes begin transmission at the same time. One will always start before the others :/)
			time_left = t.end - self.env.now
			if time_left > max_time:
				max_time = time_left
		return max_time


class Gnb(object):
	def __init__(self, env, id, channel):
		self.env = env
		self.channel = channel
		self.id = id
		self.cw = CW_MIN			# current contention window size
		self.N = None				# backoff counter
		self.successful_trans = 0	# number of successful transmissions
		self.total_trans = 0		# total number of transmissions

		self.env.process(self.run())

	def transmit(self, transmission):
		"""Start occupying the channel for the transmission's duration and check for collison"""
		log("{:.2f}:\t {} is now occupying the channel for the next {:.2f}".format(self.env.now, self.id, transmission.airtime))
		yield self.env.process(self.channel.occupy(transmission))
		log("{:.2f}:\t {} frees the channel".format(self.env.now, self.id))
		self.channel.check_collision(transmission)

	def wait_for_idle_channel(self):
		"""Wait until the channel is sensed idle"""
		waiting_time = self.channel.time_until_free()
		while waiting_time != 0:
			log("{:.2f}:\t {} is sensing channel busy (for at least {:.2f})".format(self.env.now, self.id, waiting_time))
			yield self.env.timeout(waiting_time)
			waiting_time = self.channel.time_until_free() # in case a new transmission started check again

	def sense_channel(self, slots_to_wait):
		"""Wait for the duration of slots_to_wait x SLOT_DURATION. Return remaining slots (0 if procedure was successful)"""
		try:
			while slots_to_wait > 0:
				yield self.env.timeout(SLOT_DURATION)
				slots_to_wait -= 1
				log("{:.2f}:\t {} Channel idle for the duration of a single observation slot, remaining slots: {}".format(self.env.now, self.id, slots_to_wait))
		except simpy.Interrupt:
			log("{:.2f}:\t {} Channel sensed busy during observation slot, remaining slots: {}".format(self.env.now, self.id, slots_to_wait))
		return slots_to_wait

	def wait_prioritization_period(self):
		"""Wait initial 16 us + m x SLOT_DURATION us"""
		m = M
		while m > 0:  # if m == 0 continue to backoff
			yield self.env.process(self.wait_for_idle_channel())
			log("{:.2f}:\t {} Channel is now idle, waiting the duration of the deter period ({:.2f})".format(self.env.now, self.id, DETER_PERIOD))
			yield self.env.timeout(DETER_PERIOD)
			if self.channel.time_until_free() == 0:
				log("{:.2f}:\t {} Checking the channel after deter period: IDLE - wait {} observation slots".format(self.env.now, self.id, M))
			else:
				log("{:.2f}:\t {} Checking the channel after deter period: BUSY - wait for idle channel".format(self.env.now, self.id))
				continue  # start the whole proces over again

			sensing_proc = self.env.process(self.sense_channel(M))
			self.channel.sensing_processes.append(sensing_proc)  # let the channel know that there is a process sensing it
			m = yield sensing_proc
			self.channel.sensing_processes.remove(sensing_proc)
	
	def wait_random_backoff(self):
		"""Wait random number of slots N x SLOT_DURATION us"""
		sensing_proc = self.env.process(self.sense_channel(self.N))
		self.channel.sensing_processes.append(sensing_proc)
		self.N = yield sensing_proc
		self.channel.sensing_processes.remove(sensing_proc)


	def run(self):
		"""Main process. Genrate new transmission, wait for channel to become idle and begin transmission"""
		while True:
			log("{:.2f}:\t {} begins new transmisson procedure".format(self.env.now, self.id))
			
			self.N = random.randint(0, self.cw)  # draw a random backoff
			log("{:.2f}:\t {} gNB has drawn a random backoff counter = {}".format(self.env.now, self.id, self.N))
			while self.N > 0:
				yield self.env.process(self.wait_prioritization_period())
				log("{:.2f}:\t {} prioritization period has finnished".format(self.env.now, self.id, self.N))
				yield self.env.process(self.wait_random_backoff())
				if self.N != 0:
					log("{:.2f}:\t {} backoff is frozen".format(self.env.now, self.id))

			trans_time = random.expovariate(1.0 / TRANSMITION_TIME) * 10e6 # TODO take MCOT into account
			transmission = Transmission(self.env.now, self.env.now + trans_time, trans_time)
			yield self.env.process(self.transmit(transmission))
			if not transmission.collided:
				self.cw = CW_MIN
				log_success("{:.2f}:\t {} transmission was successful. Current CW={}".format(self.env.now, self.id, self.cw))
				self.successful_trans += 1
				self.total_trans += 1
			else:
				if self.cw < CW_MAX:
					self.cw = ((self.cw + 1) * 2) - 1
				log_fail("{:.2f}:\t {} transmission resulted in a collision. Current CW={}".format(self.env.now, self.id, self.cw))
				self.total_trans += 1
			

random.seed(RANDOM_SEED)

env = simpy.Environment()
channel = Channel(env)
gnbs = [Gnb(env, 'gNB {}'.format(i), channel) for i in range(GNB_NUMBER)]

env.run(until=(SIM_TIME*10e9))

total_transmissions = 0
successful_transmissions = 0

for gnb in gnbs:
	print("------------------------------------")
	print(gnb.id)
	print('Collsions: {}/{} ({:.2f}%)'.format(gnb.total_trans - gnb.successful_trans,
		                                      gnb.total_trans,
		                                      (1 - gnb.successful_trans / gnb.total_trans) * 100))
	total_transmissions += gnb.total_trans
	successful_transmissions += gnb.successful_trans

print('====================================')
print('\nMean colission probablility: {:.4f}'.format(1 - (successful_transmissions / total_transmissions)))