import random
import simpy


RANDOM_SEED = 42
TRANSMITION_INTERVAL = 2 # Mean time between successive transmitions
TRANSMITION_TIME = 2 	 # Mean tranmission time
SIM_TIME = 20            # Simulation time in minutes
UE_NUMBER = 2            # Number of UEs/nodes
DETER_PERIOD = 1         # Time which a node is required to wait before transmiting after sensing the channel to be idle
MAX_RETRIES = 4          # Max number of attempts for starting a new transmission (this is NOT retransmission limit!)

TOTAL_ATTEMPTS = 0
SUCCESSFUL_ATTEMPTS = 0

class Transmission(object):
	def __init__(self, start, end):
		self.start = start
		self.end = end
		self.collided = False

class Channel(object):
	def __init__(self, env):
		self.ongoing_transmisions = []
		self.env = env

	def occupy(self, transmission):
		self.ongoing_transmisions.append(transmission)

	def free(self, transmission):
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
			if self.env.now != t.start:  # don't count transmissions which had just started (workaround for case when a few nodes begin transmission at the same time. One will always start before the others :/)
				time_left = t.end - self.env.now
				if time_left > max_time:
					max_time = time_left
		return max_time


class Ue(object):
	def __init__(self, env, id, channel):
		self.env = env
		self.channel = channel
		self.id = id

	def transmit(self, transmission, time):
		"""Start occupying the channel for the transmission's duration and check for collison"""
		print("{:.2f}:\t {} is now occupying the channel for the next {:.2f}".format(self.env.now, self.id, time))
		self.channel.occupy(transmission)
		yield self.env.timeout(time)
		self.channel.free(transmission)
		print("{:.2f}:\t {} frees the channel".format(self.env.now, self.id))
		self.channel.check_collision(transmission)

	def run(self):
		"""Main process. Genrate new transmission, wait for channel to become idle and begin transmission"""
		global TOTAL_ATTEMPTS
		global SUCCESSFUL_ATTEMPTS

		while(True):
			interval_time = random.expovariate(1.0 / TRANSMITION_INTERVAL)
			yield env.timeout(interval_time)
			print("{:.2f}:\t {} begins new transmisson procedure after waiting {:.2f}".format(self.env.now, self.id, interval_time))
			trans_time = random.expovariate(1.0 / TRANSMITION_TIME)

			waiting_time = self.channel.time_until_free()
			retries = 0

			while(retries <= MAX_RETRIES):
				while(waiting_time != 0):
					print("{:.2f}:\t {} is sensing channel busy (for at least {:.2f})".format(self.env.now, self.id, waiting_time))
					yield self.env.timeout(waiting_time)
					waiting_time = self.channel.time_until_free()
				print("{:.2f}:\t {} Channel is now idle, waiting the duration of the deter period ({:.2f})".format(self.env.now, self.id, DETER_PERIOD))
				yield self.env.timeout(DETER_PERIOD)
				waiting_time = self.channel.time_until_free()
				if waiting_time == 0:
					print("{:.2f}:\t {} Channel has been idle for the duration of the deter period ({:.2f})".format(self.env.now, self.id, DETER_PERIOD))
					break     # proceed to the transmission
				else:
					print("{:.2f}:\t {} Channel has become busy again for the duration of the deter period ({:.2f})".format(self.env.now, self.id, DETER_PERIOD))
					retries += 1
					continue  # start the whole proces over again

			if retries == MAX_RETRIES:
				print("{:.2f}:\t {} Retry limit reached, aborting transmission".format(self.env.now, self.id))
				TOTAL_ATTEMPTS += 1
				continue

			transmission = Transmission(self.env.now, self.env.now + trans_time)
			yield self.env.process(self.transmit(transmission, trans_time))
			if not transmission.collided:
				print("{:.2f}:\t {} transmission was successful".format(self.env.now, self.id))
				SUCCESSFUL_ATTEMPTS += 1
				TOTAL_ATTEMPTS += 1
			else:
				print("{:.2f}:\t {} transmission resulted in a collision".format(self.env.now, self.id))
				TOTAL_ATTEMPTS += 1
			

random.seed(RANDOM_SEED)

env = simpy.Environment()
channel = Channel(env) # simpy.Resource(env, 1)

for i in range(UE_NUMBER):
	ue = Ue(env, 'UE {}'.format(i), channel)
	env.process(ue.run())

env.run(until=SIM_TIME)
print('\nColission probablility: {:.2f}'.format(1 - (SUCCESSFUL_ATTEMPTS / TOTAL_ATTEMPTS)))