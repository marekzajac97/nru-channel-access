import random
import simpy


RANDOM_SEED = 42
TRANSMITION_INTERVAL = 2 # Mean time between successive transmitions
TRANSMITION_TIME = 2 	 # Mean tranmission time
SIM_TIME = 20         # Simulation time in minutes
UE_NUMBER = 3

TOTAL_ATTEMPTS = 0
SUCCESSFUL_ATTEMPTS = 0

class Transmission(object):
	def __init__(self, start, end):
		self.start = start
		self.end = end
		self.collided = False

class Channel(object):
	def __init__(self):
		self.ongoing_transmisions = []

	def occupy(self, transmission):
		self.ongoing_transmisions.append(transmission)

	def free(self, transmission):
		self.ongoing_transmisions.remove(transmission)

	def check_collision(self, transmission):
		for t in self.ongoing_transmisions:
			if t.end > transmission.start and t.start < transmission.end:
			 	transmission.collided = True
			 	t.collided = True

class Ue(object):
	def __init__(self, env, id, channel):
		self.env = env
		self.channel = channel
		self.id = id

	def transmit(self, transmission, time):
		print("{:.2f}:\t {} is now occupying the channel for the next {:.2f}".format(self.env.now, self.id, time))
		self.channel.occupy(transmission)
		yield self.env.timeout(time)
		self.channel.free(transmission)
		print("{:.2f}:\t {} frees the channel".format(self.env.now, self.id))
		self.channel.check_collision(transmission)

	def run(self):
		global TOTAL_ATTEMPTS
		global SUCCESSFUL_ATTEMPTS

		while(True):
			interval_time = random.expovariate(1.0 / TRANSMITION_INTERVAL)
			yield env.timeout(interval_time)
			print("{:.2f}:\t {} starts a new transmission after waiting {:.2f}".format(self.env.now, self.id, interval_time))
			trans_time = random.expovariate(1.0 / TRANSMITION_TIME)
			transmission = Transmission(self.env.now, self.env.now + trans_time)
			yield self.env.process(self.transmit(transmission, trans_time))
			TOTAL_ATTEMPTS += 1
			if not transmission.collided:
				print("{:.2f}:\t {} transmission was successful".format(self.env.now, self.id))
				SUCCESSFUL_ATTEMPTS += 1
			else:
				print("{:.2f}:\t {} transmission resulted in a collision".format(self.env.now, self.id))
			

random.seed(RANDOM_SEED)

env = simpy.Environment()
channel = Channel() # simpy.Resource(env, 1)

for i in range(UE_NUMBER):
	ue = Ue(env, 'UE {}'.format(i), channel)
	env.process(ue.run())

env.run(until=SIM_TIME)
print('\nColission probablility: {:.2f}'.format(1 - (SUCCESSFUL_ATTEMPTS / TOTAL_ATTEMPTS)))