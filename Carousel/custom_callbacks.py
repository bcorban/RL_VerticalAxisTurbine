from typing import Dict, Tuple
import argparse
import numpy as np
import wandb
import math 
import os
import ray
from datetime import datetime
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

class CustomCallbacks(DefaultCallbacks):
	
	def on_algorithm_init(self, *, algorithm, **kwargs):
		# wandb server 
		wandb.init(project='RL_VAWT')
		now = datetime.now()
		name = now.strftime("_%m_%d_%Y_%H_%M_%S")
		wandb.run.name='SAC'+name
		# monitoring 
		self.time=0

	# def on_episode_start(
	# 	self,
	# 	*,
	# 	worker: RolloutWorker,
	# 	base_env: BaseEnv,
	# 	policies: Dict[str, Policy],
	# 	episode: Episode,
	# 	env_index: int,
	# 	**kwargs
	# ):
	# 	# Make sure this episode has just been started (only initial obs
	# 	# logged so far).
	# 	assert episode.length == 0, (
	# 		"ERROR: `on_episode_start()` callback should be called right "
	# 		"after env reset!"
	# 	)
	

	# def on_episode_step(
	# 	self,
	# 	*,
	# 	worker: RolloutWorker,
	# 	base_env: BaseEnv,
	# 	policies: Dict[str, Policy],
	# 	episode: Episode,
	# 	env_index: int,
	# 	**kwargs
	# ):
	# 	# Make sure this episode is ongoing.
	# 	assert episode.length > 0, (
	# 		"ERROR: `on_episode_step()` callback should not be called right "
	# 		"after env reset!"
	# 	)
	

	# def on_episode_end(
	# 	self,
	# 	*,
	# 	worker: RolloutWorker,
	# 	base_env: BaseEnv,
	# 	policies: Dict[str, Policy],
	# 	episode: Episode,
	# 	env_index: int,
	# 	**kwargs
	# ):
	# 	# Check if there are multiple episodes in a batch, i.e.
	# 	# "batch_mode": "truncate_episodes".
	# 	if worker.policy_config["batch_mode"] == "truncate_episodes":
	# 		# Make sure this episode is really done.
	# 		assert episode.batch_builder.policy_collectors["default_policy"].batches[
	# 			-1
	# 		]["dones"][-1], (
	# 			"ERROR: `on_episode_end()` should only be called "
	# 			"after episode is done!"
	# 		)
	# 	print('Total reward episode : ', episode.total_reward)
	

	# def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
	# 	print("returned sample batch of size {}".format(samples.count))

	# def on_train_result(self, *, algorithm, result: dict, **kwargs):
	# 	print(
	# 		"Algorithm.train() result: {} -> {} episodes".format(
	# 			algorithm, result["episodes_this_iter"]
	# 		)
	# 	)
	

	# def on_learn_on_batch(
	# 	self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
	# ) -> None:
	# 	result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
	# 	print(
	# 		"policy.learn_on_batch() result: {} -> sum actions: {}".format(
	# 			policy, result["sum_actions_in_train_batch"]
	# 		)
	# 	)

	# def on_postprocess_trajectory(
	# 	self,
	# 	*,
	# 	worker: RolloutWorker,
	# 	episode: Episode,
	# 	agent_id: str,
	# 	policy_id: str,
	# 	policies: Dict[str, Policy],
	# 	postprocessed_batch: SampleBatch,
	# 	original_batches: Dict[str, Tuple[Policy, SampleBatch]],
	# 	**kwargs
	# ):
	# 	print("postprocessed {} steps".format(postprocessed_batch.count))
	# 	if "num_batches" not in episode.custom_metrics:
	# 		episode.custom_metrics["num_batches"] = 0
	# 	episode.custom_metrics["num_batches"] += 1


	def on_train_result(self,*,algorithm,result: dict,**kwargs,) -> None:
		"""Called at the end of Algorithm.train().

		Args:
			algorithm: Current Algorithm instance.
			result: Dict of results returned from Algorithm.train() call.
				You can mutate this object to add additional metrics.
			kwargs: Forward compatibility placeholder.
		"""
		num_healthy_workers=result['num_healthy_workers']
		# wandb
		data={'time': self.time}
		recursif_wandb(result['info'],data)
		recursif_wandb(result['sampler_results'], data)
		# time 
		self.time+=1
		wandb.log(data)
		

def recursif_dict(d,k=0):
	if isinstance(d,dict) :
		for key in d.keys():
			print(k*str('    ')+key) if isinstance(d[key],dict) else print(k*str('   ')+key+' : '+str(d[key]))
			recursif_dict(d[key],k+1)

def recursif_wandb(d,o,k=0):
	if isinstance(d,dict) :
		for key in d.keys():
			if isinstance(d[key],float):
				o[key] = d[key]  
			recursif_wandb(d[key],o,k+1)