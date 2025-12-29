"Synchronous simulation of the Go2 robot walking task using a trained PPO policy."

import os
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from mujoco_playground.config import locomotion_params
from mujoco_playground import registry
import jax
from mujoco_playground._src.mjx_env import get_sensor_data
import mujoco
import time
import mujoco.viewer
import logging
from mujoco import mjx
from typing import List
from mujoco_playground._src.locomotion.go2 import go2_constants as consts



os.environ['JAX_PLATFORMS'] = 'gpu'
jax.config.update('jax_platform_name', 'gpu')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
jax.config.update('jax_disable_jit', False)
# Remove jax debug logging
logging.getLogger('jax').setLevel(logging.WARNING)

def get_obs_sensors(model: mujoco.MjModel, data: mujoco.MjData) -> List:
	"""
	Gets the observation sensors for the Go2 robot

	Args:
		model (mujoco.MjModel): The Mujoco model
		data (mujoco.MjData): The Mujoco data
	Returns:
		List: A list of observation sensors
	"""
	linvel = get_sensor_data(model, data, consts.LOCAL_LINVEL_SENSOR)
	gyro = get_sensor_data(model, data, consts.GYRO_SENSOR)
	resized_matrix = jax.numpy.array(data.site_xmat[model.site("imu").id]).reshape((3, 3))
	gravity = resized_matrix @ jax.numpy.array([0, 0, -1])
	joint_angles = data.qpos[7:] - model.keyframe("home").qpos[7:]
	joint_velocities = data.qvel[6:]

	return [linvel, gyro, gravity, joint_angles, joint_velocities]



def import_model(env_name='Go2JoystickFlatTerrain', env_cfg=None):
	"""
	This function imports the PPO trained model for the task 'Go2JoystickFlatTerrain'.

	Args:
		env_name (str): The name of the environment. Default is 'Go2JoystickFlatTerrain'.
		env_cfg: The environment configuration.

	Returns:
		inference_fnTEST: The inference function for the trained model.
	"""
	
	
	ppo_params = locomotion_params.brax_ppo_config(env_name)
	model_params = model.load_params('ppo_go2joystick_flatterrain_params_v1_ctrl_002_sim_0004_impratio_100')
	ppo = ppo_networks.make_ppo_networks(action_size=env_cfg.action_size, observation_size=env_cfg.observation_size, **ppo_params.network_factory)
	make_inference = ppo_networks.make_inference_fn(ppo)
	inference_fnTEST = make_inference(model_params, deterministic=True)

	return inference_fnTEST

if __name__ == "__main__":
	rng = jax.random.PRNGKey(0)
	env_name = 'Go2JoystickFlatTerrain'
	env_cfg = registry.get_default_config(env_name)

	m = mujoco.MjModel.from_xml_path('./xmls/scene_mjx_feetonly_flat_terrain.xml')
	d = mujoco.MjData(m)

	inference = import_model(env_name, env_cfg)
	last_action = jax.numpy.zeros(env_cfg.action_size)
	inference = jax.jit(inference)
	counter_control = 0
	counter_init = 0
	command = jax.numpy.array([0.0, 0.0, 0])

	def on_key_release(key):
		"""
		Callback function for key release events in the Mujoco viewer. 
		This function updates the global command variable based on the arrow
		keys released or spacebar.

		Args:
			key: int
				The key code of the released key.

		Returns:
			None

		"""
		global command
		if key == 265:
			print("Up arrow released")
			command = jax.numpy.array([1.0, 0.0, 0])
		

		elif key == 262:
			print("Right arrow released")
			command = jax.numpy.array([0.0, 1.0, 0])

		elif key == 264:
			print("Down arrow released")
			command = jax.numpy.array([-1.0, 0.0, 0])

		elif key == 263:
			print("Left arrow released")
			command = jax.numpy.array([0.0, -1.0, 0])

		elif key == 32:
			print("Spacebar released")
			command = jax.numpy.array([0.0, 0.0, 0.0])

	with mujoco.viewer.launch_passive(m, d, key_callback=on_key_release) as viewer:
		mujoco.mj_resetData(m, d)
		timer_control = time.time()
		motors_targets = m.keyframe("home").qpos[7:]
		while viewer.is_running:
			step_start = time.time()

			if (counter_control % int(env_cfg.ctrl_dt / env_cfg.sim_dt)) == 0:
				act_rng, rng = jax.random.split(rng)
				obs_sensors = get_obs_sensors(m, d)
				obs = jax.numpy.concatenate(obs_sensors + [last_action, command])
				ctrl, _ = inference(obs, act_rng)
				motors_targets = m.keyframe("home").qpos[7:] + ctrl * env_cfg.action_scale
				timer_control = time.time()
				last_action = ctrl
			
			d.ctrl = motors_targets	

			time_until_next_step = env_cfg.sim_dt - (time.time() - step_start)

			mujoco.mj_step(m, d)
			viewer.sync()
			counter_control += 1

