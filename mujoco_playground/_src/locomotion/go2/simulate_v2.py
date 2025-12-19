import os
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from mujoco_playground.config import locomotion_params
from mujoco_playground import registry, MjxEnv
import jax
from mujoco_playground._src.mjx_env import get_sensor_data
import mujoco
import time
import mujoco.viewer
from mujoco import mjx
from typing import Any, Dict, List
from mujoco_playground._src.locomotion.go2 import go2_constants as consts


os.environ['JAX_PLATFORMS'] = 'gpu'
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_disable_jit', False)

def get_obs_sensors(model: mujoco.MjModel, data: mjx.Data) -> List:
	gyro = get_sensor_data(model, data, consts.GYRO_SENSOR)
	resized_matrix = jax.numpy.array(data.site_xmat[model.site("imu").id]).reshape((3, 3))
	gravity = resized_matrix @ jax.numpy.array([0, 0, -1])
	joint_angles = data.qpos[7:]
	joint_velocities = data.qvel[6:]
	linvel = get_sensor_data(model, data, consts.LOCAL_LINVEL_SENSOR)

	return [gyro, gravity, joint_angles, joint_velocities, linvel]


def import_model(env_name='Go2JoystickFlatTerrain', env_cfg=None):
	"""
	This function imports the PPO trained model for the task 'Go2JoystickFlatTerrain'.
	"""
	
	
	ppo_params = locomotion_params.brax_ppo_config(env_name)
	model_params = model.load_params('ppo_go2joystick_flatterrain_params_v1')
	ppo = ppo_networks.make_ppo_networks(action_size=env_cfg.action_size, observation_size=env_cfg.observation_size, **ppo_params.network_factory)
	make_inference = ppo_networks.make_inference_fn(ppo)
	inference_fnTEST = make_inference(model_params)

	return inference_fnTEST

if __name__ == "__main__":
	
	start = time.time()
	rng = jax.random.PRNGKey(0)
	env_name = 'Go2JoystickFlatTerrain'
	env_cfg = registry.get_default_config(env_name)

	m = mujoco.MjModel.from_xml_path('./xmls/scene_mjx_feetonly_flat_terrain.xml')
	d = mujoco.MjData(m)

	inference = import_model(env_name, env_cfg)
	rng = jax.random.PRNGKey(0)
	command = jax.numpy.array([1.0, 0.0, 0.0])  # Command: "Go forward at 1.0 m/s"
	last_action = jax.numpy.zeros(env_cfg.action_size)
	inference = jax.jit(inference)

	with mujoco.viewer.launch_passive(m, d) as viewer:
		# Obtain observations

		mujoco.mj_resetData(m, d)
		while viewer.is_running:
			step_start = time.time()
			act_rng, rng = jax.random.split(rng)
			obs_sensors = get_obs_sensors(m, d)
			obs = jax.numpy.concatenate(obs_sensors + [last_action, command])
			# Inference from command -> angle off set for each of the joints
			ctrl, _ = inference(obs, act_rng)
			motors_targets = m.keyframe("home").qpos[7:] + ctrl * 0.3
			d.ctrl[:] = motors_targets
			number_substeps = round(int(env_cfg.ctrl_dt / env_cfg.sim_dt))
			i = 0
			for _ in range(number_substeps):
				mujoco.mj_step(m, d)
				d.ctrl = motors_targets
				viewer.sync()
				time_until_next_step = env_cfg.ctrl_dt - (time.time() - step_start) / (number_substeps *2)
				# print(time_until_next_step)
				if time_until_next_step > 0:
					time.sleep(time_until_next_step)

			last_action = ctrl


