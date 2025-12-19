import os

import jax
os.environ['JAX_PLATFORMS'] = 'gpu'
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_disable_jit', False)
from mujoco_playground.config import locomotion_params

# print(f"Dispositivos disponibles: {jax.devices()}")

import time

import mujoco
import mujoco.viewer
from mujoco_playground import registry
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from jax import numpy as jp

velocity_kick_range = [0.0, 0.0]  # Disable velocity kick.
kick_duration_range = [0.05, 0.2]

def sample_pert(rng):
  rng, key1, key2 = jax.random.split(rng, 3)
  pert_mag = jax.random.uniform(
      key1, minval=velocity_kick_range[0], maxval=velocity_kick_range[1]
  )
  duration_seconds = jax.random.uniform(
      key2, minval=kick_duration_range[0], maxval=kick_duration_range[1]
  )
  duration_steps = jp.round(duration_seconds / eval_env.dt).astype(jp.int32)
  state.info["pert_mag"] = pert_mag
  state.info["pert_duration"] = duration_steps
  state.info["pert_duration_seconds"] = duration_seconds
  return rng


def key_callback(keycode):
	if keycode == 265:
		print('Up arrow key pressed')
	elif keycode == 263:
		print('Left arrow key pressed')
	elif keycode == 264:
		print('Down arrow key pressed')
	elif keycode == 262:
		print('Right arrow key pressed')

if __name__ == "__main__":
	env_name = 'Go2JoystickFlatTerrain'
	env_cfg = registry.get_default_config(env_name)
	env = registry.load(env_name, config=env_cfg)
	ppo_params = locomotion_params.brax_ppo_config(env_name)
	model_params = model.load_params('ppo_go2joystick_flatterrain_params_v1')
	ppo = ppo_networks.make_ppo_networks(action_size=env_cfg.action_size, observation_size=env_cfg.observation_size, **ppo_params.network_factory)
	make_inference = ppo_networks.make_inference_fn(ppo)
	inference_fnTEST = make_inference(model_params)
	jit_env_reset = jax.jit(env.reset)
	jit_env_step = jax.jit(env.step)
	jit_inference_fn = jax.jit(inference_fnTEST)
	
	m = mujoco.MjModel.from_xml_path('./xmls/scene_mjx_feetonly_flat_terrain.xml')
	d = mujoco.MjData(m)
	  
	eval_env = registry.load(env_name, config=env_cfg)

	with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
		# Close the viewer automatically after 30 wall-seconds.
		start = time.time()
		rng = jax.random.PRNGKey(0)
		state = jit_env_reset(rng)
		state.info["command"] = jp.array([1.0, 0.0, 0.0])  # Initial command: go forward at 1 m/s
		print(f"Initial joint positions: {d.qpos}")
		while viewer.is_running() and time.time() - start < 90:
			step_start = time.time()
			if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
				rng = sample_pert(rng)	

			act_rng, rng = jax.random.split(rng)
			jax.debug.print(" State Obs shape: {obs}", obs=state.obs["state"].shape)
			jax.debug.print("Privileged state shape: {privileged_state_shape}", privileged_state_shape=state.obs["privileged_state"].shape)
			jax.debug.print(" Action:{act_rng}", act_rng=act_rng.shape)
			ctrl, _ = jit_inference_fn(state.obs["state"])
			print("Action: ", ctrl)
			jax.debug.print("State observation: {obs}", obs=state.obs)

			state = jit_env_step(state, ctrl)
			state.info["command"] = jp.array([1.0, 0.0, 0.0])  # Keep command: go forward at 1 m/s
			
			print(f"Joint positions: {d.qpos}")

			# Pick up changes to the physics state, apply perturbations, update options from GUI.
			viewer.sync(state_only=False)