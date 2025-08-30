import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from smac.env import StarCraft2Env
import numpy as np

def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits




def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def get_input_shape(args) :
    input_shape = args.obs_shape
    if args.obs_last_action:
        input_shape += args.n_actions
    if args.obs_agent_id:
        input_shape += args.n_agents

    return input_shape

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    args.input_shape = get_input_shape(args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "alive_allies": {"vshape": (env_info["n_agents"],)},
        "action_all": {"vshape": (args.MT_traj_length, 1), "dtype": th.long, "group": "agents"},
        "obs_all": {"vshape": (args.MT_traj_length, args.input_shape), "group": "agents"},

    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    rvaebuffer = ReplayBuffer(scheme, groups, args.buffer_size_rvae, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    buffer_for_MT = ReplayBuffer(scheme, groups, args.batch_size_run, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)


    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))


        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))


    ##################################################################
    ######################### SRVAE PRETRAINING #########################
    ##################################################################
    fine_tune_cnt = 0
    # fine_tune_step = args.training_interval * buffer.max_seq_length
    fine_tune_step = 0

    if args.use_mae_mode:
        print("PRETRAINING PHASE")
        cnt = 0
        cnt2 = 0
        n_repeat = args.MT_train_n_repeat
        avg_mae_loss = list()
        print(int(args.MT_max_pretraining_episode / args.batch_size_run))
        while cnt <= int(args.MT_max_pretraining_episode / args.batch_size_run):

            with th.no_grad():
                print("PRETRAINING PHASE sampling")

                episode_batch = runner.run(test_mode=False)
                buffer.insert_episode_batch(episode_batch)
                del episode_batch
            # cnt += 1

            # print("number of episodes are ", buffer.episodes_in_buffer)
            # avg_loss = learner.pretrain_srvae(buffer, args.device)

            if buffer.can_sample(args.MT_batchsize):

                episode_sample = buffer.sample(args.MT_batchsize)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                avg_loss = learner.MT_train(episode_sample, runner.t_env, episode, logger=logger,
                                            batch_size=args.MT_batchsize, n_repeat=n_repeat, write_log=True)
                del episode_sample

                if len(avg_mae_loss) < 100:
                    avg_mae_loss.append(avg_loss)
                else:
                    avg_mae_loss[0:99] = avg_mae_loss[1:100]
                    avg_mae_loss[99] = avg_loss
                print('MAE_loss : {}'.format(np.mean(avg_mae_loss).item()))
                if cnt2 % 10 == 0 and args.save_MT:
                    filename = 'results/params/{}_Pretraining_{}.pt'.format(group_name, cnt)
                    learner.save_MT(filename=filename)
                cnt2 += 1
                '''
                if np.mean(avg_mae_loss).item() <= args.MT_pretrain_threshold and args.masking_type == "agent" :
                    cnt = int(args.MT_max_pretraining_episode/args.batch_size_run) + 100
                '''
            cnt += 1

        del buffer
        buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                              preprocess=preprocess,
                              device="cpu" if args.buffer_cpu_only else args.device)

    if args.use_srvae_mode and args.pre_train:

        print("PRETRAINING PHASE")
        cnt = 0
        cnt2 = 0
        n_repeat = args.srvae_train_n_repeat
        avg_mae_loss = list()
        while cnt <= int(args.srvae_max_pretraining_episode / args.batch_size_run):

            with th.no_grad():
                print("PRETRAINING PHASE sampling")

                episode_batch = runner.run(test_mode=False, srvae_train_mode=True)
                buffer.insert_episode_batch(episode_batch)
                # device = episode_batch.device()
                del episode_batch

            cnt+=1

        avg_loss = learner.pretrain_srvae(buffer, args.device)


        del buffer
        buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                              preprocess=preprocess,
                              device="cpu" if args.buffer_cpu_only else args.device)


    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time

        with th.no_grad():
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)
            if args.use_srvae_mode:
                rvaebuffer.insert_episode_batch(episode_batch)
            if args.use_mae_mode:
                buffer_for_MT.insert_episode_batch(episode_batch)
        if buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            if args.learner == "limarl_nq_learner":
                learner.train(episode_sample, runner.t_env, episode, buffer=buffer, rvae_buffer = rvaebuffer )
            else:
                learner.train(episode_sample, runner.t_env, episode)

            del episode_sample

        ##################################################################
        ######################### SRVAE FT #########################
        ##################################################################

        #########################################
        ############ MT FINETUNING ##############

        if args.use_mae_mode and fine_tune_cnt <= args.MT_fine_tune_total and fine_tune_step <= runner.t_env and args.MT_finetune:
            print("#########################################  "
                  "############ MT FINETUNING ############## ")
            fine_tune_step += args.MT_fine_tune_per_step
            fine_tune_cnt += 1
            if args.MT_fine_tune_total * 0.5 <= fine_tune_cnt:
                n_repeat = 1

            # if buffer_for_MT.can_sample(args.vae_batch) :
            #     episode_sample_for_MT = buffer_for_MT.sample(args.vae_batch)

            if buffer_for_MT.can_sample(args.batch_size_run):
                episode_sample_for_MT = buffer_for_MT.sample(args.batch_size_run)

                max_ep_t = episode_sample_for_MT.max_t_filled()
                episode_sample_for_MT = episode_sample_for_MT[:, :max_ep_t]

                if episode_sample_for_MT.device != args.device:
                    episode_sample_for_MT.to(args.device)

                avg_loss = learner.MT_train(episode_sample_for_MT, runner.t_env, episode, logger=logger,
                                            batch_size=args.batch_size_run, n_repeat=2, write_log=True)

                del episode_sample_for_MT

        # # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)

            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
