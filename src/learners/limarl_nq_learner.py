
import copy
from collections import deque

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam, SGD
import numpy as np
from utils.th_utils import get_parameters_num
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR


class SRVAENQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.scheme = scheme

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        # self.params = list(mac.vae.parameters())
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())


        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        rvae_params = list(self.mac.re.parameters()) + list(self.mac.decoder.parameters())
        state_params = list(self.mac.state_pred.parameters()) + list(self.mac.decoder.parameters())

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr)
            self.vae_optimizer = Adam(params=rvae_params, lr=args.lr_obs_vae)
            self.state_optimizer = Adam(params=state_params, lr=args.lr_state_vae)
            # self.state_scheduler = LambdaLR(self.state_optimizer, lr_lambda=lambda epoch: self.lr_lambda(epoch, min_lr=5*1e-4))
            # self.vae_scheduler = LambdaLR(self.vae_optimizer, lr_lambda=lambda epoch: self.lr_lambda(epoch, min_lr=1e-4))

        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.vae_optimizer = RMSprop(params=self.mac.rvae.parameters(), lr=args.lr_obs_vae,
                                         alpha=args.optim_alpha, eps=args.optim_eps)
        self.state_scheduler = LambdaLR(self.state_optimizer,
                                        lr_lambda=lambda epoch: self.lr_lambda(epoch, min_lr= 1e-4))
        self.vae_scheduler = LambdaLR(self.vae_optimizer, lr_lambda=lambda epoch: self.lr_lambda(epoch, min_lr= 1e-4))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # VAE training setup
        self.vae_loss_fn = self._mean_error_loss

        self.state_loss_fn = self._mean_error_loss

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')

        self.condition_met = False

        # Early stopping parameters
        self.patience = 2  # Number of epochs with no improvement before stopping
        self.best_loss = float('inf')  # Best validation loss initialized to infinity
        self.counter = 0  # Counter to track number of epochs without improvement
        self.state_best_loss = float('inf')


        self.finetune_steps = 0

        self.prev_loss = 0
        self.train_steps_state = 0

        self.train_steps = 0
        self.prev_losses = deque(maxlen=5)
        self.best_losses = []
        self.last_retrain = 0
        self.already_trained = False


    def lr_lambda(self,epoch, min_lr=1e-4):
        # Compute the scaling factor for the learning rate
        decay_factor = 0.97  # Example decay factor (similar to your StepLR gamma)
        return max(decay_factor ** (epoch // 1000), min_lr)

    def _mean_error_loss(self, recon_s, s):
        recon_s_loss = F.mse_loss(recon_s, s, reduction='mean')
        loss = recon_s_loss
        return loss

    def kl_divergence(self, mu, log_var):
        # Standard normal distribution (zero mean, unit variance)
        return -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def kl_divergence_between_distributions(self, mu1, log_var1, mu2, log_var2):
        # Compute the KL divergence between two Gaussian distributions
        std1 = th.exp(0.5 * log_var1)
        std2 = th.exp(0.5 * log_var2)

        kl_loss = th.sum(log_var2 - log_var1 + (std1 ** 2 + (mu1 - mu2) ** 2) / (2 * std2 ** 2) - 0.5)
        return kl_loss


    def train_state_pred(self, device, buffer, steps=1):
        """
        Trains the State autoencoder
        """
        # input_shape = self.mac.get_input_shape(self.scheme)
        k = 0

        while buffer.can_sample(self.args.train_batch) and k < steps:
            episode_sample = buffer.sample_epoch(self.args.train_batch)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            # if episode_sample.device != batch.device:
            episode_sample.to(device)

            state = episode_sample["state"].reshape(-1, episode_sample["state"].size()[-1])

            self.state_optimizer.zero_grad()

            mu, logvar = self.mac.state_pred(state)

            if self.args.variational_state:
                z = self.mac.state_pred.reparameterize(mu,logvar)
                recon_state = self.mac.decoder(z)
            # print("mu ", mu.shape)
            else:
                recon_state = self.mac.decoder(mu)  # Guess from z the state space

            loss = self.state_loss_fn(recon_state, state)
            if self.args.variational_state:
                beta = min(self.args.beta_max, self.train_steps_state / float(self.args.kl_warmup_steps))
                loss += beta * (self.kl_divergence(mu, logvar) / mu.shape[0])

            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.mac.state_pred.parameters(), self.args.grad_norm_clip)

            self.state_optimizer.step()

            del episode_sample
            k += 1
            self.train_steps_state +=1

        return loss.item()

    #####################
    def vae_data(self, batch, t, agent_input):
        # Assumes homogenous agents with flat observations.
        bs = batch.batch_size

        # Extract observations and states for time t
        origin_obs = agent_input #batch["obs"][:, t]

        origin_state = batch["state"][:, t]

        origin_state = origin_state.unsqueeze(1).expand(bs, self.args.n_agents, -1)

        origin_state = origin_state.reshape(bs*self.args.n_agents, -1)

        inputs = []
        inputs.append(origin_obs)  # b1av

        inputs = th.cat([x.reshape(-1, origin_obs.size(-1)) for x in inputs], dim=-1)

        outputs = []
        outputs.append(origin_state)  # b1av
        outputs = th.cat([x for x in outputs], dim=-1)

        return inputs, outputs
    #

    def train_rvae(self, device,  buffer, steps=1, t_env= 0):
        """
        Trains the VAE using observations from the batch.
        """
        k = 0
        while buffer.can_sample(self.args.train_batch) and k < steps:
            episode_sample = buffer.sample_epoch(self.args.train_batch)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            # if episode_sample.device != device:
            episode_sample.to(device)

            self.vae_optimizer.zero_grad()
            bs = episode_sample.batch_size
            loss = 0
            kl_loss = 0
            previous_hidden = None  # Reset hidden state per batch
            episode_length = episode_sample.max_seq_length
            total_loss = 0
            for t in range(episode_length):
                agent_input = self.mac.get_vae_inputs(episode_sample, t)
                inputs, states = self.vae_data(episode_sample, t, agent_input)

                # print("masked inputs ", inputs.shape)
                if inputs.shape[0] == 0:
                    continue

                # Get z, mu, logvar from state_pred
                with th.no_grad():
                    mu_state_pred, logvar_state_pred = self.mac.state_pred(states)

                mu, log_var, previous_hidden = self.mac.re(inputs.unsqueeze(1), previous_hidden)

                # Decode the latent representation to reconstruct the input
                z = self.mac.re.reparameterize(mu, log_var)
                # Decode the latent representation to reconstruct the input
                if self.args.variational:
                    recon_state = self.mac.decoder(z)
                else:
                    recon_state = self.mac.decoder(mu)


                # Compute the reconstruction loss
                if self.args.variational:
                    beta = min(self.args.beta_max, self.train_steps / float(self.args.kl_warmup_steps))
                    loss += self.vae_loss_fn(recon_state, states)
                    self.logger.log_stat("beta", beta, t_env)
                    loss += beta * (self.kl_divergence(mu, log_var) / mu.shape[0])
                else:
                    loss += self.vae_loss_fn(recon_state, states)

                kl_with_state_loss = F.mse_loss(mu, mu_state_pred.detach(), reduction='mean')
                kl_loss += self.args.alpha_mu * kl_with_state_loss

            total_loss = loss.item() / episode_length
            (loss + kl_loss).backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.mac.re.parameters(), self.args.grad_norm_clip)
            self.vae_optimizer.step()
            del episode_sample
            k += 1
            self.train_steps += 1

        return total_loss, kl_loss.item() / episode_length


    def test_rvae(self, buffer, training = False):
        """
        Tests the trained VAE using the last batch from the buffer.
        """
        assert buffer.can_sample(self.args.vae_batch), "Not enough data to sample for testing."

        # Sample the last batch
        test_batch = buffer.sample_test_episode(self.args.vae_batch, training=training)
        max_ep_t = test_batch.max_t_filled()
        test_batch = test_batch[:, :max_ep_t]

        if test_batch.device != self.device:
            test_batch.to(self.device)

        bs = test_batch.batch_size
        loss = 0
        kl_loss = 0
        previous_hidden = None  # Reset hidden state per batch
        episode_length = test_batch.max_seq_length
        total_loss = 0
        with th.no_grad():  # No gradients required for testing
            for t in range(episode_length):
                agent_input = self.mac.get_vae_inputs(test_batch, t)
                inputs, states = self.vae_data(test_batch, t, agent_input)

                if inputs.shape[0] == 0:
                    continue

                # Get z, mu, logvar from state_pred
                mu_state_pred, logvar_state_pred = self.mac.state_pred(states)
                mu, log_var, previous_hidden = self.mac.re(inputs.unsqueeze(1), previous_hidden)

                # Decode the latent representation to reconstruct the input
                recon_state = self.mac.decoder(mu)

                # Compute the reconstruction loss
                loss += self.vae_loss_fn(recon_state, states)

                kl_with_state_loss = F.mse_loss(mu, mu_state_pred, reduction='mean')
                kl_loss += self.args.alpha_mu * kl_with_state_loss

        total_loss = loss.item() / episode_length
        kl_loss = kl_loss.item() / episode_length

        # print(f"Test Loss: {total_loss:.4f}, KL Loss: {kl_loss:.4f}")
        print(f"Test Loss: {total_loss:.4f}, KL Loss: {kl_loss:.4f}")

        return total_loss, kl_loss

    def test_state_pred(self, buffer, training=False):
        """
        Tests the state prediction using the last batch from the buffer.
        """
        assert buffer.can_sample(self.args.vae_batch), "Not enough data to sample for testing."

        # Sample the last batch
        test_batch = buffer.sample_test_episode(self.args.vae_batch, training=training)
        max_ep_t = test_batch.max_t_filled()
        test_batch = test_batch[:, :max_ep_t]

        if test_batch.device != self.device:
            test_batch.to(self.device)

        bs = test_batch.batch_size
        loss = 0
        previous_hidden = None  # Reset hidden state per batch
        episode_length = test_batch.max_seq_length

        with th.no_grad():  # No gradients required for testing
            for t in range(episode_length):
                # Get state from the batch and reshape
                state = test_batch["state"].reshape(-1, test_batch["state"].size()[-1])

                # Get predicted mean (mu) and logvar for the state
                mu, logvar = self.mac.state_pred(state)

                # Reconstruct the state using the decoder
                recon_state = self.mac.decoder(mu)

                # Compute the reconstruction loss
                state_loss = self.state_loss_fn(recon_state, state)

                # Accumulate the loss
                loss += state_loss.item()

        # Compute the average loss across the episode length
        avg_loss = loss / episode_length

        print(f"Test State Prediction Loss: {avg_loss:.4f}")
        return avg_loss



    def pretrain_srvae(self, buffer, device):

        self.mac.re.train()
        self.mac.decoder.train()
        self.mac.state_pred.train()
        # Train the state prediction model
        self.mac.decoder.unfreeze_decoder() #CHANGED
        if self.args.train_state_pred:
            for i in range(2):
                number_of_batches_in_epoch = buffer.episodes_in_buffer / self.args.train_batch
                print("number_of_batches_in_epoch ", number_of_batches_in_epoch)

                for j in range(int(number_of_batches_in_epoch)):
                    state_loss = self.train_state_pred(device, buffer)
                    self.state_scheduler.step()  # Update the LR for state optimizer
                    self.logger.log_stat("state_loss", state_loss, i * j )

                state_test_loss = self.test_state_pred(buffer, training=True)

                if state_test_loss < self.state_best_loss:
                    self.state_best_loss = state_test_loss  # Update best loss
                    self.counter = 0  # Reset counter as loss improved
                else:
                    print("best loss, ", self.state_best_loss, "state loss ", state_test_loss)
                    self.counter += 1  # No improvement, increment counter

                # If the counter exceeds patience, stop the training
                if self.counter >= self.patience:
                    print("Early stopping in the state triggered!")
                    break  # End the training early

        # Train the VAE
        self.mac.decoder.freeze_decoder()
        self.counter = 0
        for i in range(2):
            number_of_batches_in_epoch = buffer.episodes_in_buffer / self.args.train_batch
            print("number_of_batches_in_epoch ", number_of_batches_in_epoch)
            for j in range(int(number_of_batches_in_epoch)):
                vae_loss, kl_loss = self.train_rvae(device, buffer)
                self.logger.log_stat("vae_loss", vae_loss, i * j )
                self.logger.log_stat("kl_loss ", kl_loss, i * j )

            self.vae_scheduler.step()  # Update the LR for VAE optimizer

            # Test the model periodically
            # if i % 5 == 0:
            test_loss, kl_test_loss = self.test_rvae(buffer, training=True)
            self.logger.log_stat("test_vae_loss", test_loss, i )
            self.logger.log_stat("test_kl_loss ", kl_test_loss, i)

            # Early stopping condition
            if test_loss + kl_test_loss < self.best_loss:
                self.best_loss = test_loss + kl_test_loss  # Update best loss
                self.counter = 0  # Reset counter as loss improved
            else:
                self.counter += 1  # No improvement, increment counter

            # If the counter exceeds patience, stop the training
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                self.best_loss = float('inf')
                break  # End the training early

        self.best_loss = float('inf')
        self.state_best_loss = float('inf')
        self.target_mac.load_reccurent_state(self.mac)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, buffer=None, rvae_buffer=None):
        # Update the learning rate for fine-tuning

        for param_group in self.state_optimizer.param_groups:
            param_group['lr'] = self.args.finetune_lr  # e.g., 10x smaller than pretrain LR

        for param_group in self.vae_optimizer.param_groups:
            param_group['lr'] = self.args.finetune_lr  # Same here

        # print("t_env is ", t_env)

        # Check the RVAE loss (reconstruction + KL)
        recon_loss, kl_loss = self.test_rvae(rvae_buffer, training=False)
        tot_l = recon_loss + kl_loss
        self.logger.log_stat("rvae_loss_kl_for_shift_detection", tot_l, t_env)

        # In your __init__ (not shown here), add:


        # Compute average of previous losses to check for distribution shift
        if len(self.prev_losses)!=0:
            avg_prev_loss = sum(self.prev_losses) / len(self.prev_losses) if self.prev_losses else float('inf')
            # mean = np.mean(self.prev_losses)
            # std = np.std(self.prev_losses)

        else:
            avg_prev_loss = 0
            # mean = 0
            # std = 0

        # self.logger.log_stat("std", std, t_env)
        # self.logger.log_stat("mean", mean, t_env)


        if recon_loss + kl_loss > (1 + 0.1) * avg_prev_loss: #and not self.already_trained:  # #(mean + std):

            # self.already_trained = True
            # Initialize session best tracker
            session_best_model = None

            self.mac.re.train()
            self.mac.decoder.train()
            self.mac.state_pred.train()

            # Train the state prediction model
            self.mac.decoder.unfreeze_decoder()
            if self.args.train_state_pred:
                for i in range(5):  # Number of epochs
                    if i == 0:
                        session_best_model = self.mac.save_session_best_model()
                        session_best_model['vae_optimizer'] = copy.deepcopy(self.vae_optimizer.state_dict())
                        session_best_model['state_optimizer'] = copy.deepcopy(self.state_optimizer.state_dict())

                    number_of_batches_in_epoch = rvae_buffer.episodes_in_buffer / self.args.train_batch
                    print("number_of_batches_in_epoch ", number_of_batches_in_epoch)

                    for j in range(int(number_of_batches_in_epoch)):
                        state_loss = self.train_state_pred(self.args.device, rvae_buffer)
                        self.state_scheduler.step()
                        self.logger.log_stat("state_loss", state_loss, i * j * t_env)

                    state_test_loss = self.test_state_pred(rvae_buffer, training=True)

                    if state_test_loss < self.state_best_loss:
                        self.state_best_loss = state_test_loss
                        self.counter = 0
                        session_best_model = self.mac.save_session_best_model()
                        session_best_model['vae_optimizer'] = copy.deepcopy(self.vae_optimizer.state_dict())
                        session_best_model['state_optimizer'] = copy.deepcopy(self.state_optimizer.state_dict())
                        print("Saved new session-best model at t_env =", t_env)
                    else:
                        print("best loss, ", self.state_best_loss, "state loss ", state_test_loss)
                        self.counter += 1

                    if self.counter >= self.patience:
                        print("Early stopping in the state triggered!")
                        if session_best_model is not None:
                            self.mac.agent.load_state_dict(session_best_model['mac'])
                            self.mac.re.load_state_dict(session_best_model['re'])
                            self.mac.decoder.load_state_dict(session_best_model['decoder'])
                            self.mac.state_pred.load_state_dict(session_best_model['state_pred'])
                            self.vae_optimizer.load_state_dict(session_best_model['vae_optimizer'])
                            self.state_optimizer.load_state_dict(session_best_model['state_optimizer'])
                            print("Restored and saved the session-best model before early stopping.")
                        break

            # Train the VAE
            self.mac.decoder.freeze_decoder()
            self.counter = 0
            session_best_model = None
            for i in range(5):  # Number of epochs
                if i == 0:
                    session_best_model = self.mac.save_session_best_model()
                    session_best_model['vae_optimizer'] = copy.deepcopy(self.vae_optimizer.state_dict())
                    session_best_model['state_optimizer'] = copy.deepcopy(self.state_optimizer.state_dict())

                number_of_batches_in_epoch = rvae_buffer.episodes_in_buffer / self.args.train_batch
                print("number_of_batches_in_epoch ", number_of_batches_in_epoch)

                for j in range(int(number_of_batches_in_epoch)):
                    vae_loss, kl_loss = self.train_rvae(self.args.device, rvae_buffer, t_env= t_env)
                    self.logger.log_stat("vae_loss", vae_loss, i * j * t_env)
                    self.logger.log_stat("kl_loss ", kl_loss, i * j * t_env)
                    self.vae_scheduler.step()

                test_loss, kl_test_loss = self.test_rvae(rvae_buffer, training=True)
                current_loss = test_loss + kl_test_loss
                print("current_loss ", current_loss)
                self.logger.log_stat("test_vae_loss", test_loss, i * t_env)
                self.logger.log_stat("test_kl_loss ", kl_test_loss, i * t_env)

                # Update best_losses list and average
                self.best_losses.append(current_loss)
                self.best_losses = sorted(self.best_losses)[:5]
                avg_best_loss = sum(self.best_losses) / len(self.best_losses)

                if current_loss <= avg_best_loss:
                    self.counter = 0
                    session_best_model = self.mac.save_session_best_model()
                    session_best_model['vae_optimizer'] = copy.deepcopy(self.vae_optimizer.state_dict())
                    session_best_model['state_optimizer'] = copy.deepcopy(self.state_optimizer.state_dict())
                    print("Saved new session-best model at t_env =", t_env)

                else:
                    self.counter += 1

                if self.counter >= self.patience:
                    print("Early stopping triggered!")
                    if session_best_model is not None:
                        self.mac.agent.load_state_dict(session_best_model['mac'])
                        self.mac.re.load_state_dict(session_best_model['re'])
                        self.mac.decoder.load_state_dict(session_best_model['decoder'])
                        self.mac.state_pred.load_state_dict(session_best_model['state_pred'])
                        self.vae_optimizer.load_state_dict(session_best_model['vae_optimizer'])
                        self.state_optimizer.load_state_dict(session_best_model['state_optimizer'])
                        print("Restored and saved the session-best model before early stopping.")
                    break

                # Update the prev_losses for detecting future distribution shift
                self.prev_losses.append(current_loss)

            # Load recurrent state to target_mac after fine-tuning
            self.target_mac.load_reccurent_state(self.mac)
            self.last_retrain = t_env
            # Update fine-tune step count
        else:
            print("No distribution shift detected. Skipping training to avoid overfitting.")


        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # print("STATE shape ", batch["state"].shape)
        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t, t_env = t_env)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_ = chosen_action_qvals

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.load_reccurent_state(self.mac)
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t, t_env = t_env)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                                 self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                                  self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        loss = L_td = masked_td_error.sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # if state_loss != None:
            #     self.logger.log_stat("state_loss", state_loss, t_env)
            #     self.logger.log_stat("vae_loss", vae_loss, t_env)
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)
            if self.args.env == "three_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                        / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                         / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


