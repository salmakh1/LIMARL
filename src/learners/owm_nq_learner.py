import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
import torch.nn.functional as F

from modules.vae.vae_agent import VAE


class OWM_NQLearner:
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

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr)
            self.vae_optimizer = Adam(params=self.mac.vae.parameters(), lr=args.lr)
            self.rnn_optimizer = Adam(params=self.mac.rnn.parameters(), lr=args.lr)


        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.vae_optimizer = RMSprop(params=self.mac.vae.parameters(), lr=args.lr,
                                         alpha=args.optim_alpha, eps=args.optim_eps)

            self.rnn_optimizer = RMSprop(params=self.mac.rnn.parameters(), lr=args.lr,
                                         alpha=args.optim_alpha, eps=args.optim_eps)
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # VAE training setup
        self.vae_loss_fn = self._vae_loss

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')

    def _vae_loss(self, recon_x, x, mu, logvar, recon_s=None, s=None):

        recon_x_loss = F.mse_loss(recon_x, x, reduction='mean')
        # recon_s_loss = F.mse_loss(recon_s, s, reduction='mean')

        kl_loss = -0.5 * th.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_x_loss + 0.1 * kl_loss #+ recon_s_loss

        return loss, recon_x_loss, kl_loss#, recon_s_loss

    def vae_data(self, batch):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size

        outputs = []
        outputs.append(batch["state"])  # b1av
        outputs = th.cat([x for x in outputs], dim=-1)

        inputs = []
        inputs.append(batch["obs"])  # b1av
        inputs = th.cat([x.reshape(bs, self.args.n_agents, -1) for x in inputs], dim=-1)

        return inputs, outputs

    def train_vae(self, batch, buffer, steps=1):
        """
        Trains the VAE using observations from the batch.
        """
        # input_shape = self.mac.get_input_shape(self.scheme)
        t = 0
        all_z = []
        all_states = []
        while buffer.can_sample(batch.batch_size) and t < steps:
            episode_sample = buffer.sample(batch.batch_size)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            if episode_sample.device != batch.device:
                episode_sample.to(batch.device)

            input_shape = self.scheme["obs"]["vshape"]
            output_shape = self.scheme["state"]["vshape"]

            inputs, _ = self.vae_data(episode_sample)

            input = inputs.view(-1, input_shape)
            # state = states.unsqueeze(1).expand(batch.batch_size, self.args.n_agents, -1, -1)
            # state = state.view(-1, output_shape)

            self.vae_optimizer.zero_grad()

            z, mu, logvar = self.mac.vae(input)

            all_z.append(z.reshape(episode_sample.batch_size, episode_sample['actions_onehot'].shape[1], self.args.n_agents, -1))
            all_states.append(episode_sample["state"].unsqueeze(2).expand(-1,-1,self.args.n_agents,-1))

            recon_obs = self.mac.vae.decode(z)  # Decode z back to observation space

            # recon_state = self.mac.vae.guess(z)  # Guess from z the state space

            # print(recon_state.shape)
            # print(state.shape)

            loss, recon_obs_loss, kl_loss = self.vae_loss_fn(recon_obs, input, mu, logvar)
            # loss, recon_obs_loss, recon_s_loss, kl_loss  = self.vae_loss_fn(recon_obs, input, mu, logvar, recon_state, state)

            # loss.backward()
            # self.vae_optimizer.step()

            del episode_sample
            t += 1

        all_z = th.cat(all_z, dim=0)
        all_states = th.cat(all_states, dim=0)

        return loss, recon_obs_loss, kl_loss, all_z, all_states

    def train_rnn(self, s, z):

        h=None
        loss= 0
        self.rnn_optimizer.zero_grad()
        for t in range(z.shape[1]):
            z_ = z[:, t]
            if h != None:
                h = h.detach()

            s_pred, h = self.mac.rnn(z_.detach(), h)

            s_target = s[:, t].detach()
            loss += F.mse_loss(s_target, s_pred)
        loss.backward()
        self.rnn_optimizer.step()

        return loss/z.shape[1]


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, buffer=None):

        # Train vae

        for i in range(10):

            self.mac.vae.train()
            self.mac.rnn.train()

            vae_loss, recon_obs_loss, kl_loss, all_z, all_states = self.train_vae(batch, buffer)

            # Train RNN and ensure backpropagation through the encoder
            h = None
            self.mac.init_hidden(batch.batch_size)
            rnn_loss = 0


            self.rnn_optimizer.zero_grad()

            for t in range(all_z.shape[1]):
                z_ = all_z[:, t]  # Keep gradient flow intact (do not detach)

                if h is not None:
                    h1, c = h
                    h = (h1.detach(), c.detach())  # Detach hidden state if needed

                # print("z ", z_.shape)
                s_pred, h = self.mac.rnn(z_, h)
                s_target = all_states[:, t]
                rnn_loss += F.mse_loss(s_target, s_pred)

            # Combine losses
            total_loss = 0.5 * vae_loss + (rnn_loss / all_z.shape[1])

            # Backpropagate through both VAE and RNN
            total_loss.backward()

            # Optimize both VAE and RNN
            self.rnn_optimizer.step()
            self.vae_optimizer.step()
            # rnn_loss = self.train_rnn(all_states, all_z)

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        self.mac.agent.train()
        self.mac.init_hidden(batch.batch_size)
        mac_out = []
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_ = chosen_action_qvals

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
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
            self.logger.log_stat("vae_loss", vae_loss, t_env)
            self.logger.log_stat("obs_rec_loss", recon_obs_loss, t_env)
            self.logger.log_stat("kl_loss", kl_loss, t_env)
            # self.logger.log_stat("rnn_loss", rnn_loss, t_env)

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
