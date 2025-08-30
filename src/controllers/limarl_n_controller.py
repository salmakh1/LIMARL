from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
from modules.vae.vae_agent import VAE

from modules.vae.state_rvae import Decoder

from modules.vae.state_rvae import RE

from modules.vae.state_rvae import StateEncoder
import torch.nn.functional as F
import copy


class SL_SRVAE_NMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(SL_SRVAE_NMAC, self).__init__(scheme, groups, args)

        # construct VAE instance
        state_shape = scheme["state"]["vshape"]
        self.latent_dim_vae = args.latent_dim_vae
        self.hidden_encoder = args.hidden_dim_vae

        self.state_hidden_encoder = args.state_hidden_encoder
        input_shape_rvae = scheme["obs"]["vshape"]

        if self.args.obs_last_action:
            input_shape_rvae += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape_rvae += self.n_agents

        self.re = RE(input_shape_rvae, self.hidden_encoder, self.latent_dim_vae).to("cuda:0")
        self.decoder = Decoder(self.latent_dim_vae, self.state_hidden_encoder, state_shape).to("cuda:0")
        self.state_pred = StateEncoder(state_shape, self.state_hidden_encoder, self.latent_dim_vae).to("cuda:0")

        self.r_vae_hidden_state = None
        # agent input
        input_shape = scheme["obs"]["vshape"]
        print("Observation shape is ", scheme["obs"]["vshape"])
        print("State shape is ", scheme["state"]["vshape"])
        # input_shape += self.latent_dim_vae
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        print("input_shape ", input_shape)

        self._build_agents(input_shape, self.latent_dim_vae)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode, t_env = t_env)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, batch, t, test_mode=False, t_env = 0):

        if test_mode:
            self.agent.eval()
            self.re.eval()


        re_inputs = self.get_vae_inputs(batch, t)
        re_inputs = re_inputs.reshape(re_inputs.shape[0] * re_inputs.shape[1], -1).unsqueeze(1)
        # print("SHAPE ",re_inputs.shape )
        with th.no_grad():

            mu_re, logvar, self.r_vae_hidden_state = self.re(re_inputs, self.r_vae_hidden_state)
            z = self.re.reparameterize(mu_re, logvar)
            mu = mu_re

            # if not test_mode:
            #     agent_input = self.get_inputs(batch, t, z)
            # else:
            agent_input = self.get_inputs(batch, t, mu)

            # if not test_mode:
            #     p = 0.96**(t_env**(1/4))
            #     # print("p ", p)
            #     state = batch["state"][:, t]
            #     state = state.unsqueeze(1).expand(batch.batch_size, self.args.n_agents, -1)
            #     mu_state, _ = self.state_pred(state.reshape(state.shape[0]* state.shape[1], -1))
            #     mu = p * mu_state + (1 - p) * mu_re
            # #
            # else:
            #     mu = mu_re

            # if self.args.oracle:
            #     state = batch["state"][:, t]
            #     state = state.unsqueeze(1).expand(batch.batch_size, self.args.n_agents, -1)
            #     mu_state, _ = self.state_pred(state.reshape(state.shape[0] * state.shape[1], -1))
            #     mu = mu_state
        # if not test_mode:
        #     agent_outs, self.hidden_states = self.agent(agent_input.detach(), z.detach(), self.hidden_states)
        # else:
        agent_outs, self.hidden_states = self.agent(agent_input.detach(), mu.detach(), self.hidden_states)


        return agent_outs.view(batch.batch_size, self.n_agents, -1)

    # recons_state = self.decoder.decode(z)
    # forward_loss = F.mse_loss(recons_state.reshape(batch.batch_size, self.args.n_agents, -1), state, reduction='mean')
    # print("forward_loss ", forward_loss)
    # mu = mu.reshape(batch.batch_size, self.args.n_agents, -1)
    # recon_state = self.decoder(mu)

    # origin_state = batch["state"][:, t]
    # origin_state = origin_state.unsqueeze(1).expand(batch.batch_size, self.args.n_agents, -1)

    def get_inputs(self, batch, t, z=None):
        bs = batch.batch_size
        inputs = []
        # print("shape ", batch["obs"].shape)
        obs_part = batch["obs"][:, t]  # Observation part
        inputs.append(obs_part)  # Add observation to inputs
        # if z is not None:
        #     inputs.append(z)  # Add observation to inputs
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def get_vae_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        # print("shape ", batch["obs"].shape)
        obs_part = batch["obs"][:, t]  # Observation part
        inputs.append(obs_part)  # Add observation to inputs

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def load_reccurent_state(self, other_mac):
        self.re.load_state_dict(other_mac.re.state_dict())
        self.state_pred.load_state_dict(other_mac.state_pred.state_dict())
        self.decoder.load_state_dict(other_mac.decoder.state_dict())

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

        self.r_vae_hidden_state = None


    def save_session_best_model(self):
        session_best_model = {
            'mac': copy.deepcopy(self.agent.state_dict()),
            're': copy.deepcopy(self.re.state_dict()),
            'decoder': copy.deepcopy(self.decoder.state_dict()),
            'state_pred': copy.deepcopy(self.state_pred.state_dict()),
        }
        return session_best_model
