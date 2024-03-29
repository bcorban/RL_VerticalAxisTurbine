# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import warnings

warnings.filterwarnings("ignore")
import argparse
import os
import random
import time
from distutils.util import strtobool
import gclib
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from load_mat import load_mat
import RL_
from scipy.io import savemat
from config_ENV import CONFIG_ENV

savemat("CONFIG_ENV.mat", CONFIG_ENV)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="SAC_VAWT",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="RL_VAWT",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="RL_/CustomEnv-v0-phase",
    # parser.add_argument("--env-id", type=str, default="RL_/CustomEnv-v0",
    # parser.add_argument("--env-id", type=str, default="Pendulum-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100000,  #TOTAL TIMESTEPS
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(150000),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.97, #GAMMA PARAMETER FOR THE Q VALUES
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=512,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5000, #HAS TO BE HIGHER THAN THE BATCH SIZE
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=5e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=1, #was 2
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5
CHECKPOINT_FREQUENCY = 50


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod(), 256
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = (
            normal.rsample()
        )  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    # -----------------Connect to galil and set parameters ----------------------
    UTD = 1  # Update to data ratio (higher=less steps per second)

    import getpass

    user = getpass.getuser()
    g = gclib.py()

    g.GOpen("192.168.255.200 --direct -s ALL")

    args = parse_args()
    bc = CONFIG_ENV["bc"]
    date = CONFIG_ENV["date"]
    ms = CONFIG_ENV["ms"]
    run_name = f"bc{bc} date{date} ms{ms}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    if args.track:
        writer = SummaryWriter(f"{wandb.run.dir}")
    else:
        writer = SummaryWriter(f"runs/{run_name}")

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    # device = torch.device("cpu")

    print(f"running on {device}")
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    # actor.load_state_dict(torch.load('Actor_open_loop.pt'))
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    RESUME_EXPERIMENT = False
    path = "./wandb/run-20230628_135335-lki4uz1i/files"
    if RESUME_EXPERIMENT:
        actor.load_state_dict(torch.load(f"{path}/actor_final.pt"))
        qf1.load_state_dict(torch.load(f"{path}/qf1_final.pt"))
        qf2.load_state_dict(torch.load(f"{path}/qf2_final.pt"))
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        if args.autotune:
            log_alpha = torch.load(f"{path}/log_alpha_final.pt")
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)

    envs.single_observation_space.dtype = np.float32

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    if CONFIG_ENV["pre-fill-RB"]:
        transition = load_mat("./RB.mat")["transition"]
        for transi in range(len(transition["r"])):
            rb.add(
                [transition["s"][transi]],
                [transition["s_next"][transi]],
                [transition["a"][transi]],
                [transition["r"][transi]],
                [False],
                [{}],
            )

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()

    start_time = time.time()
    mean_r = 0
    # mean_cp=0
    update = 0

    SPS_time = time.time()
    time_total = []
    time_1 = []
    time_2 = []
    SPS_list = []

    for global_step in range(args.total_timesteps):
        try:
            # ALGO LOGIC: put action logic here

            obs = np.array(
                [envs.envs[0].state]
            )  # Read state before deciding action

            if global_step < args.learning_starts:
                # if global_step<200:
                #     actions=np.array([0])
                # else:
                actions = np.array(
                    [
                        envs.single_action_space.sample()
                        for _ in range(envs.num_envs)
                    ]
                )
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.

            _, _, dones, infos = envs.step([[actions[0],obs[0]]])

            # ALGO LOGIC: training loop begin--------------------------------------------------
            if global_step > args.learning_starts:
                for _ in range(UTD):
                    data = rb.sample(args.batch_size)

                    with torch.no_grad():
                        (
                            next_state_actions,
                            next_state_log_pi,
                            _,
                        ) = actor.get_action(data.next_observations)
                        qf1_next_target = qf1_target(
                            data.next_observations, next_state_actions
                        )
                        qf2_next_target = qf2_target(
                            data.next_observations, next_state_actions
                        )
                        min_qf_next_target = (
                            torch.min(qf1_next_target, qf2_next_target)
                            - alpha * next_state_log_pi
                        )
                        next_q_value = data.rewards.flatten() + (
                            1 - data.dones.flatten()
                        ) * args.gamma * (min_qf_next_target).view(-1)

                    qf1_a_values = qf1(data.observations, data.actions).view(-1)
                    qf2_a_values = qf2(data.observations, data.actions).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()

                if (
                    global_step % args.policy_frequency == 0
                ):  # TD 3 Delayed update support
                    for _ in range(
                        args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = actor.get_action(data.observations)
                        qf1_pi = qf1(data.observations, pi)
                        qf2_pi = qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                        # actor_loss = ((alpha * log_pi) - min_qf_pi).mean() #original SACv2 implementation
                        actor_loss = (
                            (max(alpha, 0.05) * log_pi) - min_qf_pi
                        ).mean()  # Quick fix of the SAC v2 version to force exploration
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(
                                    data.observations
                                )
                            alpha_loss = (
                                -log_alpha * (log_pi + target_entropy)
                            ).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(
                        qf1.parameters(), qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data
                            + (1 - args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        qf2.parameters(), qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data
                            + (1 - args.tau) * target_param.data
                        )

                if global_step % 100 == 0 and global_step > 0:
                    writer.add_scalar(
                        "losses/qf1_values",
                        qf1_a_values.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "losses/qf2_values",
                        qf2_a_values.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "losses/qf1_loss", qf1_loss.item(), global_step
                    )
                    writer.add_scalar(
                        "losses/qf2_loss", qf2_loss.item(), global_step
                    )
                    writer.add_scalar(
                        "losses/qf_loss", qf_loss.item() / 2.0, global_step
                    )
                    writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), global_step
                    )
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    writer.add_scalar(
                        "charts/SPS",
                        int(100 / (time.time() - SPS_time)),
                        global_step,
                    )
                    SPS_time = time.time()
                    if args.autotune:
                        writer.add_scalar(
                            "losses/alpha_loss", alpha_loss.item(), global_step
                        )

            else:
                time.sleep(0.014)

            # -------------------Training loop end------------------------------------

            # -------------read new state from previous action and record transition---------------
            next_obs, rewards = envs.envs[
                0
            ].get_transition()  # gets state and reward with a delay due to RL loop to ensure that the motor has reached the desired position

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            mean_r += rewards[0]

            if global_step % 45 == 0:
                writer.add_scalar(
                    "charts/mean_reward_last_45", mean_r / 45, global_step
                )

                if global_step > args.learning_starts:
                    torch.save(
                        actor.state_dict(),
                        f"{wandb.run.dir}/actor_step_{global_step}.pt",
                    )
                mean_r = 0

            for info in infos:
                if "episode" in info.keys():
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return",
                        info["episode"]["r"],
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_length",
                        info["episode"]["l"],
                        global_step,
                    )
                    break

            # TRY NOT TO MODIFY: save data to replay buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]

            if (
                not dones[0] and not infos[0]["transient"]
            ):  # if episode not terminated and not in transient rotations add to replay buffer
                rb.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

        except KeyboardInterrupt:  # if the training is interrupted
            print("training interrupted, saving last policy and q functions")
            torch.save(actor.state_dict(), f"{wandb.run.dir}/actor_final.pt")
            # wandb.save(f"{wandb.run.dir}/actor_final.pt", policy="now", base_path=f"{wandb.run.dir}")
            torch.save(qf1.state_dict(), f"{wandb.run.dir}/qf1_final.pt")
            # wandb.save(f"{wandb.run.dir}/qf1_final.pt", policy="now", base_path=f"{wandb.run.dir}")
            torch.save(qf2.state_dict(), f"{wandb.run.dir}/qf2_final.pt")
            # wandb.save(f"{wandb.run.dir}/qf2_final.pt", policy="now", base_path=f"{wandb.run.dir}")
            torch.save(log_alpha, f"{wandb.run.dir}/log_alpha_final.pt")
            # wandb.save(f"{wandb.run.dir}/log_alpha_final.pt", base_path=f"{wandb.run.dir}")

    envs.close()
    writer.close()
    torch.save(actor.state_dict(), f"{wandb.run.dir}/actor_final.pt")
    # wandb.save(f"{wandb.run.dir}/actor_final.pt", policy="now", base_path=f"{wandb.run.dir}")
    torch.save(qf1.state_dict(), f"{wandb.run.dir}/qf1_final.pt")
    # wandb.save(f"{wandb.run.dir}/qf1_final.pt", policy="now", base_path=f"{wandb.run.dir}")
    torch.save(qf2.state_dict(), f"{wandb.run.dir}/qf2_final.pt")
    # wandb.save(f"{wandb.run.dir}/qf2_final.pt", policy="now", base_path=f"{wandb.run.dir}")
    torch.save(log_alpha, f"{wandb.run.dir}/log_alpha_final.pt")
    # wandb.save(f"{wandb.run.dir}/log_alpha_final.pt", base_path=f"{wandb.run.dir}")

    g.GClose()
