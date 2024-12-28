import torch
from agent import Agent
from environment import QEDRewardMolecule
import math
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np

def get_fingerprint(smiles, fingerprint_length, fingerprint_radius):
    """Get Morgan Fingerprint of a specific SMILES string.

  Args:
    smiles: String. The SMILES string of the molecule.
    hparams: tf.contrib.training.HParams. Hyper parameters.

  Returns:
    np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
  """
    if smiles is None:
        return np.zeros((fingerprint_length,))
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.zeros((fingerprint_length,))
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, fingerprint_radius, fingerprint_length
    )
    arr = np.zeros((1,))
    # ConvertToNumpyArray takes ~ 0.19 ms, while
    # np.asarray takes ~ 4.69 ms
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr




TENSORBOARD_LOG = True
TB_LOG_PATH = "./runs/dqn/run2"
episodes = 0
iterations = 200000
update_interval = 20
batch_size = 128
num_updates_per_it = 2
start_molecule = None
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 2000
optimizer = "Adam"
polyak = 0.995
atom_types = ["C", "O", "N"]
max_steps_per_episode = 40
allow_removal = True
allow_no_modification = True
allow_bonds_between_rings = False
allowed_ring_sizes = [3, 4, 5, 6]
replay_buffer_size = 1000000
learning_rate = 1e-4
gamma = 0.95
fingerprint_radius = 3
fingerprint_length = 2048
discount_factor = 0.9



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

environment = QEDRewardMolecule(
    discount_factor=discount_factor,
    atom_types=set(atom_types),
    init_mol=start_molecule,
    allow_removal=allow_removal,
    allow_no_modification=allow_no_modification,
    allow_bonds_between_rings=allow_bonds_between_rings,
    allowed_ring_sizes=set(allowed_ring_sizes),
    max_steps=max_steps_per_episode,
)

# DQN Inputs and Outputs:
# input: appended action (fingerprint_length + 1) .
# Output size is (1).

agent = Agent(fingerprint_length + 1, 1, device, replay_buffer_size, optimizer, learning_rate, fingerprint_length)


environment.initialize()

eps_threshold = 1.0
batch_losses = []

for it in range(iterations):

    steps_left = max_steps_per_episode - environment.num_steps_taken

    # Compute a list of all possible valid actions. (Here valid_actions stores the states after taking the possible actions)
    valid_actions = list(environment.get_valid_actions())

    # Append each valid action to steps_left and store in observations.
    observations = np.vstack(
        [
            np.append(
                get_fingerprint(
                    act, fingerprint_length, fingerprint_radius
                ),
                steps_left,
            )
            for act in valid_actions
        ]
    )  # (num_actions, fingerprint_length)

    observations_tensor = torch.Tensor(observations)
    # Get action through epsilon-greedy policy with the following scheduler.
    # eps_threshold = hyp.epsilon_end + (hyp.epsilon_start - hyp.epsilon_end) * \
    #     math.exp(-1. * it / hyp.epsilon_decay)

    a = agent.get_action(observations_tensor, eps_threshold)

    # Find out the new state (we store the new state in "action" here. Bit confusing but taken from original implementation)
    action = valid_actions[a]
    # Take a step based on the action
    result = environment.step(action)

    action_fingerprint = np.append(
        get_fingerprint(action, fingerprint_length,fingerprint_radius),
        steps_left,
    )

    next_state, reward, done = result

    # Compute number of steps left
    steps_left = max_steps_per_episode - environment.num_steps_taken

    # Append steps_left to the new state and store in next_state
    next_state = get_fingerprint(
        next_state, fingerprint_length, fingerprint_radius
    )  # (fingerprint_length)

    action_fingerprints = np.vstack(
        [
            np.append(
                get_fingerprint(
                    act, fingerprint_length, fingerprint_radius
                ),
                steps_left,
            )
            for act in environment.get_valid_actions()
        ]
    )  # (num_actions, fingerprint_length + 1)

    # Update replay buffer (state: (fingerprint_length + 1), action: _, reward: (), next_state: (num_actions, fingerprint_length + 1),
    # done: ()

    agent.replay_buffer.add(
        obs_t=action_fingerprint,  # (fingerprint_length + 1)
        action=0,  # No use
        reward=reward,
        obs_tp1=action_fingerprints,  # (num_actions, fingerprint_length + 1)
        done=float(result.terminated),
    )

    if done:
        final_reward = reward
        if episodes != 0 and episodes % 2 == 0 and len(batch_losses) != 0:
            print(
                "reward of final molecule at episode {} is {}".format(
                    episodes, final_reward
                )
            )
            print(
                "mean loss in episode {} is {}".format(
                    episodes, np.array(batch_losses).mean()
                )
            )
        episodes += 1
        eps_threshold *= 0.99907
        batch_losses = []
        environment.initialize()

    if it % update_interval == 0 and agent.replay_buffer.__len__() >= batch_size:
        for update in range(num_updates_per_it):
            loss = agent.update_params(batch_size, gamma,polyak)
            loss = loss.item()
            batch_losses.append(loss)
