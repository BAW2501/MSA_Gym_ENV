import gymnasium as gym
import numpy as np
from pymsa import MSA, SumOfPairs,Blosum62
from Encode import encoder

training_data = [
    ['SGVPDR', 'GVPDR', 'VPDR', 'SGVPD'],
    ['TFGGGT', 'FGGGT', 'GGGT', 'TFGGG'],
    ['DSAVYY', 'SAVYF', 'AVYF', 'VYY'],
    ['VPDRFS', 'PDRFS', 'DRFS', 'RFS'],
    ['DQASIS', 'QASIS', 'ASIS', 'SIS'],
    ['LYTLSS', 'YTLSS', 'TLSS', 'LSS'],
    ['KLEIKR', 'LEIKR', 'EIKR', 'IKR'],
    ['SLPVSL', 'LPVSL', 'PVSL', 'VSL'],
    ['LGCLVK', 'GCLVK', 'CLVK', 'LVK'],
    ['QFGRCS', 'FGRCS', 'GRCS', 'RCS'],
]


def get_random_shuffled_sequences(data, seed=0):
    # np.random.seed(seed)
    number_of_sequences = len(data)
    index = np.random.choice(range(number_of_sequences))
    sequences = data[index]
    np.random.shuffle(sequences)
    return sequences


class MultipleSequenceAlignmentEnv(gym.Env):
    alphabet = 'ARNDCQEGHILKMFPSTWYV'

    def __init__(self, sequences=get_random_shuffled_sequences(training_data)):
        super(MultipleSequenceAlignmentEnv, self).__init__()
        self.score = 0
        self.state = None
        self.gap_counts = None
        self.encoded_sequences = None
        self.sequences = sequences
        self.n_sequences = len(sequences)
        self.ids = list(range(self.n_sequences))
        self.n_characters = len(self.alphabet)
        self.max_length = max(len(sequence) for sequence in sequences)
        self.action_space = gym.spaces.MultiDiscrete([self.n_sequences, self.max_length])
        self.observation_space = gym.spaces.Box(low=0,
                                                high=1,
                                                shape=(self.n_sequences, self.max_length, self.n_characters + 1),
                                                dtype=np.float32)

    def reset(self, seed=0, options=None):
        self.sequences = get_random_shuffled_sequences(training_data, seed)
        self.n_sequences = len(self.sequences)
        self.n_characters = len(self.alphabet)
        self.max_length = max(len(sequence) for sequence in self.sequences)
        self.score = 0
        encoded_shape = (self.n_sequences, self.max_length, self.n_characters)
        self.encoded_sequences = np.zeros(encoded_shape, dtype=np.float32)
        self.state = np.zeros([self.n_sequences, self.max_length], dtype=np.int_)
        self.gap_counts = np.zeros([self.n_sequences, self.max_length], dtype=np.int_)

        for i, sequence in enumerate(self.sequences):
            self.state[i, :len(sequence)] = np.arange(1, len(sequence) + 1)
            self.encoded_sequences[i, :len(sequence), :] = encoder(sequence, encoding='OneHot_20D')
        self.score = self.calculate_reward()
        return self._get_observation(), {}

    def step(self, _action):
        seq_idx, pos = _action
        self._insert_gap(seq_idx, pos)
        _reward = self.calculate_reward()
        diff = _reward - self.score
        self.score = _reward
        _reward = diff

        _done = np.sum(self.gap_counts) == self.n_sequences  # temporary
        _info = {}
        # observation, reward, terminated, False, info
        return self._get_observation(), _reward, _done, False, _info

    def _get_observation(self):
        normalized_gap_counts = self.gap_counts / self.max_length
        return np.concatenate([self.encoded_sequences, normalized_gap_counts[:, :, np.newaxis]], axis=2,dtype=np.float32)

    def _insert_gap(self, seq_idx, pos):
        # inserting a gap in the sequence seq_idx at position pos
        self.state[seq_idx, pos:len(self.sequences[seq_idx])] += 1
        self.gap_counts[seq_idx, pos] += 1

    def calculate_reward(self):
        current_alignment = self.mat_string_alignment()
        msa_obj = MSA(current_alignment, self.ids)
        return SumOfPairs(msa_obj, Blosum62()).compute()

    def mat_string_alignment(self):
        # length of the longest aligned sequence
        max_len = np.max(self.state)
        # make a matrix full of gaps
        alignment = np.full([self.n_sequences, max_len], '-', dtype=str)
        # fill the matrix with the aligned bases in the correct positions
        alignments_string = []
        for i, seq in enumerate(self.sequences):
            for j, base in enumerate(seq):
                alignment[i, self.state[i, j] - 1] = base
            #alignments_string.append(''.join(alignment[i]))

        return alignment #alignments_string

    def print_mat_string_alignment(self):
        aligns = self.mat_string_alignment()
        for align in aligns:
            print(align)

    def render(self, mode='human'):
        self.print_mat_string_alignment()


if __name__ == "__main__":

    env = MultipleSequenceAlignmentEnv()
    obs = env.reset()
    print(env.calculate_reward())
    actions = [(1, 0), (2, 0), (2, 0), ]
    for action in actions:
        _, reward, done, _, info = env.step(action)
        print(action, reward)
        env.print_mat_string_alignment()
        print()
