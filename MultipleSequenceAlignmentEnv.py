import gymnasium as gym
import numpy as np
from pymsa import MSA, SumOfPairs
from pymsa import Blosum62
from Encode import encoder


class MultipleSequenceAlignmentEnv(gym.Env):
    alphabet = 'ARNDCQEGHILKMFPSTWYV'
    def __init__(self, sequences):
        super(MultipleSequenceAlignmentEnv, self).__init__()
        self.state = None
        self.gap_counts = None
        self.encoded_sequences = None
        self.sequences = sequences
        self.n_sequences = len(sequences)
        self.n_characters = len(self.alphabet) 
        self.max_length = max(len(sequence) for sequence in sequences)
        self.action_space = gym.spaces.MultiDiscrete([self.n_sequences, self.max_length])
        self.observation_space = gym.spaces.Box(low=0,
                                                high=1,
                                                shape=(self.n_sequences, self.max_length, self.n_characters + 1),
                                                dtype=np.float32)

        
        

    def reset(self,seed=None):
        
        encoded_shape = (self.n_sequences, self.max_length, self.n_characters)
        self.encoded_sequences = np.zeros(encoded_shape, dtype=np.float32)          
        self.state = np.zeros([self.n_sequences, self.max_length], dtype=np.int_)
        self.gap_counts = np.zeros([self.n_sequences, self.max_length], dtype=np.int_)
        
        for i, sequence in enumerate(self.sequences):
            self.state[i, :len(sequence)] = np.arange(1, len(sequence)+1)
            self.encoded_sequences[i, :len(sequence), :] = encoder(sequence,encoding='OneHot_20D')
        return self._get_observation(),{}
           
    def step(self, action):
        seq_idx, pos = action
        self._insert_gap(seq_idx, pos)
        reward = self._calculate_reward()
        done = np.sum(self.gap_counts) == self.n_sequences # temporary
        info = {}
        #observation, reward, terminated, False, info
        return self._get_observation(), reward, done,False, info

    def _get_observation(self):
        normalized_gap_counts = self.gap_counts / self.max_length
        return np.concatenate([self.encoded_sequences, normalized_gap_counts[:, :, np.newaxis]], axis=2)


    def _insert_gap(self, seq_idx, pos):
        # inserting a gap in the sequence seq_idx at position pos
        self.state[seq_idx, pos:len(self.sequences[seq_idx])] += 1
        self.gap_counts[seq_idx, pos] += 1

    def _calculate_reward(self):
        current_alignment = self.mat_string_alignment()
        msa_obj = MSA(current_alignment, range(self.n_sequences))
        return SumOfPairs(msa_obj, Blosum62()).compute()
 
    def mat_string_alignment(self):
        # length of the longest aligned sequence
        max_len = np.max(self.state)
        # make a matrix full of gaps
        alignment = np.full([self.n_sequences, max_len], '-', dtype=str)
        # fill the matrix with the aligned bases in the correct positions
        alignments_string = []
        for i,seq in enumerate(self.sequences):
            for j,base in enumerate(seq):
                alignment[i, self.state[i,j]-1] = base
            alignments_string.append(''.join(alignment[i]))
        
        return alignments_string
    
    def print_mat_string_alignment(self):
        aligns = self.mat_string_alignment()
        for align in aligns:
            print(align)
    
    def render(self, mode='human'):
        self.print_mat_string_alignment()
        
if __name__ == "__main__":

    env = MultipleSequenceAlignmentEnv(['MCRIAGGRGTLLPLLAALLQA',
                                        'MSFPCKFVASFLLIFNVSSKGA',
                                        'MPGKMVVILGASNILWIMF'])
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(action, reward)
    env.print_mat_string_alignment()