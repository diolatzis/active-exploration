import math
import random


class MarkovChainSampler:
    def __init__(self, large_step_prob, dimensions, mutation_size_small=1.0 / 25.0, mutation_size_large=1.0 / 20.0):
        self.large_step_prob = large_step_prob
        self.dimensions = dimensions

        self.s1 = mutation_size_small
        self.s2 = mutation_size_large

        self.log_ratio = -math.log(self.s2 / self.s1)

        self.accepted_large_steps = 0
        self.rejected_large_steps = 0

        self.accepted_small_steps = 0
        self.rejected_small_steps = 0

        self.current_sample_id = None
        self.current_state = None
        self.current_state_lifespan = 0

        self.last_step = 'large'

        self.samples = []
        self.proposed_samples = []

        self.sample_index = 0

        for dim in range(self.dimensions):
            self.proposed_samples.append(random.uniform(0, 1))

    def mutate(self):
        if random.uniform(0, 1) < self.large_step_prob:
            self.last_step = 'large'
            self.large_step()
        else:
            self.last_step = 'small'
            self.small_step()

    def large_step(self):
        for dim in range(self.dimensions):
            self.proposed_samples[dim] = random.uniform(0, 1)

    def small_step(self):
        for dim in range(self.dimensions):
            sample = random.uniform(0, 1)

            if sample < 0.5:
                add = True
                sample *= 2.0
            else:
                add = False
                sample = 2.0 * (sample - 0.5)

            dv = self.s2 * math.exp(sample * self.log_ratio)
            if add:
                self.proposed_samples[dim] = self.samples[dim] + dv
                if self.proposed_samples[dim] > 1.0:
                    self.proposed_samples[dim] -= 1.0
            else:
                self.proposed_samples[dim] = self.samples[dim] - dv
                if self.proposed_samples[dim] < 0.0:
                    self.proposed_samples[dim] += 1.0

    def accept(self):
        self.current_state_lifespan = 0
        self.sample_index = 0
        self.samples = self.proposed_samples.copy()
        if self.last_step == 'large':
            self.accepted_large_steps += 1
        else:
            self.accepted_small_steps += 1

    def reject(self):
        self.current_state_lifespan += 1
        self.sample_index = 0
        if self.last_step == 'large':
            self.rejected_large_steps += 1
        else:
            self.rejected_small_steps += 1

    def get_samples(self):
        return self.proposed_samples

    def get_sample(self):
        sample = self.proposed_samples[self.sample_index]
        self.sample_index += 1
        assert self.sample_index <= self.dimensions, 'Samples asked are more than the defined dimensions'
        return sample

    def get_large_acceptance_rate(self):
        if (self.accepted_large_steps + self.rejected_large_steps) == 0:
            return 0
        else:
            return self.accepted_large_steps / (self.accepted_large_steps + self.rejected_large_steps)

    def get_small_acceptance_rate(self):
        if (self.accepted_small_steps + self.rejected_small_steps) == 0:
            return 0
        else:
            return self.accepted_small_steps / (self.accepted_small_steps + self.rejected_small_steps)

    def reset_statistics(self):
        self.accepted_large_steps = 0
        self.rejected_large_steps = 0

        self.accepted_small_steps = 0
        self.rejected_small_steps = 0

