"""
    Evaluation solver.
    This solver takes a list of Gaussian nodes and a macro solution based on
    macro solution indicies.
"""
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

def sample_gaussian(g_node, num_points, confidence_interval, min_spacing, candidates=None):

    """
        Return Gaussian Samples within the confidence interval
    """
    chi2_thresh = chi2.ppf(confidence_interval, 2)
    mean, cov = g_node.get_gaussian()

    if candidates is None:
        num_candidates = 10000
        candidates = np.random.multivariate_normal(mean, cov, num_candidates)

    delta = candidates - mean
    inv_cov = np.linalg.inv(cov)
    dists = np.einsum("ni, ij, nj ->n", delta, inv_cov, delta)  
    within_ci_mask = dists <= chi2_thresh
    filtered = candidates[within_ci_mask]

    if len(filtered) == 0:
        return np.zeros((0, mean.shape[0]))

    selected = [filtered[0]]
    for point in filtered[1:]:
        if len(selected) >= num_points:
            break
        distances = cdist([point], selected)[0]
        if np.all(distances > min_spacing):
            selected.append(point)
    return np.array(selected)

class EvaluationSolver:

    def __init__(self, gaussian_nodes, macro_solution, agent_radius, timestep,
                     num_agents, starts_idx, goals_idx, 
                     starts_agent_count, goals_agent_count,
                     confidence_interval=0.95,
                    interpolation_count=10
                 ):
        self.gaussian_nodes = gaussian_nodes 
        self.macro_solution = macro_solution
        self.timestep = timestep
        self.num_agents = num_agents
        self.starts_idx = starts_idx
        self.goals_idx = goals_idx
        self.starts_agent_count = starts_agent_count
        self.goals_agent_count = goals_agent_count
        self.interpolation_count = interpolation_count

        # Gaussian Sampling Parameters
        # self.safety_gap = 0 
        self.safety_gap = 2*agent_radius
        self.confidence_interval = confidence_interval # Confidence Interval

        # precompute Gaussian Candidates
        self.gaussian_candidates = []
        for g_node in self.gaussian_nodes:
            mean, cov = g_node.get_gaussian()
            self.gaussian_candidates.append(np.random.multivariate_normal(mean, cov, 1000))

        self.agent_locations = {} # Track current agent locations
        self.node_agents = defaultdict(list) # Track agents assigned to each node
        self.agent_assigned = [False] * self.num_agents

    def choose_unassigned_agents(self, prev_node, node, num_agents):
        """
            Choose agents in node that is not assigned and closest to the next node
        """

        curr_agents = [agent for agent in self.node_agents[prev_node] if not self.agent_assigned[agent]]
        curr_positions = np.array([self.agent_locations[agent] for agent in curr_agents])

        # Find agents that are closest to the next node
        g_node = self.gaussian_nodes[node]
        mean = g_node.get_mean()
        sorted_agents = np.argsort(np.linalg.norm(curr_positions - mean, axis=1)) # Take num_agents closest agents
        agents = sorted_agents[:num_agents]
        agents_idx = [curr_agents[agent] for agent in agents]
        return agents_idx
    
    def get_node_samples(self, node_idx, num_samples):
        """
            Sample points in the Gaussian Node according to the # of agents
            assigned to the node based on the macro solution.
            
            Input: 
                node_idx: Index of the Gaussian Node
                num_samples: # of sample to get from the node 
            Output:
                samples: List of samples in the Gaussian Node

        """
        g_node = self.gaussian_nodes[node_idx]
        return sample_gaussian(g_node, num_samples, self.confidence_interval, self.safety_gap, self.gaussian_candidates[node_idx])

    def update_agent_locations(self, agents_idx, gaussains_samples):
        """
            Match agents to the node based on the flow graph.
            For each goal locations, we get the # of agents assigned to the node,
            and sample points in the Gaussian Node. All the agents travelling to the
            same node will be matched to the samples based on the min sum of distances.
        """

        # match agents to the samples based on distances
        agents_positions = [self.agent_locations[idx] for idx in agents_idx]
        distance_matrix = cdist(agents_positions, gaussains_samples)
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        matchings = np.stack((row_ind, col_ind), axis=-1)

        # update agent-wise trajectory
        trajectory = {}
        for m in matchings:
            trajectory[agents_idx[m[0]]] = gaussains_samples[m[1]]
        return trajectory

    def solve(self):
        solution = [[] for _ in range(self.num_agents)]
        curr_agent_idx = 0

        # Step 1: assign agents to start samples
        for i, start_idx in enumerate(self.starts_idx):
            if self.starts_agent_count[i] == 0:
                continue
            samples = self.get_node_samples(start_idx, self.starts_agent_count[i])
            for s in samples:
                self.agent_locations[curr_agent_idx] = s
                self.node_agents[start_idx].append(curr_agent_idx)
                solution[curr_agent_idx].append(s)
                curr_agent_idx += 1

        # Step 2: step through macro flow
        for t in range(self.timestep + 1):
            new_node_agents = defaultdict(list)

            for prev_node in self.macro_solution[t]:
                agents = self.node_agents[prev_node]
                start = 0

                for next_node, flow in self.macro_solution[t][prev_node]:
                    if flow == 0:
                        continue

                    subset = agents[start:start + flow]
                    if len(subset) < flow:
                        break
                    start += flow

                    # Sample targets and assign
                    samples = self.get_node_samples(next_node, flow)
                    if len(samples) < flow:
                        continue

                    prev_pos = np.array([self.agent_locations[a] for a in subset])
                    cost_matrix = cdist(prev_pos, samples)
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    for r, c in zip(row_ind, col_ind):
                        agent = subset[r]
                        pos = samples[c]
                        self.agent_locations[agent] = pos
                        new_node_agents[next_node].append(agent)
                        solution[agent].append(pos)

            self.node_agents = new_node_agents

            # Pad idle agents who weren't assigned
            for i in range(self.num_agents):
                if len(solution[i]) < t + 2:
                    solution[i].append(solution[i][-1])

        # Convert and compute cost
        solution = np.array(solution)
        path_costs = [
            [np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])]
            for path in solution
        ]
        cost = np.sum(np.max(path_costs, axis=0))
        return solution, cost
