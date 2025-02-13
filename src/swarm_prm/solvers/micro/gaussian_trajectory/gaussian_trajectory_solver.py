"""
    Finding single agent trajectory by sampling points in Gaussian
    and finding best matching pairs in each group
"""
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from swarm_prm.solvers.utils.gaussian_prm import GaussianPRM

def is_within_ci(point, g_node, chi2_thresh):
    delta = point - g_node.get_mean()
    mahalanobis_distance_square = delta.T @ np.linalg.inv(g_node.covariance) @ delta
    return mahalanobis_distance_square <= chi2_thresh

def sample_gaussian(g_node, num_points, ci, min_spacing, candidates=None):

    """
        Return Gaussian Samples within the confidence interval
    """
    points = []
    tree = KDTree([g_node.get_mean()])
    chi2_thresh = chi2.ppf(ci, 2)
    if candidates is None:
        num_candidates = 1000
        mean, cov = g_node.get_gaussian()
        candidates = np.random.multivariate_normal(mean, cov, num_candidates)
    for candidate in candidates:
        if len(points) == num_points:
            break
        if is_within_ci(candidate, g_node, chi2_thresh) \
            and tree.query(candidate)[0] > min_spacing :
            points.append(candidate)
            tree = KDTree(points)
    return np.array(points)

class GaussianTrajectorySolver:

    def __init__(self, gaussian_prm:GaussianPRM, macro_solution, timestep,
                    num_agent, safety_gap = 0.2, ci=0.8,
                 ):
        self.gaussian_prm = gaussian_prm 
        self.macro_solution = macro_solution
        self.timestep = timestep
        self.num_agent = num_agent

        # Gaussian Sampling Parameters
        self.safety_gap = safety_gap
        self.ci = ci # Confidence Interval

        # precompute Gaussian Candidates
        self.gaussian_candidates = []
        for g_node in self.gaussian_prm.gaussian_nodes:
            mean, cov = g_node.get_gaussian()
            self.gaussian_candidates.append(np.random.multivariate_normal(mean, cov, 1000))

        self.agent_locations = {} # Track current agent locations
        self.node_agents = {} # Track agents assigned to each node
        self.agent_assigned = [False for _ in range(self.num_agent)]

    def choose_unassigned_agents(self, node_idx, next_node_idx, num_agents):
        """
            Choose agents in node that is not assigned and closest to the next node
        """
        curr_agents = [agent for agent in self.node_agents[node_idx] if not self.agent_assigned[agent]]
        curr_positions = [self.agent_locations[agent] for agent in curr_agents]

        # Find agents that are closest to the next node
        g_node = self.gaussian_prm.gaussian_nodes[next_node_idx]
        mean = g_node.get_mean()
        agents = np.argsort(np.linalg.norm(curr_positions - mean))[:num_agents] # Take num_agents closest agents
        agents_idx = [curr_agents[agent] for agent in agents]
        return agents_idx, [curr_positions[agent] for agent in agents_idx]
    

    def get_node_samples(self, node_idx, node_agent_count):
        """
            Sample points in the Gaussian Node according to the # of agents
            assigned to the node based on the macro solution.
            
            Input: 
                node_idx: Index of the Gaussian Node
                node_agent_count: dict of # of agents assigned to the node
            Output:
                samples: List of samples in the Gaussian Node

        """
        g_node = self.gaussian_prm.gaussian_nodes[node_idx]
        num_agents = node_agent_count[node_idx]
        return sample_gaussian(g_node, num_agents, self.ci, self.safety_gap, self.gaussian_candidates[node_idx])


    def update_agent_locations(self, agents_idx, agents_positions, gaussian_node_idx, gaussains_samples):
        """
            Match agents to the node based on the flow graph.
            For each goal locations, we get the # of agents assigned to the node,
            and sample points in the Gaussian Node. All the agents travelling to the
            same node will be matched to the samples based on the min sum of distances.
        """

        # match agents to the samples based on distances
        distance_matrix = cdist(agents_positions, gaussains_samples)
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        matchings = np.stack((row_ind, col_ind), axis=-1)

        # update agent-wise trajectory
        trajectory = {}
        for m in matchings:
            trajectory[agents_idx[m[0]]] = gaussains_samples[m[1]]
        return trajectory

    def solve(self):
        """
            Find trajectories per timestep. We find start-goal pairs based on minimizing
            sum of distances between start and goal points.
        """

        trajectories = [[] for _ in range(self.num_agent)] # Store agent-wise trajectories
        g_node_agent_count = defaultdict(lambda: 0)
        curr_agent_idx = 0

        # Initialize agent locations
        for i, start_idx in enumerate(self.gaussian_prm.starts_idx):
            g_node_agent_count[start_idx] += int(self.gaussian_prm.starts_weight[i]*self.num_agent)
            samples = self.get_node_samples(start_idx, g_node_agent_count)

            for sample in samples:
                self.agent_locations[curr_agent_idx] = sample
                self.node_agents[start_idx].append(curr_agent_idx)
                curr_agent_idx += 1
        
        for t in range(1, self.timestep+1):
            self.agent_assigned = [False for _ in range(self.num_agent)]
            # compute all Gaussian node # of agents
            incoming_flow = defaultdict(list)
            for u in self.macro_solution[t]:
                for flow in self.macro_solution[t][u]:
                    g_node_agent_count[flow[0]] += flow[1]
                    g_node_agent_count[u] -= flow[1]
                    incoming_flow[flow[0]].append(flow)

            # sample new locations in goal nodes
            for g_node_idx in incoming_flow.keys():
                samples = self.get_node_samples(g_node_idx, g_node_agent_count)
                
                incoming_agents = []
                incoming_agents_locations = []
                for flow in incoming_flow[g_node_idx]:
                    agents_idx, agents_positions = self.choose_unassigned_agents(flow[0], g_node_idx, flow[1]) # decide agents in the node
                    incoming_agents += agents_idx
                    incoming_agents_locations += agents_positions
                    for agent in agents_idx:
                        self.agent_assigned[agent] = True

                # update agents inside the node
                self.node_agents[g_node_idx] = incoming_agents

                # update agent locations
                trajectory = self.update_agent_locations(incoming_agents, incoming_flow, g_node_idx, samples)
                for agent_idx, goal in trajectory.items():
                    self.agent_locations[agent_idx] = goal
                    trajectories[agent_idx].append(goal)
        return trajectories
    