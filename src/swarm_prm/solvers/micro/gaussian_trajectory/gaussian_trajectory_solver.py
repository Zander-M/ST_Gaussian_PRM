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
        self.node_agents = defaultdict(list) # Track agents assigned to each node
        self.agent_assigned = [False] * self.num_agent

    def choose_unassigned_agents(self, prev_node, node, num_agents):
        """
            Choose agents in node that is not assigned and closest to the next node
        """

        curr_agents = [agent for agent in self.node_agents[prev_node] if not self.agent_assigned[agent]]
        curr_positions = np.array([self.agent_locations[agent] for agent in curr_agents])
        if prev_node == 40:
            print(curr_agents)
        # DEBUG

        # Find agents that are closest to the next node
        g_node = self.gaussian_prm.gaussian_nodes[node]
        mean = g_node.get_mean()
        sorted_agents = np.argsort(np.linalg.norm(curr_positions - mean, axis=1)) # Take num_agents closest agents
        agents = sorted_agents[:num_agents]
        agents_idx = [curr_agents[agent] for agent in agents]
        return agents_idx
    

    def get_node_samples(self, node_idx, num_agents):
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
        return sample_gaussian(g_node, num_agents, self.ci, self.safety_gap, self.gaussian_candidates[node_idx])


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
        """
            Find trajectories per timestep. We find start-goal pairs based on minimizing
            sum of distances between start and goal points.
        """

        solution = [[] for _ in range(self.num_agent)]
        curr_agent_idx = 0

        # Initialize agent locations
        for i, start_idx in enumerate(self.gaussian_prm.starts_idx):

            # sample start locations
            gaussian_samples = self.get_node_samples(start_idx, int(self.num_agent*self.gaussian_prm.starts_weight[i]))

            for sample in gaussian_samples:
                self.agent_locations[curr_agent_idx] = sample
                self.node_agents[start_idx].append(curr_agent_idx)
                curr_agent_idx += 1
        
        for t in range(1, self.timestep+1):
      
            print("time", t)
            # Reset Agent assingment
            self.agent_assigned = [False] * self.num_agent

            # Index incoming flows for each target node
            incoming_flow = defaultdict(list)
            for prev_node in self.macro_solution[t].keys():
                for node, flow in self.macro_solution[t][prev_node]:
                    incoming_flow[node].append((prev_node, flow))
            
            next_node_agents = []
            trajectories = []
            
            moving_agents = 0
            for node in incoming_flow.keys():
                for flow in incoming_flow[node]:
                    moving_agents += flow[1]
            print("moving agents", moving_agents)

            # sample new locations in goal nodes
            for node in incoming_flow.keys():
                # gather all agents coming to the same goal
                incoming_agents = []
                for prev_node, flow in incoming_flow[node]:
                    agents_idx = self.choose_unassigned_agents(prev_node, node, flow) 
                    incoming_agents += agents_idx
                    for agent in agents_idx:
                        self.agent_assigned[agent] = True

                gaussian_samples = self.get_node_samples(node, len(incoming_agents))

                # update agent locations
                trajectories.append(self.update_agent_locations(incoming_agents, gaussian_samples))

                # store next step agent list 
                next_node_agents.append((node, incoming_agents))


            # update next node agents location
            for node_idx, node_agent in next_node_agents:
                self.node_agents[node_idx] = node_agent

            for trajectory in trajectories:
                for agent_idx, goal in trajectory.items():
                    self.agent_locations[agent_idx] = goal
                    solution[agent_idx].append(goal)
        return solution
    