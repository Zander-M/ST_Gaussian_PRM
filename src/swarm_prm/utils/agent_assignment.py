"""
    Util Functions
"""

def get_agent_assignment(num_agents, weights):
    """
        Assign agents according to weights. Redistribute the agents to respect the sum of agents
    """
    raw_allocation  = [f * num_agents for f in weights]
    int_allocation  = [int(f * num_agents) for f in weights]
    total_assigned = sum(int_allocation)
    remaining_agents = num_agents - total_assigned
    decimal_parts = [(i, raw_allocation[i] - int_allocation[i]) for i in range(len(weights))]
    decimal_parts.sort(key=lambda x: x[1], reverse=True)
    for i in range(remaining_agents):
        int_allocation[decimal_parts[i][0]] += 1  # Increment based on largest decimal parts
    return int_allocation