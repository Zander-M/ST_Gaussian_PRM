#  Space-Time SwarmPRM

We add time dimesion for SwarmPRM. Number of agent on each node is constrained by the available area in each node.

Macroscopically, we have GMMs representing groups of agents travelling on the map. The GMM has a max capacity that does not exceed the current maps max flow. 

GMM does not constraint the number of agents in each distribution. We therefore constraint the 

## Discussion

The Gaussian map does not have 

## Algorithm

The travese capacity corresponds to the max flow from the start distributions to goal distributions

During transportation, several agents need to wait at certain nodes for other agents to finish first. 
