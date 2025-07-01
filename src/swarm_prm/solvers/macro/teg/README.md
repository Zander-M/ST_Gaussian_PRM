# Time Expanded Graph Two Phase

This is a two-phased Time Expanded Graph search. We first find a min-timestep
solution using max flow on incrementing Time Expanded Graph. After finding a valid
solution, we fixed the current TEG and flow graph and try to find a min cost flow
based on the current solution to minimize the transport cost.

When used for multi-swarm planning, current swarm can use a node used by previous swarm if the capacity of a node's capacity is not exhausted.
