class Play:
	"""
	Data structure representing a molecule.

	:param nodes: When they are first parsed, nodes is a tensor of positional attributes of the player.
	:param edges: A list of tuples -- each tuple has two integers i and j representing a connection
	between the ith and jth nodes.
	:param label: np.long value of how many yards the play this node is part of rushed for.
	"""

	def __init__(self, nodes, edges, label):
		self.nodes = nodes
		self.edges = edges
		self.label = label
