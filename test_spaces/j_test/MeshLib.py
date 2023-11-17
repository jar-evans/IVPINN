### Nodes dataframe -> id, coords, elements, ext/int
### edges df -> id, node_pair, elements, ext/int
### triangle df -> node_makeup, edge_makeup

# option to create a vpinn mesh (one fidelity) and a ivpinn mesh (twofidelity)

# create standard delaunay mesh of domain and compile everything into dfs <- VPINN
# do the current way, but smartly compile all info into the dfs, s.t. the overview is more global as opposed to multi-local <- IVPINN

# make sure the inteface to all the data/info is easy and simple to use (otherwise m will forget it exists)


class MeshLib:

    def __init__(self) -> None:
        pass

    def 