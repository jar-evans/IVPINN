import meshio
import os
import matplotlib.pyplot as plt
import triangle as tr

class gmsh_worker:

    def __init__(self, geo_file: str):
        self.geo_file = geo_file

    def generate_parallel_chain(self, verbose: bool = False, delete: bool = True, write: bool = True):

        if write:
            self.run_geo()

        self.read_refinement_chain()
        self.decrypt_chain()
        if verbose:
            self.plot_chain()
        self.convert_to_Triangle()

        if delete:
            self.remove_all_msh()

    def read_msh(msh):
        """
        Takes an .msh file/object, and returns dictionary (Triangle or otherwise)
        """
        raise NotImplementedError()

    def run_geo(self):
        os.system(f"gmsh {self.geo_file + '.geo'} -2 -v 3")
        os.remove(self.geo_file + '.msh')

    def construct_base_geo(self, geo_core, lc, N, r):
        try:
            os.remove(self.geo_file + '.geo')
        except FileNotFoundError:
            pass

        f = open(self.geo_file + '.geo', 'w')
        f.write(f'lc = {lc};\n')
        with open(geo_core, 'r') as g:            
            f.write(g.read())
            f.write('\n')
        
        for i in range(N):
            f.write(f'Save "{i}_refined.msh";\n')
            if i == (N-1):
                break
            for j in range(r):
                f.write('RefineMesh;\n')
        f.close()

    @staticmethod
    def get_msh_files():
        tmp = [file for file in os.listdir() if file.endswith(".msh")]
        tmp.sort()
        return tmp
    def read_refinement_chain(self):
        self.chain = [meshio.read(f) for f in self.get_msh_files()]
    
    def decrypt_chain(self):
        decrypted = []
        for msh_obj in self.chain:
            triangle_cells = None
            for cell_data in msh_obj.cells:
                if cell_data.type == "triangle":
                    triangle_cells = cell_data
                    break
            decrypted.append({'vertices': msh_obj.points[:,0:2],
                       'triangles': triangle_cells.data})
        self.chain = decrypted
    
    def plot_chain(self) -> None:
            
        for mesh in self.chain:

            plt.figure(figsize=(4,4))
            plt.triplot(mesh['vertices'][:, 0],
                       mesh['vertices'][:, 1],
                       mesh['triangles'],
                       color='blue')
            plt.axis('off')
            plt.show()

    def convert_to_Triangle(self):
        self.chain = [
            tr.triangulate(mesh, 're') for mesh in self.chain
        ]

    def remove_all_msh(self):
        for file in self.get_msh_files():
            os.remove(file)



