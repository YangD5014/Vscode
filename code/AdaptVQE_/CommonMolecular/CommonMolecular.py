from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.drivers import PySCFDriver



class LiH(object):
    def __init__(self,dist:float) -> None:
        self.molecule = MoleculeInfo(
            ["Li", "H"], [(0.0, 0.0, 0.0), (dist, 0.0, 0.0)],
            multiplicity=1,  # = 2*spin + 1
            charge=0,)
        driver = PySCFDriver().from_molecule(self.molecule) 
        self.problem = driver.run()
        
class BeH2(object):
    def __init__(self,dist:float) -> None:
        self.molecule = MoleculeInfo(
            ["Be", "H","H"], [(0.0, 0.0, 0.0), (dist*1, 0.0, 0.0), (dist*-1, 0.0, 0.0)],
            multiplicity=1,  # = 2*spin + 1
            charge=0,)
        driver = PySCFDriver().from_molecule(self.molecule) 
        self.problem = driver.run()
        
class H2(object):
    def __init__(self,dist:float) -> None:
        self.molecule = MoleculeInfo(
            ["H", "H"], [(0.0, 0.0, 0.0), (dist, 0.0, 0.0)],
            multiplicity=1,  # = 2*spin + 1
            charge=0,)
        driver = PySCFDriver().from_molecule(self.molecule) 
        self.problem = driver.run()


class H2(object):
    def __init__(self,dist:float) -> None:
        self.molecule = MoleculeInfo(
            ["H", "H"], [(0.0, 0.0, 0.0), (dist, 0.0, 0.0)],
            multiplicity=1,  # = 2*spin + 1
            charge=0,)
        driver = PySCFDriver().from_molecule(self.molecule) 
        self.problem = driver.run()
        
class H4(object):
    def __init__(self,dist:float) -> None:
        self.molecule = MoleculeInfo(
            ["H", "H","H","H"], [(0.0, 0.0, 0.0), (dist*1, 0.0, 0.0), (dist*2, 0.0, 0.0), (dist*3, 0.0, 0.0)],
            multiplicity=1,  # = 2*spin + 1
            charge=0,)
        driver = PySCFDriver().from_molecule(self.molecule) 
        self.problem = driver.run()
        
class H6(object):
    def __init__(self,dist:float) -> None:
        self.molecule = MoleculeInfo(
            ["H", "H","H","H"], [(0.0, 0.0, 0.0), (dist*1, 0.0, 0.0), (dist*2, 0.0, 0.0), (dist*3, 0.0, 0.0), (dist*4, 0.0, 0.0), (dist*5, 0.0, 0.0)],
            multiplicity=1,  # = 2*spin + 1
            charge=0,)
        driver = PySCFDriver().from_molecule(self.molecule) 
        self.problem = driver.run()