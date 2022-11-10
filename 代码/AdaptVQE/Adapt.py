import math
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper, BravyiKitaevMapper, JordanWignerMapper

class AdaptVQE_Y(object):
    def __init__(self,es_problem:ElectronicStructureProblem) -> None:
        #es_problem 包含了molecular driver transform
        self.problem_second_op = es_problem.second_q_ops()
        converter = QubitConverter(JordanWignerMapper())
        self.molecular_hamiltoninan = converter.convert(self.problem_second_op[0])
        
        
    @property
    def hamiltoinian(self):
        #print('2次量子化后的哈密顿量为:\n')
        return self.molecular_hamiltoninan
    
    @property
    def problem_second(self):
        print('2次量子化后的哈密顿量为:\n')
        return self.problem_second_op
    