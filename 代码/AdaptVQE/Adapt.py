import math
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.second_q.circuit.library.ansatzes.utils import generate_fermionic_excitations
from qiskit_nature.second_q.operators import FermionicOp,SparseLabelOp
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.circuit import Instruction
from typing import List,Sequence

class AdaptVQE_Y(object):
    def __init__(self,es_problem:ElectronicStructureProblem) -> None:
        #es_problem 包含了molecular driver transform
        self.problem = es_problem
        self.problem_second_op = es_problem.second_q_ops()
        converter = QubitConverter(JordanWignerMapper()) #JW_converter
        self.molecular_hamiltoninan = converter.convert(self.problem_second_op['ElectronicEnergy']) #二次量子化后的哈密顿量
        self.initial_state = HartreeFock(num_particles=es_problem.num_particles,num_spin_orbitals=es_problem.num_spin_orbitals,qubit_converter=converter)
        self.mapper = JordanWignerMapper()
        self.get_operator_pool()
        self.get_operator_component()
        
        # if len(self.operator_pool) ==0:
        #     raise ValueError("算符池为空！")
                
        
    def get_operator_component(self)->List[List[Instruction]]:
        if len(self.opertor_pool_1)!= 0:
            self.opertor_component_1 =[]
            self.opertor_component_2 =[]
            for i,k in enumerate(self.opertor_pool_1):
                self.opertor_component_1.append(EvolvedOperatorAnsatz(operators=k).to_instruction(label='single_'+str(i)))
            
            for i,k in enumerate(self.opertor_pool_2):
                self.opertor_component_2.append(EvolvedOperatorAnsatz(operators=k).to_instruction(label='double_'+str(i)))
        
            
                
                
            
        
    def get_operator_pool(self):
        #单激发算符T1-T1_dagger
        excitations_1 = generate_fermionic_excitations(num_excitations=1,
                                       num_particles=self.problem.num_particles,
                                       num_spatial_orbitals=int( self.problem.num_spin_orbitals//2) )
        #双激发算符T2-T2_dagger
        excitations_2 = generate_fermionic_excitations(num_excitations=2,
                                       num_particles=self.problem.num_particles,
                                       num_spatial_orbitals=int(self.problem.num_spin_orbitals//2))
        # print('excitation_list:',excitations_1)
        
        #operator_pool 是Tn-Tn_dagger 的Pauil_OP形式 
        
        self.excitations_1_fermionop,self.opertor_pool_1 = self.excitations_to_ferop(excitations_list=excitations_1)
        self.excitations_2_fermionop,self.opertor_pool_2  =self.excitations_to_ferop(excitations_list=excitations_2)
 

    def excitations_to_ferop(self,excitations_list)->list[SparseLabelOp]:
        operators = []
        pauli_op = []
        #把[((0,), (1,)), ((2,), (3,))] 转换成FermionOp
        for exc in excitations_list:
            label = []
            for occ in exc[0]:
                label.append(f"+_{occ}")
            for unocc in exc[1]:
                label.append(f"-_{unocc}")
            op = FermionicOp({" ".join(label): 1}, num_spin_orbitals=self.problem.num_spin_orbitals,copy=True)
            op -= op.adjoint()
            # we need to account for an additional imaginary phase in the exponent (see also
            # `PauliTrotterEvolution.convert`)
            op *= 1j  # type: ignore
            operators.append(op)
        for i,k in enumerate(operators):
            pauli_op.append(self.mapper.map(k))
    
        return operators,pauli_op
    
    
    
    @property
    def hamiltoinian(self):
        #print('2次量子化后的哈密顿量为:\n')
        return self.molecular_hamiltoninan
    
    @property
    def problem_second(self):
        print('2次量子化后的哈密顿量为:\n')
        return self.problem_second_op
    
    
    