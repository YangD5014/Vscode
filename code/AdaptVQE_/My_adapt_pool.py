
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.drivers import PySCFDriver
from typing import List
from qiskit.circuit import Instruction, InstructionSet
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD,UCC
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit import QuantumCircuit
from copy import deepcopy
from qiskit.circuit import ParameterVector, Parameter
from qiskit.primitives import Estimator  # 意味着本地进行的模拟
import numpy as np
from qiskit.algorithms.optimizers import SPSA
import logging
# 定义记录器对象
logger = logging.getLogger('MyAdaptVQE_pool')
# 设置记录器级别
logger.setLevel(logging.DEBUG)
# 设置过滤器 只有被选中的可以记录
myfilter = logging.Filter('MyAdaptVQE_pool')
# 定义处理器-文件处理器
filehandler = logging.FileHandler(filename='./adapt_vqe1129.log', mode='w')
filehandler.addFilter(myfilter)
# 定义处理器-控制台处理器
concolehander = logging.StreamHandler()
concolehander.setLevel(logging.INFO)
# 记录器绑定handerler
logger.addHandler(filehandler)
logger.addHandler(concolehander)

class MyAdapt_pool(object):
    def __init__(self, ES_problem: ElectronicStructureProblem,custom_operation_pool=None,excitation_index:str='sd'):
        self.es_problem = ES_problem
        self.prolem_spatial_orbitals = ES_problem.num_spatial_orbitals
        self.problem_spin_orbitals = ES_problem.num_spin_orbitals
        self.converter = QubitConverter(JordanWignerMapper())  # 二次量子化转换器JW
        self.hamiltonian = self.converter.convert(
            ES_problem.hamiltonian.second_q_op())  # 二次量子化后的Hamiltonian
        self.optimizer = SPSA(maxiter=300)

        self.init_state_hf = HartreeFock(num_particles=self.es_problem.num_particles,
                                         num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                                         qubit_converter=self.converter)
        self.n_qubit = self.init_state_hf.num_qubits
        self._already_pick_index = []  # 存放被选择的index
        self.slover = VQE(estimator=Estimator(),ansatz=self.init_state_hf,optimizer=SLSQP())

        # 构造出UCCSD的激发算符
        if excitation_index=='sd':  
            ansatz = UCC(qubit_converter=self.converter,
                        num_particles=self.es_problem.num_particles,
                        num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                        initial_state=self.init_state_hf,
                        excitations='sd',
                        generalized=False  # 如果设为True ==> UCCGSD G=generalized
                        )
        elif excitation_index=='gsd':
            ansatz = UCC(qubit_converter=self.converter,
                        num_particles=self.es_problem.num_particles,
                        num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                        initial_state=self.init_state_hf,
                        excitations='sd',
                        generalized=True  # 如果设为True ==> UCCGSD G=generalized
                        )
        elif excitation_index=='gd':
            ansatz = UCC(qubit_converter=self.converter,
                        num_particles=self.es_problem.num_particles,
                        num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                        initial_state=self.init_state_hf,
                        excitations='d',
                        generalized=True  # 如果设为True ==> UCCGSD G=generalized
                        )   
        elif excitation_index=='d':
            ansatz = UCC(qubit_converter=self.converter,
                        num_particles=self.es_problem.num_particles,
                        num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                        initial_state=self.init_state_hf,
                        excitations='d',
                        generalized=False,
                        preserve_spin=True# 如果设为True ==> UCCGSD G=generalized
                        )   

        # UCCSD算符池 （二次量子化后）
        # uccsd.excitation_ops()=>算符池的算符列表 [FermionOp]
        if custom_operation_pool==None:
            self.excitation_pool = [self.converter.convert(i) for i in ansatz.excitation_ops()]
            # excitation_pool_instruction 是不包含hf的！
            self.excitation_pool_instruction = [EvolvedOperatorAnsatz(operators=i, insert_barriers=False, name='Term_'+str(index), parameter_prefix='term_'+str(index)).to_instruction()\
                                                for index, i in enumerate(self.excitation_pool)]
        else:
            self.excitation_pool = custom_operation_pool
            self.excitation_pool_instruction= [EvolvedOperatorAnsatz(operators=i, insert_barriers=False, name='Term_'+str(index), parameter_prefix='term_'+str(index)).to_instruction()
                                                for index, i in enumerate(self.excitation_pool)]

        self.commutors = [1j*(self.hamiltonian@i - i@self.hamiltonian) for i in self.excitation_pool]
