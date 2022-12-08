
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.drivers import PySCFDriver
from typing import List
from qiskit.circuit import Instruction, InstructionSet
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.opflow import StateFn, CircuitStateFn
from qiskit.circuit import QuantumCircuit
from copy import deepcopy
from qiskit.circuit import ParameterVector, Parameter
from qiskit.primitives import Estimator  # 意味着本地进行的模拟
import numpy as np
from qiskit.algorithms.optimizers import SPSA
import logging
# 定义记录器对象
logger = logging.getLogger('MyAdaptVQE')
# 设置记录器级别
logger.setLevel(logging.DEBUG)
# 设置过滤器 只有被选中的可以记录
myfilter = logging.Filter('MyAdaptVQE')
# 定义处理器-文件处理器
filehandler = logging.FileHandler(filename='./adapt_vqe1129.log', mode='w')
filehandler.addFilter(myfilter)
# 定义处理器-控制台处理器
concolehander = logging.StreamHandler()
concolehander.setLevel(logging.INFO)
# 记录器绑定handerler
logger.addHandler(filehandler)
logger.addHandler(concolehander)

#0.4.6  0.5.0
class MyAdaptVQE(object):

    def __init__(self, ES_problem: ElectronicStructureProblem):
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

        # 构造出UCCSD的激发算符
        uccsd = UCCSD(qubit_converter=self.converter,
                      num_particles=self.es_problem.num_particles,
                      num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                      initial_state=self.init_state_hf,
                      generalized=False  # 如果设为True ==> UCCGSD G=generalized
                      )

        # UCCSD算符池 （二次量子化后）
        # uccsd.excitation_ops()=>算符池的算符列表 [FermionOp]
        self.excitation_pool = [self.converter.convert(
            i) for i in uccsd.excitation_ops()]
        # excitation_pool_instruction 是不包含hf的！
        self.excitation_pool_instruction = [EvolvedOperatorAnsatz(operators=i, insert_barriers=False, name='Term_'+str(index), parameter_prefix='term_'+str(index)).to_instruction()
                                            for index, i in enumerate(self.excitation_pool)]

        self.commutors = [1j*(self.hamiltonian@i - i@self.hamiltonian)
                          for i in self.excitation_pool]
        self.commutors2 = [(self.hamiltonian@i - i@self.hamiltonian)
                           for i in self.excitation_pool]
        self.adapt_ansatz = []
        self.adapt_ansatz.append(self.circuit_measurement_first())
        self.converageflag = False

    @staticmethod
    # 参数是 每一轮求得的梯度的平方之和
    def check_gradient_converge(value, criterion: float = 1e-3) -> bool:
        converge = value[np.argmax(value)]
        # converge = 0
        # total = 0
        # for i in value:
        #     total += i**2
        #     converge = np.sqrt(total)

        if converge > criterion:
            logger.info(f'没有达到收敛标准,标准为{criterion},当前值为{converge}')
            return False
        else:
            logger.info(f'达到收敛标准,标准为{criterion},当前值为{converge}')
            return True

    # 【关于梯度和参数优化的顺序】每一轮先前面参数不动（维持上一轮优化后的结果） 计算假如新的算符后
    # 用来计算第一轮的梯度 第一次只用hf态 作为psi
    def circuit_measurement_first(self):
        logger.info('开始初始化:挑选第一个算符...')
        circuits = []
        for i in self.excitation_pool_instruction:
            n = self.init_state_hf.num_qubits
            qc = QuantumCircuit(n)
            init_state = deepcopy(self.init_state_hf)
            qc.append(init_state, range(n))
            circuits.append(qc)

        estimator = Estimator()
        job = estimator.run(circuits=circuits, observables=self.commutors)
        result = job.result()
        value = np.abs(result.values)
        k = np.argmax(value)
        logger.info(f'初始化结果:第{np.argmax(value)}项被选定,此项梯度最大')
        self._already_pick_index.append(k)
        # print(f'初始化结果:第{np.argmax(value)}项被选定，梯度最大,梯度最小的是第{np.argmin(value)}项')
        self.iteration_index = 1
        return self.excitation_pool_instruction[k]

    def parameter_optimizer(self, parameterized_circuit: QuantumCircuit, maxiteration: int = 200, initial_parameter=None):
        if initial_parameter == None:
            initial_parameter = np.zeros(parameterized_circuit.num_parameters)
        # 输入为参数化线路 但尚未绑定参数：

        def _loss(x):
            bound_circuit = parameterized_circuit.bind_parameters(x)
            estimator = Estimator()
            job = estimator.run(circuits=bound_circuit,
                                observables=self.hamiltonian)
            result = job.result()
            return result.values[0]
        # print(f'初始设定参数值为{initial_parameter}')
        logger.info(f'初始设定参数值为{initial_parameter} 即将开始参数优化...')
        spsa = SPSA(maxiter=maxiteration)
        optimize_result = spsa.minimize(fun=_loss, x0=initial_parameter)
        logger.info(
            f'第{parameterized_circuit.num_parameters}轮的参数优化结果为{optimize_result.x}')
        # print(f'第{parameterized_circuit.num_parameters+1}轮的参数优化结果为{optimize_result.x}')
        return optimize_result.x

    # 函数功能 根据给定的绑定好优化参数的线路 从operator pool 选择下一个加入的算符：
    # 细节：比如第三步开始 将两个块(HF块 第一个算符块)构成psi
    def pick_next_operator(self, bound_circuit: QuantumCircuit):
        logger.info(f'目前这是第{self.iteration_index}轮,正在准备挑选下一块算符,并验证是否收敛...')
        estimator = Estimator()
        job = estimator.run(
            circuits=[bound_circuit]*len(self.commutors), observables=self.commutors)
        result = job.result()
        value = np.abs(result.values)
        self.converageflag = self.check_gradient_converge(value=value)
        if self.converageflag == False:
            logger.info(f'目前还没有达到收敛！')
            k = np.argmax(value)
            logger.info(f'第{self.iteration_index}轮中梯度最大项为第{k}项,已被选入算符池...')
            self._already_pick_index.append(k)
            # print(f'第{self.iteration_index}轮中梯度最大项为第{k}项,梯度最小项为第{np.argmin(value)}项')
            self.adapt_ansatz.append(self.excitation_pool_instruction[k])
            return result.values
        else:
            logger.info(f'已经达到收敛！算法终止！')
            # print(f'第{self.iteration_index}轮已经收敛！')

    def run(self):
        while(self.converageflag == False):
            logger.info(
                f'-----------------第{self.iteration_index}轮正在进行中-----------------------')
            logger.info(f'**目前已有{self._already_pick_index}**')
            qc = QuantumCircuit(self.n_qubit)
            qc.append(self.init_state_hf, range(self.n_qubit))
            for index, k in enumerate(self.adapt_ansatz):
                qc.append(k, range(self.n_qubit))
            # ------------------------------------------------------------
            # 每一轮的优化
            optimal_parameter = self.parameter_optimizer(
                parameterized_circuit=qc)
            # print(f'第{self.iteration_index}轮的优化结果是：{optimal_parameter}')
            bound_circuit = qc.bind_parameters(optimal_parameter)
            self.pick_next_operator(bound_circuit=bound_circuit)
            self.iteration_index += 1
        logger.info(
            f'===FINAL OUTCOME===\nOrder={self._already_pick_index}\nOptimal value={optimal_parameter}\nTotal iteration={self.iteration_index-1}')

    def check_nextone(self, order_list: List[int] = None):
        qc = QuantumCircuit(self.n_qubit)
        logger.info('================临时调用==================')
        qc.append(self.init_state_hf, range(self.n_qubit))
        for i in order_list:
            qc.append(self.excitation_pool_instruction[i], range(self.n_qubit))
        optimal_value = self.parameter_optimizer(parameterized_circuit=qc)
        bound_circuit = qc.bind_parameters(optimal_value)
        estimator = Estimator()
        job = estimator.run(
            circuits=[bound_circuit]*len(self.commutors), observables=self.commutors)
        result = job.result()
        value = np.abs(result.values)
        k = np.argmax(value)
        #print(f'下一个算符:第{np.argmax(value)}项被选定,此项梯度最大')
        logger.info(f'下一个算符:第{np.argmax(value)}项被选定,此项梯度最大')
        # return result
        return bound_circuit
