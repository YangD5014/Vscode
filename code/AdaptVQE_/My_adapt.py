
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.drivers import PySCFDriver
from typing import List,Sequence
from qiskit.circuit import Instruction, InstructionSet
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD,UCC
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.opflow import StateFn, CircuitStateFn
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EvolvedOperatorAnsatz
from copy import deepcopy
from qiskit.quantum_info import Pauli
from qiskit.circuit import ParameterVector, Parameter
from qiskit.primitives import Estimator  # 意味着本地进行的模拟
import numpy as np
from qiskit.algorithms.optimizers import SPSA
import logging
import time
# 定义记录器对象
logger = logging.getLogger('MyAdaptVQE')
# 设置记录器级别
logger.setLevel(logging.DEBUG)
# 设置过滤器 只有被选中的可以记录
myfilter = logging.Filter('MyAdaptVQE')
# 定义处理器-文件处理器
filehandler = logging.FileHandler(filename='./adapt_vqe1207.log', mode='w')
filehandler.addFilter(myfilter)
# 定义处理器-控制台处理器
concolehander = logging.StreamHandler()
concolehander.setLevel(logging.INFO)
# 记录器绑定handerler
logger.addHandler(filehandler)
logger.addHandler(concolehander)

# 0.4.6  0.5.0


class MyAdaptVQE(object):

    def __init__(self, ES_problem: ElectronicStructureProblem,custom_operation_pool=None):
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
        uccsd = UCCSD(qubit_converter=self.converter,
                      num_particles=self.es_problem.num_particles,
                      num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                      initial_state=self.init_state_hf,
                      generalized=False  # 如果设为True ==> UCCGSD G=generalized
                      )

        # UCCSD算符池 （二次量子化后）
        # uccsd.excitation_ops()=>算符池的算符列表 [FermionOp]
        if custom_operation_pool==None:
            self.excitation_pool = [self.converter.convert(i) for i in uccsd.excitation_ops()]
            # excitation_pool_instruction 是不包含hf的！
            self.excitation_pool_instruction = [EvolvedOperatorAnsatz(operators=i, insert_barriers=False, name='Term_'+str(index), parameter_prefix='term_'+str(index)).to_instruction()\
                                                for index, i in enumerate(self.excitation_pool)]
        else:
            self.excitation_pool = custom_operation_pool
            self.excitation_pool_instruction= [EvolvedOperatorAnsatz(operators=i, insert_barriers=False, name='Term_'+str(index), parameter_prefix='term_'+str(index)).to_instruction()
                                                for index, i in enumerate(self.excitation_pool)]

        self.commutors = [1j*(self.hamiltonian@i - i@self.hamiltonian)
                          for i in self.excitation_pool]
        self.commutors2 = [(self.hamiltonian@i - i@self.hamiltonian)
                           for i in self.excitation_pool]
        self.adapt_ansatz = []
        self.adapt_ansatz.append(self.circuit_measurement_first())
        self.converageflag = False

    # @property #只读property
    # def excitation_pool(self):
    #     return self.excitation_pool
    
    # @excitation_pool.setter  
    # def excitation_pool(self,excitation_list:Sequence):
    #     self.excitation_pool = excitation_list
    #     self.excitation_pool_instruction = [EvolvedOperatorAnsatz(operators=i, insert_barriers=False, name='Term_'+str(index), parameter_prefix='term_'+str(index)).to_instruction()
    #                                          for index, i in enumerate(self.excitation_pool)]
         
    
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

    def slover_optimise(self,parameter_circuit:QuantumCircuit):
        self.slover.ansatz = parameter_circuit
        raw_result = self.solver.compute_minimum_eigenvalue(self.hamiltonian)
        print(f'slover optimize result:{raw_result.optimal_point}&{raw_result.optimal_parameters}')
        
        
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
            start_time = time.time()
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
            logger.info(f'本轮用时{time.time()-start_time}')
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
        # print(f'下一个算符:第{np.argmax(value)}项被选定,此项梯度最大')
        logger.info(f'下一个算符:第{np.argmax(value)}项被选定,此项梯度最大')
        # return result
        return bound_circuit
    def run_tmp(self):
        optimal_point= [0.0]
        self.slover.ansatz = None
        start_time = time.time()
        logger.info(
            f'------------第{self.iteration_index}轮正在进行中--------------')
        logger.info(f'**目前已有{self._already_pick_index}**')
        qc = QuantumCircuit(self.n_qubit)
        qc.append(self.init_state_hf, range(self.n_qubit))
        for index, k in enumerate(self.adapt_ansatz):
                qc.append(k, range(self.n_qubit))

            # ------------------------------------------------------------
            # 每一轮的优化
        self.slover.ansatz = qc
        self.slover.initial_point = np.zeros(qc.num_parameters)
        logger.info(f'initial point是:{self.slover.initial_point}')
        vqe_result = self.slover.compute_minimum_eigenvalue(self.hamiltonian)
        optimal_parameter = vqe_result.optimal_point.tolist()
        logger.info(f'第{self.iteration_index}轮的优化结果是：{optimal_parameter}')
        bound_circuit = qc.bind_parameters(optimal_parameter)
        self.pick_next_operator(bound_circuit=bound_circuit)
        optimal_point = optimal_parameter.append(0.0)
        self.iteration_index += 1
    
    def run_slover(self):
        optimal_point= [0.0]
        while(self.converageflag==False):
            self.slover.ansatz =None
            start_time =time.time()
            logger.info(f'------------第{self.iteration_index}轮正在进行中--------------')
            logger.info(f'**目前已有{self._already_pick_index}**')
            qc = QuantumCircuit(self.n_qubit)
            qc.append(self.init_state_hf, range(self.n_qubit))
            for index, k in enumerate(self.adapt_ansatz):
                qc.append(k, range(self.n_qubit))
                
            # ------------------------------------------------------------
            # 每一轮的优化
            self.slover.ansatz = qc
            self.slover.initial_point = np.zeros(qc.num_parameters)
            logger.info(f'initial point是:{self.slover.initial_point}')
            vqe_result = self.slover.compute_minimum_eigenvalue(self.hamiltonian)
            optimal_parameter = vqe_result.optimal_point.tolist()
            logger.info(f'第{self.iteration_index}轮的优化结果是：{optimal_parameter}')
            bound_circuit = qc.bind_parameters(optimal_parameter)
            self.pick_next_operator(bound_circuit=bound_circuit)
            optimal_point = optimal_parameter.append(0.0)
            self.iteration_index += 1
        logger.info(
            f'===FINAL OUTCOME===\nOrder={self._already_pick_index}\nOptimal value={optimal_parameter}\nTotal iteration={self.iteration_index-1}')


class MyG_AdaotVQE():
    def __init__(self,ES_problem:ElectronicStructureProblem) -> None:
                ###logger set
        # 定义记录器对象
        self._logger2 = logging.getLogger('MyGAdaptVQE')
        # 设置记录器级别
        self._logger2.setLevel(logging.DEBUG)
        # 设置过滤器 只有被选中的可以记录
        myfilter = logging.Filter('MyGAdaptVQE')
        # 定义处理器-文件处理器
        filehandler = logging.FileHandler(filename='./Gadapt_vqe1212.log', mode='a')
        filehandler.addFilter(myfilter)
        formatter = logging.Formatter('%(asctime)s-%(levelname)s-\n%(message)s')
        filehandler.setFormatter(formatter)
        # 定义处理器-控制台处理器
        concolehander = logging.StreamHandler()
        concolehander.setLevel(logging.INFO)
        # 记录器绑定handerler
        self._logger2.addHandler(filehandler)
        self._logger2.addHandler(concolehander)
        
        # super(MyAdaptVQE,self).__init__()
        self.es_problem = ES_problem
        self.prolem_spatial_orbitals = ES_problem.num_spatial_orbitals
        self.problem_spin_orbitals = ES_problem.num_spin_orbitals
        self.converter = QubitConverter(JordanWignerMapper())
        self.hamiltonian = self.converter.convert(ES_problem.hamiltonian.second_q_op())  # 二次量子化后的Hamiltonian
        self.init_state_hf = HartreeFock(num_particles=self.es_problem.num_particles,
                                         num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                                         qubit_converter=self.converter)
        self.n_qubit = self.init_state_hf.num_qubits
        

                
        #convert过后的玻色子算符
        self.UCCD_op = [self.converter.convert(i) for i in UCC(num_particles=self.es_problem.num_particles,num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                           qubit_converter=self.converter,generalized=False,excitations='d').excitation_ops()]
        self.UCCGD_op =[self.converter.convert(i) for i in UCC(num_particles=self.es_problem.num_particles,num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                           qubit_converter=self.converter,generalized=True,excitations='d').excitation_ops()]
        self.UCCD_instrcution = [EvolvedOperatorAnsatz(operators=i, insert_barriers=True, name='D_'+str(index), parameter_prefix='D_'+str(index)).to_instruction()\
                                                for index, i in enumerate(self.UCCD_op)]
        self.UCCGD_instrcution = [EvolvedOperatorAnsatz(operators=i, insert_barriers=True, name='D_'+str(index), parameter_prefix='D_'+str(index)).to_instruction()\
                                                for index, i in enumerate(self.UCCGD_op)]
        
        self.conclude_operator_pool()
        
        
        

        
        
    def pick_UCCD_op(self,threshold = 1e-20):
        #threshold是指 选取第一次梯度绝对值大于多少(threshold)的算符入选 
        estimator = Estimator()
        circuit_d =[]
        commutors = [1j*(self.hamiltonian@i - i@self.hamiltonian) for i in self.UCCD_op]
        for i in self.UCCD_instrcution:
            qc = QuantumCircuit(self.n_qubit)
            qc.append(self.init_state_hf,range(self.n_qubit))
            qc.append(i,range(self.n_qubit))
            circuit_d.append(qc)
        job = estimator.run(circuits=circuit_d,observables=commutors,parameter_values=[[0.0]]*len(circuit_d))
        result = job.result()
        #梯度绝对值
        abs_value = np.abs(result.values)
        np_sort = np.argsort(abs_value)#存着index
        np_sort = np_sort[::-1] #从大到小排列
        #被选中的算符index
        pick_index = [i for i in np_sort if abs_value[i]>threshold]
        self._logger2.info(f'UCCD算符池筛选完毕!共入选{len(pick_index)}个,阈值是{threshold},具体index:\n{pick_index},\n\
            入选率:{len(pick_index)/len(self.UCCD_op)*100}%')
        return pick_index

        
        
        
    def pick_UCCGD_op(self,threshold = 1e-15):
        #threshold是指 选取第一次梯度绝对值大于多少(threshold)的算符入选 
        estimator = Estimator()
        circuit_d =[]
        commutors = [1j*(self.hamiltonian@i - i@self.hamiltonian) for i in self.UCCGD_op]
        for i in self.UCCGD_instrcution:
            qc = QuantumCircuit(self.n_qubit)
            qc.append(self.init_state_hf,range(self.n_qubit))
            qc.append(i,range(self.n_qubit))
            circuit_d.append(qc)
        job = estimator.run(circuits=circuit_d,observables=commutors,parameter_values=[[0.0]]*len(circuit_d))
        result = job.result()
        #梯度绝对值
        abs_value = np.abs(result.values)
        np_sort = np.argsort(abs_value)#存着index
        np_sort = np_sort[::-1] #从大到小排列
        #被选中的算符index
        pick_index = [i for i in np_sort if abs_value[i]>threshold]
        self._logger2.info(f'UCCGD算符池筛选完毕!共入选{len(pick_index)}个,阈值是{threshold},具体index:\n{pick_index},\n\
            入选率:{len(pick_index)/len(self.UCCGD_op)*100}%')
        return pick_index
    
    def conclude_operator_pool(self):
        #由于有算符重叠风险 因此需要进行去重
        self.UCCD_index = self.pick_UCCD_op(threshold=1e-15)
        self.UCCGD_index = self.pick_UCCGD_op(threshold=1e-15)
        self.operator_pool_femonic_op = [self.UCCD_op[i] for i in self.UCCD_index]
        tmp = [self.UCCGD_op[i] for i in self.UCCGD_index if self.UCCGD_op[i] not in self.operator_pool_femonic_op]
        self._logger2.info(f'UCCGD_index的入选个数为{len(tmp)},是否有筛掉的:{len(tmp)==len(self.UCCGD_index)}')
        self.operator_pool_femonic_op.extend(tmp)    
        
            
class QubitAdapt(object):
    #qubit-adapt 从UCCSD 以及UCCGSD(未实现)的Pauli string中提取算符作为qubit pool
    def __init__(self, ES_problem: ElectronicStructureProblem) -> None:
        self.logger_init() #初始化日志
        self.converter = QubitConverter(JordanWignerMapper())
        self.es_problem = ES_problem
        self.init_state_hf = HartreeFock(num_particles=self.es_problem.num_particles,
                                         num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                                         qubit_converter=self.converter)
        self.n_qubit = self.init_state_hf.num_qubits
        self.hamiltonian = self.converter.convert(ES_problem.hamiltonian.second_q_op())
        uccsd = UCCSD(qubit_converter=self.converter,
                      num_particles=self.es_problem.num_particles,
                      num_spatial_orbitals=self.es_problem.num_spatial_orbitals,
                      initial_state=self.init_state_hf,
                      generalized=False  # 如果设为True ==> UCCGSD G=generalized
                      )
        self.uccop = [self.converter.convert(i) for i in uccsd.excitation_ops()]
        self.uccop_dict = [self.pauliOperator2Dict(pauliOperator=i) for i in self.uccop]
        self.qubit_pool_init()
        self.adapt_ansatz=[]
        self.converageflag = False
        self.commutors = [1j*(self.hamiltonian@i - i@self.hamiltonian)
                          for i in self.qubit_pool_withoutz]
        self._already_pick_index = []
        self.slover = VQE(estimator=Estimator(),ansatz=self.init_state_hf,optimizer=SLSQP())
        self.adapt_ansatz.append(self.circuit_measurement_first())

        
    @staticmethod
    def pauliOperator2Dict(pauliOperator):
        paulis = pauliOperator.to_pauli_op().oplist
        paulis_dict = {}
        for x in paulis:
            if x.coeff == 0.0:
                continue
            label = x.primitive.to_label()
            coeff = x.coeff
            paulis_dict[label] = coeff

        return paulis_dict
    @staticmethod
    def random_pick():
        pass
    
    def qubit_pool_init(self):
        tmp = [list(i) for i in self.uccop_dict]
        self.paulistring_withz=[]
        for i in tmp:
            if len(i)==2:
                i.pop()
            elif len(i)==8:
                del i[4:8]
            self.paulistring_withz.append(i)
        self.paulistring_withoutz = [k.replace('Z','I')  for i in self.paulistring_withz for k in i]
        self.qubit_pool_withoutz = [PauliOp(primitive=Pauli(data=i),coeff=1j) for i in self.paulistring_withoutz]
        self.paulistring_withoutz_instruction = [EvolvedOperatorAnsatz(operators=PauliOp(Pauli(data=i),coeff=1j),name='Term_'+str(index),parameter_prefix="Qubit_"+str(index)).to_instruction()  \
                                                 for index,i in enumerate(self.paulistring_withoutz)]
        self.logger.info(f'Qubit pool 共有{len(self.qubit_pool_withoutz)}个')
        
        
    @staticmethod
    # 参数是 每一轮求得的梯度的最大值
    def check_gradient_converge(value, criterion: float = 1e-3) -> bool:
        converge = value[np.argmax(value)]
        if converge > criterion:
            logger.info(f'没有达到收敛标准,标准为{criterion},当前值为{converge}')
            return False
        else:
            logger.info(f'达到收敛标准,标准为{criterion},当前值为{converge}')
            return True
    
    #挑选下一块
    def pick_next_operator(self, bound_circuit: QuantumCircuit):
        self.logger.info(f'目前这是第{self.iteration_index}轮,正在准备挑选下一块算符,并验证是否收敛...')
        estimator = Estimator()
        job = estimator.run(
            circuits=[bound_circuit]*len(self.commutors), observables=self.commutors)
        result = job.result()
        value = np.abs(result.values)
        self.converageflag = self.check_gradient_converge(value=value)
        if self.converageflag == False:
            self.logger.info(f'目前还没有达到收敛！')
            k = np.argmax(value)
            self.logger.info(f'第{self.iteration_index}轮中梯度最大项为第{k}项,已被选入算符池...')
            self._already_pick_index.append(k)
            # print(f'第{self.iteration_index}轮中梯度最大项为第{k}项,梯度最小项为第{np.argmin(value)}项')
            self.adapt_ansatz.append(self.excitation_pool_instruction[k])
            return result.values
        else:
            self.logger.info(f'已经达到收敛！算法终止！')
            # print(f'第{self.iteration_index}轮已经收敛！')
    
    def logger_init(self):
        # 定义记录器对象
        self.logger = logging.getLogger('Qubit_AdaptVQE')
        # 设置记录器级别
        self.logger.setLevel(logging.DEBUG)
        # 设置过滤器 只有被选中的可以记录
        myfilter = logging.Filter('Qubit_AdaptVQE')
        # 定义处理器-文件处理器
        filehandler = logging.FileHandler(filename='./Qubit_AdaptVQE.log', mode='a')
        filehandler.addFilter(myfilter)
        formatter = logging.Formatter('%(asctime)s-%(levelname)s-\n%(message)s')
        filehandler.setFormatter(formatter)
        # 定义处理器-控制台处理器
        concolehander = logging.StreamHandler()
        concolehander.setLevel(logging.INFO)
        # 记录器绑定handerler
        self.logger.addHandler(filehandler)
        self.logger.addHandler(concolehander)
        
    
    def circuit_measurement_first(self):
        self.logger.info('开始初始化:挑选第一个算符...')
        circuits = []
        for i in self.paulistring_withoutz_instruction:
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
        self.logger.info(f'初始化结果:第{np.argmax(value)}项被选定,此项梯度最大')
        self._already_pick_index.append(k)
        # print(f'初始化结果:第{np.argmax(value)}项被选定，梯度最大,梯度最小的是第{np.argmin(value)}项')
        self.iteration_index = 1
        return self.paulistring_withoutz_instruction[k]
    
    def run_slover(self):
        optimal_point= [0.0]
        while(self.converageflag==False):
            self.slover.ansatz =None
            start_time =time.time()
            self.logger.info(f'------------第{self.iteration_index}轮正在进行中--------------')
            self.logger.info(f'**目前已有{self._already_pick_index}**')
            qc = QuantumCircuit(self.n_qubit)
            qc.append(self.init_state_hf, range(self.n_qubit))
            for index, k in enumerate(self.adapt_ansatz):
                qc.append(k, range(self.n_qubit))
                
            # ------------------------------------------------------------
            # 每一轮的优化
            self.slover.ansatz = qc
            #self.slover.initial_point = np.zeros(qc.num_parameters)
            self.logger.info(f'initial point是:{self.slover.initial_point}')
            vqe_result = self.slover.compute_minimum_eigenvalue(operator=self.hamiltonian)
            optimal_parameter = vqe_result.optimal_point.tolist()
            self.logger.info(f'第{self.iteration_index}轮的优化结果是：{optimal_parameter}')
            bound_circuit = qc.bind_parameters(optimal_parameter)
            self.pick_next_operator(bound_circuit=bound_circuit)
            optimal_point = optimal_parameter.append(0.0)
            self.iteration_index += 1
        self.logger.info(
            f'===FINAL OUTCOME===\nOrder={self._already_pick_index}\nOptimal value={optimal_parameter}\nTotal iteration={self.iteration_index-1}')

        
        
        

        
    

   
        
            
            