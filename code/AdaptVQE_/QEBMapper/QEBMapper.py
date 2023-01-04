from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.opflow import I,X,Y

#用于对激发列表二次量子化成
class QEBMapper(object):
    def __init__(self,Pauli_length:int,Excitationlist:list) -> None:
        self.length = Pauli_length
        self.excitation_list = Excitationlist
        for i in Excitationlist:
            if len(i[0])==1:
                #双激发算符
                T = self.Q(i[0][0]*1j)@self.Q_dagger(i[1][0])
                T_dagger = T.adjoint()
            print(T-T_dagger)

    
    
    def convert(self,exctiation_list:tuple):
        for i in exctiation_list:
            
            pass
        
        
        pass
    
    #对应着JW变换中的
    #Qubit-湮灭算符
    
    #index从0开始 高比特位在前 低位在后：eg: a2_dagger = IXII -1j IYII
    def Q_dagger(self,index:int):
        if index==0:
            operator=(X-1j*Y)
        else:
            operator=I
        for i in range(1,self.length):
            if index==i:
                operator = (X-1j*Y)^operator
            else:
                operator= I^operator
        return operator
    #Qubit-产生算符
    #index从0开始 高比特位在前 低位在后：eg: a2 = IXII +1j IYII
    def Q(self,index:int):
        if index==0:
            operator=(X+1j*Y)
        else:
            operator=I
        for i in range(1,self.length):
            if index==i:
                operator = (X+1j*Y)^operator
            else:
                operator= I^operator
        return operator
    
class QEB_Operator(object):
    def __init__(self) -> None:
        print('sds')
    