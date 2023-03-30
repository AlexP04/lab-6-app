import numpy as np
import networkx as nx
import scipy

class Map:
    # Конструктор - на вхід задається матриця для графу та імена вузлів 
    def __init__(self, adj_matrix, node_names):
        adj_matrix = np.array(adj_matrix).T
        node_names = np.array(node_names)
        
        # Ініціалізуємо матрицю зв'язків та зберішаємо як граф (двонапрямлений)
        self.graph = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)
        self.matrix = adj_matrix
        
        # Ініціалізуємо назви вузлів та ініціалізуємо ці параметри для вузлів вищенаведеного об'єкту
        self.node_names = node_names
        nx.set_node_attributes(self.graph, node_names, 'name')
        
        # отримаємо список простих циклів графу
        self.cycles = [c for c in nx.simple_cycles(self.graph)]
    
    # Виведення матриці
    def getMatrix(self):
        return self.matrix
    
    # Виведення циклів - усі, чи тільки парні
    def getCycles(self, only_even = False):
        if only_even:
            return [c for c in self.cycles if self.__evenCheck( c, self.getMatrix() )]
        else:
            return self.cycles
    
    #@staticmethod
    # визначення чи є цикл парним
    def __evenCheck(self, cycle, adj_matrix=None):
        # копія циклу
        cycle_copy = cycle.copy()
        cycle_copy.append(cycle[0])
        
        # проходження по циклу та підрахунок вершин (кількості проходів - граней)
        sign = 1
        for i in range(len(cycle_copy) - 1 ):
            sign *= np.sign(adj_matrix[cycle_copy[i], cycle_copy[i+1]])
        return sign == 1
    
    # Беремо радіус власних чисел (тобто малдуль найбільшого) - радіус спектру
    def getRadius(self):
        return np.max(np.abs( self.getEigenvalues() ))

    # Визначення, чи когнітивна карта стабільна (стійка)
    def isStable(self):
        return self.getRadius() <= 1
    
    # Визначення, чи когнітивна карта є абсолютна стабільною
    def isStable2(self):
        return self.getRadius() < 1

    # Отримаємо список власних векторів
    def getEigenvalues(self):
        return np.linalg.eigvals(self.getMatrix())
            
    # Випробовування ситсеми пенвим імпульсом (impulse) та кількість кроків (ітерацій)
    def impulse(self, impulse, steps):
        #Визначаємо кількість вузлів та ініціалізуємо вектори та матриці
        N = len(self.node_names)
        x_0 = np.zeros((N, 1))
        x = [x_0, x_0]
        
        # імпульс
        q = np.array(impulse).reshape((N, 1))
        
        # Матриця зв'зків графу
        A = self.getMatrix()
        
        x_n = x[-1] + A @ (x[-1] - x[-2]) + q
        x.append(x_n)
        
        for _ in range(1, steps):
            # Використовується формула для збурення на кожному кроці
            x_n = x[-1] + A @ (x[-1] - x[-2])
            x.append(x_n)

        # Результат - ланцюг x для кожного кроку
        res = np.array(x)[1:, :, 0]

        return res
