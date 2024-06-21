import re
import numpy as np
from cga import CGA


class SAT_CGA(CGA):
    def __init__(self, pop_size, chrome_size, clauses):
        self.clauses = clauses
        super().__init__(pop_size, chrome_size)

    def fitness_func(self, chromosome):
        summation = 0
        for clause in self.clauses:
            for lit in clause:
                if (1 if chromosome[abs(lit)-1] == 0 else 0) if (abs(lit)/lit) == -1 else (
                chromosome[abs(lit)-1]) == 1:
                    summation += 1
                    break

        return summation


def main():
    # Leer el archivo en formato cnf
    clauses = None
    with open('uf20-01.cnf', mode='r', encoding='utf-8') as f:
        match = None
        while match == None:
            line = f.readline()
            if line == '':
                raise Exception('El formato del archivo no es cnf')
            match = re.search('clause length = (\d{1,2})', line)
        clause_length = int(match.group(1))

        match = None
        while match == None:
            line = f.readline()
            if line == '':
                raise Exception('El formato del archivo no es cnf')
            match = re.search('^p cnf (\d{1,2}) {1,2}(\d{1,2}) *$', line)
        variables_num, clauses_num = int(match.group(1)), int(match.group(2))

        clauses = np.empty((clauses_num, clause_length), dtype=np.int16)

        pattern = '^ ?' + ' '.join('(-?\d{1,2})' for _ in range(clause_length))
        for i in range(clauses_num):
            match = re.search(pattern, f.readline())
            for j in range(clause_length):
                clauses[i][j] = match.group(j+1)

    # Ejecutar el algoritmo genetico para encontrar la mejor solución
    best_solution = SAT_CGA(clauses_num, variables_num, clauses).run()
    print(f'Mejor solución: {best_solution}')

    return


if __name__ == '__main__':
    main()
