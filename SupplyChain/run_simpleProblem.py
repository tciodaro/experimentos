
import Problem
import Tests
import logging
import pandas



# def main():

logging.basicConfig(level=logging.INFO)

fname = 'SupplyChain_SimpleProblem.xlsx'
logging.info('STARTING')

logging.info('=='*10)
logging.info('CONFIGURING PARAMETERS')
parameters = Problem.get_parameters(fname)
# print(parameters)

logging.info('=='*10)

# Run Solver
solver = Problem.SupplyChainProblem()
solver.configure(parameters)
solution_status = solver.run()

# Check solver status
if not solution_status:
    raise Exception('Solution not found!')

# Print output summary
# solver.print_summary()
pandas.options.display.max_rows = solver._tables['summary'].shape[0] 
pandas.options.display.max_columns = solver._tables['summary'].shape[1]
print(solver._tables['summary'])
print(solver._tables['summary'].sum(axis=0))



solver.export_results()



