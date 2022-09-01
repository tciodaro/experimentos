
import numpy as np 
import pandas
from colorama import Fore, Style
import Problem

# Header + data
constants = (
    ('Constant','Factory','Warehouse','SKU','Customer','Value'),
    ('FactoryProduction','F1',''  ,'S1',  '',  0),
    ('FactoryLeadTime'  ,'F1','W1',  '',  '',  0),
    ('FactoryCapacity'  ,'F1',  '','S1',  '',100),
    ('MaxTimeSimulation',  '',  '',  '',  '', 10),
    ('Cost_PO'          ,'F1',  '','S1',  '',10),
    ('Cost_Extra'       ,'F1',  '','S1',  '',100),
    ('WarehouseCapacity',  '','W1','S1',  '',100),
    ('WarehouseLeadTime',  '','W1',  '','C1',  0),
    ('DiscountRate'     ,'F1',  '','S1', '', 0.5),
    
)

forecasts = (
    ('Customer', 'SKU', 'Time', 'Total'),
    ('C1', 'S1', 0, 0),
    ('C1', 'S1', 1, 0),
    ('C1', 'S1', 2, 0),
    ('C1', 'S1', 3, 1),
    ('C1', 'S1', 4, 2),
    ('C1', 'S1', 5, 1),
    ('C1', 'S1', 6, 5),
    ('C1', 'S1', 7, 0),
    ('C1', 'S1', 8, 0),
    ('C1', 'S1', 9, 0),
)


initials = (
    ('Variable','Factory','Warehouse','SKU','Customer','Time','Value'),
    ('ArrivalWarehouse', 'F1', 'W1', 'S1',  '', 0, 0),
    ('ArrivalWarehouse', 'F1', 'W1', 'S1',  '', 1, 0),
    ('ArrivalCustomer' ,   '', 'W1', 'S1','C1', 0, 0),
    ('ArrivalCustomer' ,   '', 'W1', 'S1','C1', 1, 0),
    ('InitialStock'    ,   '', 'W1', 'S1',  '', 0, 0),
)




def get_default_data():
    databases = {
        'constants': pandas.DataFrame(data=np.array(constants[1:]), columns=constants[0]),
        'forecast' : pandas.DataFrame(data=np.array(forecasts[1:]), columns=forecasts[0]),
        'initials' : pandas.DataFrame(data=np.array(initials[1:]), columns=initials[0]),
    }
    # Fix data formats
    databases['constants']['Value'] = databases['constants']['Value'].astype(float)
    databases['forecast']['Total']  = databases['forecast']['Total'].astype(float)
    databases['forecast']['Time']   = databases['forecast']['Time'].astype(int)
    databases['initials']['Value']  = databases['initials']['Value'].astype(float)
    databases['initials']['Time']   = databases['initials']['Time'].astype(int)

    for key, df in databases.items():
        databases[key] = df.replace('', np.nan)

    return databases



# Run with standard data
def test_solver():
    print('==> test_solver:\t\t\t\t', end='')
    # Load default data
    parameters = get_default_data()
    # Run Solver
    solver = Problem.SupplyChainProblem(debug=True)
    solver.configure(parameters)
    solution_status = solver.run()
    # Check solver status
    if solution_status:
        print(Fore.GREEN, ' passed')
    else:
        print(Fore.RED, ' failed')

    print(Style.RESET_ALL, end='')
    return solution_status, solver



# Adjust the data to test it
def test_factory_production_delay():
    print('==> test_factory_production_delay:\t', end='')
    # 1) Simulate with the standard pattern
    base_parameters = get_default_data()
    solver = Problem.SupplyChainProblem(debug=False)
    solver.configure(base_parameters)
    solution_status = solver.run()
    if not solution_status:
        print(Fore.YELLOW, 'invalid')
        print(Style.RESET_ALL, end='')
        return solution_status, solver
    # Get output
    base_orders = solver._product_order.copy()
    # 2) Change the Coefficient linearly
    test_parameters = [0,1,2,3]
    count = 0
    for par in test_parameters:
        parameters = base_parameters.copy()
        idx = (parameters['constants'].Constant == Problem.FAC_PROD_TIME) &\
            (parameters['constants'].Factory == 'F1') &\
            (parameters['constants'].SKU == 'S1')
        parameters['constants'].loc[idx,'Value'] = par
        # Rerun solution
        solver = Problem.SupplyChainProblem(debug=False)
        solver.configure(parameters)
        solution_status = solver.run()
        if not solution_status:
            print(Fore.YELLOW, 'invalid')
            print(Style.RESET_ALL, end='')
            return solution_status, solver
        
        # 3) Measure the X variable for each coefficient value
        test_orders = solver._product_order.copy()
        
        # 4) Measure the delay: shift left the tested X (by the parameter value tested), make a bit-wise equal, sum the number of ones,
        count = count + int(np.array_equal(base_orders, np.roll(test_orders, par)))
       
    # Check how many ware correct
    if count == len(test_parameters):
        print(Fore.GREEN, ' passed')
    else:
        print(Fore.RED, ' failed')
    
    print(Style.RESET_ALL, end='')
    return True, (solver)



# Adjust the data to test it
def test_factory_lead_time():
    print('==> test_factory_lead_time:\t\t', end='')
    # 1) Simulate with the standard pattern
    base_parameters = get_default_data()
    solver = Problem.SupplyChainProblem(debug=False)
    solver.configure(base_parameters)
    solution_status = solver.run()
    if not solution_status:
        print(Fore.YELLOW, 'invalid')
        print(Style.RESET_ALL)
        return solution_status, solver
    # Get output
    base_orders = solver._product_order.copy()
    # 2) Change the Coefficient linearly
    test_parameters = [0,1,2,3]
    count = 0
    for par in test_parameters:
        parameters = base_parameters.copy()
        idx = (parameters['constants'].Constant == Problem.FAC_LEAD_TIME) &\
            (parameters['constants'].Factory == 'F1') &\
            (parameters['constants'].Warehouse == 'W1')
        parameters['constants'].loc[idx,'Value'] = par
        # Rerun solution
        solver = Problem.SupplyChainProblem(debug=False)
        solver.configure(parameters)
        solution_status = solver.run()
        if not solution_status:
            print(Fore.YELLOW, 'invalid')
            print(Style.RESET_ALL, end='')
            return solution_status, solver
        
        # 3) Measure the X variable for each coefficient value
        test_orders = solver._product_order.copy()
        
        # 4) Measure the delay: shift left the tested X (by the parameter value tested), make a bit-wise equal, sum the number of ones,
        count = count + int(np.array_equal(base_orders, np.roll(test_orders, par)))
       
    # Check how many ware correct
    if count == len(test_parameters):
        print(Fore.GREEN, ' passed')
    else:
        print(Fore.RED, ' failed')
    
    print(Style.RESET_ALL, end='')
    return True, (solver)



# Adjust the data to test it
def test_warehouse_leadtime():
    print('==> test_warehouse_leadtime:\t\t', end='')
    # 1) Simulate with the standard pattern
    base_parameters = get_default_data()
    solver = Problem.SupplyChainProblem(debug=False)
    solver.configure(base_parameters)
    solution_status = solver.run()
    if not solution_status:
        print(Fore.YELLOW, 'invalid')
        print(Style.RESET_ALL)
        return solution_status, solver
    # Get output
    base_orders = solver._product_order.copy()
    # 2) Change the Coefficient linearly
    test_parameters = [0,1,2,3]
    count = 0
    for par in test_parameters:
        parameters = base_parameters.copy()
        idx = (parameters['constants'].Constant == Problem.WARE_LEAD_TIME) &\
              (parameters['constants'].Customer == 'C1') &\
              (parameters['constants'].Warehouse == 'W1')
        parameters['constants'].loc[idx,'Value'] = par
        # Rerun solution
        solver = Problem.SupplyChainProblem(debug=False)
        solver.configure(parameters)
        solution_status = solver.run()
        if not solution_status:
            print(Fore.YELLOW, 'invalid')
            print(Style.RESET_ALL, end='')
            return solution_status, solver
        
        # 3) Measure the X variable for each coefficient value
        test_orders = solver._product_order.copy()
        
        # 4) Measure the delay: shift left the tested X (by the parameter value tested), make a bit-wise equal, sum the number of ones,
        count = count + int(np.array_equal(base_orders, np.roll(test_orders, par)))
       
    # Check how many ware correct
    if count == len(test_parameters):
        print(Fore.GREEN, ' passed')
    else:
        print(Fore.RED, ' failed')
    
    print(Style.RESET_ALL, end='')
    return True, (solver)



# Adjust the data to test it
def test_factory_capacity():
    print('==> test_factory_capacity:\t\t', end='')
    test_capacity = 3
    # 1) Simulate with the standard pattern
    base_parameters = get_default_data()
    solver = Problem.SupplyChainProblem(debug=False)
    solver.configure(base_parameters)
    solution_status = solver.run()
    if not solution_status:
        print(Fore.YELLOW, 'invalid')
        print(Style.RESET_ALL)
        return solution_status, solver
    # Get output
    base_orders = solver._product_order.copy()

    parameters = base_parameters.copy()
    idx = (parameters['constants'].Constant == Problem.FAC_CAPACITY) &\
          (parameters['constants'].SKU == 'S1')
    parameters['constants'].loc[idx,'Value'] = test_capacity
    # Rerun solution
    solver = Problem.SupplyChainProblem(debug=False)
    solver.configure(parameters)
    solution_status = solver.run()
    if not solution_status:
        print(Fore.YELLOW, 'invalid')
        print(Style.RESET_ALL, end='')
        return solution_status, solver
    test_orders = solver._product_order.copy()
    if np.max(base_orders) > test_capacity and np.max(test_orders) <= test_capacity:
        print(Fore.GREEN, ' passed')
    else:
        print(Fore.RED, ' failed')
    
    print(Style.RESET_ALL, end='')
    return True, (solver)



# FULL DEFAULT TESTS

test_pipeline = (
    # test_solver,
    test_factory_production_delay,
    test_factory_lead_time,
    test_warehouse_leadtime,
    test_factory_capacity,
    
)



# end of file


