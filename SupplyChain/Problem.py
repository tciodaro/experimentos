

from ortools.linear_solver import pywraplp as lp
import pandas
import numpy as np
import logging


WAREHOUSE_CAPACITY = 'WarehouseCapacity'
ARRIVAL_CUSTOMER = 'ArrivalCustomer'
ARRIVAL_WAREHOUSE= 'ArrivalWarehouse'
COST_EXTRA_SKU = 'Cost_Extra'
COST_PO = 'Cost_PO'
MAX_TIME_PAR = 'MaxTimeSimulation'
FAC_CAPACITY = 'FactoryCapacity'
FAC_LEAD_TIME= 'FactoryLeadTime'
FAC_PROD_TIME='FactoryProduction'
WARE_LEAD_TIME='WarehouseLeadTime'
DISCOUNTRATE = 'DiscountRate'
INITIALSTOCK = 'InitialStock'



def get_parameters(fname):
    wb = pandas.ExcelFile(fname)
    parameters = {}
    parameters['constants'] = wb.parse('Constants')
    parameters['forecast'] = wb.parse('Forecast')
    parameters['initials'] = wb.parse('InitialConditions')
    return parameters

class SupplyChainProblem(object):
    def __init__(self, **kwargs):
        self.debug = kwargs.get('debug')
        self.solver = None
        self.status = -1
        self.variables = {}
        self.constraints = {}
        self.parameters = {}

        self.warehouse = []
        self.factories = []
        self.skus = []
        self.customers = []
        self.timeslots = []
        self._tables = {}
        

    def configure(self, parameters):
        self.solver = lp.Solver('LinearExample', lp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        self.variables = {}
        self.constraints = {}
        self.factories  = parameters['constants'].Factory.dropna().unique()
        self.warehouses = parameters['constants'].Warehouse.dropna().unique()
        self.skus       = parameters['forecast'].SKU.dropna().unique()
        self.customers  = parameters['forecast'].Customer.dropna().unique()
        self.timeslots  = np.arange(int(parameters['constants'].Value[parameters['constants'].Constant == MAX_TIME_PAR].values[0]))
      
        self.parameters=parameters
        self._build_variables()
        self._build_constraints()
        self._build_objective()



    def _build_variables(self):
        parameters = self.parameters
        logging.info('==> Building Variables')        
        
        # Get the products that are being delivered in the warehouses
        # These initial conditions have to be subtracted from the Factory capacity
        # Initial Condition: product arrival at the warehouse. ArrWare(c,s,t)
        
        x_df = parameters['initials'][parameters['initials'].Variable == 'ArrivalWarehouse']
        x_df = x_df.groupby(['Factory','SKU','Time'], as_index=False)['Value'].sum()
        # Build X variables: X(f,s,w,t) = (0, UpdatedFactoryCapacity)
        logging.debug('X(f,s,w,t) pre-order variables')
        idx_p = parameters['constants'].Constant == FAC_CAPACITY
        idx_prod = parameters['constants'].Constant == FAC_PROD_TIME
        # Update capacity considering what is already beeing sent to the warehouses
        for f in self.factories:
            logging.debug('\tFactory: ' + f)
            idx_f = parameters['constants'].Factory == f
            for s in self.skus:
                logging.debug('\tSKU: ' + s)
                idx_s = parameters['constants'].SKU == s
                idx = idx_p & idx_f & idx_s
                prod_time = parameters['constants'].Value[idx_prod & idx_f & idx_s].values
                delivery = x_df[(x_df.Factory == f) & (x_df.SKU == s)].Value.values
                capacity = int(parameters['constants'].Value[idx].values[0]) if idx.sum() else 0
                capacity = np.ones(self.timeslots.shape) * capacity
                base_capacity = capacity.copy()

                if prod_time.shape != 0 and delivery.shape[0] != 0:
                    production = np.convolve(delivery, np.ones(prod_time.shape), 'full')
                    production = np.cumsum(production[::-1])[::-1]
                    production = np.pad(production, [0, self.timeslots.shape[0] - production.shape[0]], mode='constant')
                    capacity = capacity - production
                    capacity[capacity < 0] = 0
                
                for w in self.warehouses:
                    for t in self.timeslots:
                        vname = 'X_%s_%s_%s_t%i'%(f,s,w, t)
                        self.variables[vname] = self.solver.NumVar(0, capacity[t], vname)
                        logging.debug('\t\tX(%s,%s,%s,%i) = (0, %i)'%(f,s,w,t,capacity[t]))

        # Build E variables: E(c,s,t) = (0, inf)
        # Extra products delivered to the customer in order to make the solution feasible
        logging.debug('E(c,s,t) extra variables')
        for c in self.customers:
            for s in self.skus:
                for t in self.timeslots:
                    vname = 'E_%s_%s_t%i'%(c,s,t)
                    self.variables[vname] = self.solver.NumVar(0, self.solver.infinity(), vname)
                    logging.debug('\t\tE(%s,%s,%i) = (0, inf)'%(c,s,t))

        # Build D variables: D(c, s, t) = constant from the forecast
        logging.debug('D(c,s,t) demand variables')
        for c in self.customers:
            idx_c = parameters['forecast'].Customer == c
            for s in self.skus:
                idx_s = parameters['forecast'].SKU == s
                for t in self.timeslots:
                    vname = 'D_%s_%s_t%i'%(c,s,t)
                    idx = idx_s & idx_c & (parameters['forecast'].Time == t)
                    value = int(parameters['forecast'].Total[idx].values[0])
                    self.variables[vname] = self.solver.NumVar(value, value, vname)
                    logging.debug('\t\tD(%s,%s,%i) = (%i, %i)'%(c,s,t, value, value))

        # Initial Condition: product arrival at the customer. ArrCust(c,s,t)
        logging.debug('ArrCust(c,s,t) initial customer arrival variables')
        idx_p = parameters['initials'].Variable == ARRIVAL_CUSTOMER
        for c in self.customers:
            idx_c = parameters['initials'].Customer == c
            for s in self.skus:
                idx_s = parameters['initials'].SKU == s
                for t in self.timeslots:
                    vname = 'ArrCust_%s_%s_t%i'%(c,s,t)
                    idx = idx_p & idx_c & idx_s & (parameters['initials'].Time == t)
                    value = int(parameters['initials'].Value[idx].values[0] if idx.sum() else 0)
                    self.variables[vname] = self.solver.NumVar(value, value, vname)
                    logging.debug('\t\tArrCust(%s,%s,%i) = (%i, %i)'%(c,s,t, value, value))

        # Initial Condition: product arrival at the warehouse. ArrWare(w,s,t)
        logging.debug('ArrWare(w,s,t) initial warehouse arrival variables')
        idx_p = parameters['initials'].Variable == ARRIVAL_WAREHOUSE
        for w in self.warehouses:
            idx_w = parameters['initials'].Warehouse == w
            for s in self.skus:
                idx_s = parameters['initials'].SKU == s
                for t in self.timeslots:
                    vname = 'ArrWare_%s_%s_t%i'%(w,s,t)
                    idx = idx_p & idx_w & idx_s & (parameters['initials'].Time == t)
                    # Sum from all factories
                    value = int(parameters['initials'].Value[idx].sum() if idx.sum() else 0)
                    self.variables[vname] = self.solver.NumVar(value, value, vname)
                    logging.debug('\t\tArrWare(%s,%s,%i) = (%i, %i)'%(w,s,t, value, value))

        # Shipping from Warehouse to customer: SH(c, w, s, t)
        logging.debug('SH(c,w,s,t) warehouse shipping variables')
        idx_p = parameters['constants']['Constant'] == 'ShippingTripCapacity'
        for c in self.customers:
            idx_c = parameters['constants']['Customer'] == c
            for w in self.warehouses:
                idx_w = parameters['constants']['Warehouse'] == w
                for s in self.skus:
                    idx_s = parameters['constants']['SKU'] == s
                    capacity = parameters['constants'][idx_p & idx_w & idx_c & idx_s].Value.values[0]
                    # self.solver.infinity()
                    for t in self.timeslots:
                        vname = 'SH_%s_%s_%s_t%i'%(c,w,s,t)
                        self.variables[vname] = self.solver.NumVar(0, capacity, vname)
                        logging.debug(f'\t\tSH(%s,%s,%s,%i) = (0, {capacity})'%(c,w,s,t))

        ## STOCK VARIABLE:
        # Stock(w,s,t) = ArrivalWarehouse(w,s,t) + Stock(w,s,t-1) + SUM_f{X(f,w,s,t-LF)} - SUM_c{Shipped(c,w,s,t)}
        # The condition is:
        # Arrival(w,s,t) + Stock(w,s,t-1) + SUM_f{X(f,w,s,t-LF)} - SUM_c{Shipped(c,w,s,t)} - Stock(w,s,t) == 0

        logging.debug('S(w,s,t) warehouse stock variables')
        idx_p = parameters['constants'].Constant == WAREHOUSE_CAPACITY
        for w in self.warehouses :
            for s in self.skus:
                ## Set initial stock variable
                idx = (parameters['initials'].Warehouse == w) &\
                      (parameters['initials'].SKU == s)       &\
                      (parameters['initials'].Variable == INITIALSTOCK)
                initial_stock = int(parameters['initials'].Value[idx].values[0] if idx.sum() else 0)
                vname = 'StockInit_%s_%s'%(w,s)
                self.variables[vname] = self.solver.NumVar(initial_stock, initial_stock, vname)
                # Estoque maximo
                idx = ((parameters['constants'].Warehouse == w) & 
                       (parameters['constants'].SKU == s) &
                       (idx_p))
                capacity = int(parameters['constants'].Value[idx].values[0] if idx.sum() else 0)
                # Estoque minimo
                idx = ((parameters['constants'].Warehouse == w) & 
                       (parameters['constants'].SKU == s) &
                       (parameters['constants'].Constant == 'WarehouseMinStock'))
                min_stock = int(parameters['constants'].Value[idx].values[0] if idx.sum() else 0)
                print('ESTOQUE MIN:', min_stock)
                # Loop over time
                for t in self.timeslots:
                    # Stock capacity
                    # Create constraint
                    cname = 'C_Stock_%s_%s_%i'%(w,s,t)
                    self.constraints[cname] = self.solver.Constraint(0,0,cname)
                    debug_str = cname
                    # Create variable to summarize
                    vname = 'S_%s_%s_t%i'%(w,s,t)
                    self.variables[vname] = self.solver.NumVar(min_stock,capacity, vname)
                    debug_str += (': ' + vname + ' = ')

                    ## Add the variables to the constraint

                    # Initial stock variable
                    if t == 0:
                        vname = 'StockInit_%s_%s'%(w,s)
                        self.constraints[cname].SetCoefficient(self.variables[vname], 1)

                    # All products ordered from the factories: calculate lead time from factories
                    for f in self.factories:
                        idx = (parameters['constants'].Factory == f) & (parameters['constants'].Warehouse == w) & \
                              (parameters['constants'].Constant == FAC_LEAD_TIME)
                        lead_time = parameters['constants'].Value[idx].values[0]
                        idx = (parameters['constants'].Factory == f) & (parameters['constants'].SKU == s) & \
                            (parameters['constants'].Constant == FAC_PROD_TIME)
                        prod_time = parameters['constants'].Value[idx].values[0]
                        vname = 'X_%s_%s_%s_t%i'%(f, s, w, t - lead_time - prod_time)
                        if vname in self.variables.keys(): # negative 't' will not be found in the dictionary
                            self.constraints[cname].SetCoefficient(self.variables[vname], 1)
                            debug_str += (' + ' + vname)

                    # All products arriving from the initial conditions
                    vname = 'ArrWare_%s_%s_t%i'%(w,s,t)
                    self.constraints[cname].SetCoefficient(self.variables[vname], 1)
                    debug_str += (' + ' + vname)

                    # All products that were already in the stock
                    if 'S_%s_%s_t%i'%(w,s,t-1) in self.variables.keys():
                        vname = 'S_%s_%s_t%i'%(w,s,t-1)
                        self.constraints[cname].SetCoefficient(self.variables[vname],  1)
                        debug_str += (' + ' + vname)
                    
                    # Subtract all products being shipped to the customers (coefficient == -1)
                    for c in self.customers:
                        vname = 'SH_%s_%s_%s_t%i'%(c,w,s,t)
                        if vname in self.variables.keys():
                            self.constraints[cname].SetCoefficient(self.variables[vname], -1)
                            debug_str += (' - ' + vname)
                    # Subtract all products being left in the current stock (coefficient == -1)
                    vname = 'S_%s_%s_t%i'%(w,s,t) # we just created this variable. No need to check
                    self.constraints[cname].SetCoefficient(self.variables[vname], -1)

                    logging.debug(debug_str)




    def _build_constraints(self):
        parameters = self.parameters
        # Demmand constraint:
        # C_demand(c,s,t): D(c,s,t) - ArrCustomer(c,s,t) - SUM_w{Shipped(w,c,s,t-LW(w,c))} - E(c,s,t) == 0
        logging.debug('\tDemand Constraint')
        for c in self.customers:
            for s in self.skus:
                for t in self.timeslots:
                    # Create constraint
                    cname = 'C_Demand_%s_%s_%i'%(c,s,t)
                    self.constraints[cname] = self.solver.Constraint(0, 0, cname)
                    debug_str = cname

                    # Create variable
                    vname = 'D_%s_%s_t%i'%(c,s,t)
                    self.constraints[cname].SetCoefficient(self.variables[vname], 1)
                    debug_str += (': ' + vname + " = ")
                    

                    # All products shipped from the warehouses
                    for w in self.warehouses:
                        idx = (parameters['constants'].Warehouse == w) & (parameters['constants'].Customer == c) & \
                            (parameters['constants'].Constant == WARE_LEAD_TIME)
                        lead_time = parameters['constants'].Value[idx].values[0]
                        vname = 'SH_%s_%s_%s_t%i'%(c, w, s, t - lead_time)
                        if vname in self.variables.keys():
                            self.constraints[cname].SetCoefficient(self.variables[vname], -1)
                            debug_str += (' + ' + vname)
                    
                    # Subtract all products arriving to the customer (initial conditions)
                    vname = 'ArrCust_%s_%s_t%i'%(c,s,t)
                    self.constraints[cname].SetCoefficient(self.variables[vname], -1)
                    debug_str += (' + ' + vname)
                    
                    # Subtract all extra products shipped to the customer (solution feasibility)
                    vname = 'E_%s_%s_t%i'%(c,s,t)
                    self.constraints[cname].SetCoefficient(self.variables[vname], -1)
                    debug_str += (' + ' + vname)
                    
                    logging.debug(debug_str)
        



    def _build_objective(self):
        parameters = self.parameters        
        self.objective = self.solver.Objective()
        ## P.O. Costs (X variables)
        idx_p = parameters['constants'].Constant == COST_PO
        for f in self.factories:
            idx_f = parameters['constants'].Factory == f
            for s in self.skus:
                idx_s = parameters['constants'].SKU == s
                base_cost = parameters['constants'].Value[idx_p & idx_f & idx_s].values[0]
                # Get discount rate
                idx_discount = (parameters['constants'].Constant == DISCOUNTRATE) & idx_f & idx_s
                discount = parameters['constants'].Value[idx_discount].astype(float).values[0]
                for w in self.warehouses:
                    for t in self.timeslots:
                        cost = base_cost * (1+discount)**(-float(t))  
                        self.objective.SetCoefficient(self.variables['X_%s_%s_%s_t%i'%(f,s,w,t)], cost)
        
        
        # Variavel de envio (ship) SH(c, w, s, t)
        # ShippingCostPerSKU
        # ShippingCostPerTrip
#         idx_per_sku = (parameters['constants'].Constant == "ShippingCostPerSKU")
#         idx_per_trip = (parameters['constants'].Constant == "ShippingCostPerTrip")
#         for c in self.customers:
#             idx_c = parameters['constants']['Customer'] == c
#             for w in self.warehouses:
#                 idx_w = parameters['constants']['Warehouse'] == w
#                 cost_per_trip = parameters['constants'].Value[idx_per_trip & idx_w & idx_c].values[0]
#                 for t in self.timeslots:
#                     sku_ship = []
#                     for s in self.skus:
#                         idx_s = parameters['constants']['SKU'] == s
#                         cost = parameters['constants'].Value[idx_per_sku & idx_s & idx_w & idx_c]
                        
#                         self.objective.SetCoefficient(self.variables['SH_%s_%s_%s_t%i'%(c,w,s,t)],
#                                                       int(cost_per_trip / cost))
                        
#                         sku_ship.append(int(cost)/self.variables['SH_%s_%s_%s_t%i'%(c,w,s,t)])
                    

#                     # Custo total de envio
#                     cost_per_ship = self.solver.Sum(sku_ship) * cost_per_trip
                
        
        
        ## Extra SKU Costs (E variables)
        idx_p = parameters['constants'].Constant == COST_EXTRA_SKU
        for s in self.skus:
            idx_s = parameters['constants'].SKU == s
            cost = parameters['constants'].Value[idx_p & idx_s].values[0]
            for c in self.customers:
                for t in self.timeslots:
                    self.objective.SetCoefficient(self.variables['E_%s_%s_t%i'%(c,s,t)], int(cost))
        
        self.objective.SetMinimization()
        



    def run(self):
        parameters = self.parameters

        logging.info('=='*10)
        logging.info('\tTotal Factories : %i'%self.factories.shape[0])
        logging.info('\tTotal Warehouses: %i'%self.warehouses.shape[0])
        logging.info('\tTotal Customers : %i'%self.customers.shape[0])    
        logging.info('\tTotal Simulation: %i time slots'%self.timeslots.shape[0])
        logging.info('=='*10)

        # Solve the system.
        self.status = self.solver.Solve()

        # Check that the problem has an optimal solution.
        if self.status != lp.Solver.OPTIMAL:
            logging.info("The problem does not have an optimal solution!")
        else:
            logging.info('OPTIMAL SOLUTION FOUND')
            logging.info('Objective value: %i'%self.objective.Value())
            logging.info('#iterations    : %i'%self.solver.iterations())
            self._summarize()
            self._result_tables()
        return self.status == lp.Solver.OPTIMAL



    def _summarize(self):
        self._product_order   = np.zeros(self.timeslots.shape[0])
        self._extra_products  = np.zeros(self.timeslots.shape[0])
        self._stock           = np.zeros(self.timeslots.shape[0])
        self._shipping        = np.zeros(self.timeslots.shape[0])
        self._demand          = np.zeros(self.timeslots.shape[0])
        self._arrival_customer= np.zeros(self.timeslots.shape[0])
        self._arrival_warehouse= np.zeros(self.timeslots.shape[0])

        # Fill the variables if solution was found
        if self.status != lp.Solver.OPTIMAL:
            return

        # Loop over parameters
        for t in self.timeslots:
            # X Variables
            self._product_order[t] = np.sum([self.variables['X_%s_%s_%s_t%i'%(f,s,w,t)].solution_value()
                                             for f in self.factories
                                             for w in self.warehouses
                                             for s in self.skus])
            
            # E variables
            self._extra_products[t] = np.sum([self.variables['E_%s_%s_t%i'%(c,s,t)].solution_value()
                                              for c in self.customers
                                              for s in self.skus])

            # Stock
            self._stock[t] = np.sum([self.variables['S_%s_%s_t%i'%(w,s,t)].solution_value()
                                                    for w in self.warehouses
                                                    for s in self.skus])
             
            # Shipping
            self._shipping[t] = np.sum([self.variables['SH_%s_%s_%s_t%i'%(c,w,s,t)].solution_value()
                                        for w in self.warehouses
                                        for c in self.customers
                                        for s in self.skus])
            
            # Arrival Customer
            self._arrival_customer[t] = np.sum([self.variables['ArrCust_%s_%s_t%i'%(c,s,t)].solution_value()
                                                for c in self.customers
                                                for s in self.skus])
            
            # Arrival Warehouse
            self._arrival_warehouse[t] = np.sum([self.variables['ArrWare_%s_%s_t%i'%(c,s,t)].solution_value()
                                                 for c in self.warehouses
                                                 for s in self.skus])
                                   
            # Demand
            self._demand[t]  = np.sum([self.variables['D_%s_%s_t%i'%(c,s,t)].solution_value()
                                       for c in self.customers
                                       for s in self.skus])
        # Make a single pandas view
        self._tables['summary'] = pandas.DataFrame({
                                                    'Time': self.timeslots,
                                                    'PreOrder': self._product_order,
                                                    'ArrWare': self._arrival_warehouse,
                                                    'Stock': self._stock,
                                                    'Shipping': self._shipping,
                                                    'Extra': self._extra_products,
                                                    'ArrCust': self._arrival_customer,
                                                    'Demand': self._demand,
                                                    'CumShipping': np.cumsum(self._shipping + self._arrival_customer),
                                                    'CumExtra': np.cumsum(self._extra_products),
                                                    'CumDemand': np.cumsum(self._demand),
                                                })  



    def _result_tables(self):
        parameters = self.parameters
        # Create Pre Order Table
        df = pandas.DataFrame(columns=['Time', 'Factory', 'Warehouse', 'SKU', 'PreOrder'])
        for f in self.factories:
            for w in self.warehouses:
                for s in self.skus:
                    for t in self.timeslots:
                        df.loc[df.shape[0]] = [t, f, w, s, self.variables['X_%s_%s_%s_t%i'%(f,s,w,t)].solution_value()]
        self._tables['preorders'] = df

        # Create Stock table
        df = pandas.DataFrame(columns=['Time', 'Warehouse', 'SKU', 'Stock'])
        for w in self.warehouses:
            for s in self.skus:
                for t in self.timeslots:
                    df.loc[df.shape[0]] = [t, w, s, self.variables['S_%s_%s_t%i'%(w,s,t)].solution_value()]
        self._tables['stock'] = df

        # Create Shipping table
        df = pandas.DataFrame(columns=['Time', 'Warehouse', 'SKU','Customer', 'Shipping'])
        for w in self.warehouses:
            for s in self.skus:
                for c in self.customers:
                    for t in self.timeslots:
                        df.loc[df.shape[0]] = [t, w, s, c, self.variables['SH_%s_%s_%s_t%i'%(c,w,s,t)].solution_value()]
        self._tables['shipping'] = df

        # Create demand table
        df = pandas.DataFrame(columns=['Time', 'SKU','Customer', 'Demand'])
        for s in self.skus:
            for c in self.customers:
                for t in self.timeslots:
                    df.loc[df.shape[0]] = [t, s, c, self.variables['D_%s_%s_t%i'%(c,s,t)].solution_value()]
        self._tables['demand'] = df

        # Create extra products table
        df = pandas.DataFrame(columns=['Time', 'SKU', 'Customer', 'Extra'])
        for s in self.skus:
            for c in self.customers:
                for t in self.timeslots:
                    # Get the cost parameters
                    value=int(self.variables['E_%s_%s_t%i'%(c,s,t)].solution_value())
                    df.loc[df.shape[0]] = [t, s, c, value]
        self._tables['extra'] = df
        
        # Create cost table
        df = pandas.DataFrame(columns=['Time', 'SKU','Customer', 'CostPreOrder','CostExtra','CostTotal'])
        for s in self.skus:
            for c in self.customers:
                for t in self.timeslots:
                    # Get the cost parameters
                    idx = (parameters['constants'].Constant == COST_EXTRA_SKU) & (parameters['constants'].SKU == s)
                    extra_cost = self.variables['E_%s_%s_t%i'%(c,s,t)].solution_value() * (parameters['constants'].Value[idx].values[0])
                    po_cost = 0
                    for f in self.factories:
                        idx = (parameters['constants'].Constant == COST_PO) & (parameters['constants'].SKU == s) & (parameters['constants'].Factory == f)
                        cost = parameters['constants'].Value[idx].values[0]
                        po_cost += np.sum([self.variables['X_%s_%s_%s_t%i'%(f,s,w,t)].solution_value() for w in self.warehouses])*cost

                    df.loc[df.shape[0]] = [t, s, c, po_cost, extra_cost, po_cost + extra_cost]
        self._tables['cost'] = df





    def export_results(self, filename = './solver_results.xlsx'):
        if len(self._tables) == 0:
            raise Exception('Solver do not have valid results')

        # Export all tables to the workbook
        wb = pandas.ExcelWriter(filename)
        for key,tb in self._tables.items():
            tb.to_excel(wb, sheet_name=key, index=None)
        # Merge all tables
        df = self._tables['preorders']
        df = df.merge(self._tables['stock'], on=['Time', 'Warehouse', 'SKU'])
        df = df.merge(self._tables['shipping'], on=['Time', 'Warehouse', 'SKU'])
        df = df.merge(self._tables['demand'], on=['Time', 'Customer', 'SKU'])
        df = df.merge(self._tables['extra'], on=['Time', 'Customer', 'SKU'])
        df = df.merge(self._tables['cost'], on=['Time', 'SKU', 'Customer'])
        df.to_excel(wb, sheet_name='database', index=None)
        wb.close()



    def print_summary(self):
        parameters = self.parameters
        for t in self.timeslots:
            for c in self.customers:
                for s in self.skus:
                    po = sum([self.variables['X_%s_%s_%s_t%i'%(f,s,w,t)].solution_value() for f in self.factories for w in self.warehouses])
                    extra   = self.variables['E_%s_%s_t%i'%(c,s,t)].solution_value()
                    stock = np.sum([self.variables['S_%s_%s_t%i'%(w,s,t)].solution_value() for w in self.warehouses])
                    shipped = np.sum([self.variables['SH_%s_%s_%s_t%i'%(c,w,s,t)].solution_value() for w in self.warehouses])
                    arrivalc = self.variables['ArrCust_%s_%s_t%i'%(c,s,t)].solution_value()
                    arrivalw = np.sum([self.variables['ArrWare_%s_%s_t%i'%(w,s,t)].solution_value() for w in self.warehouses])
                    demand  = self.variables['D_%s_%s_t%i'%(c,s,t)].solution_value()
                    text = """%i\t%s\t%s\tP.O.: %i\t\tArrivalWare: %i\tStock: %i\tShipped: %i\tArrivalCust: %i\tExtra: %i\tDemand: %i"""\
                        %(t, c, s, po, arrivalw,stock,shipped,arrivalc, extra, demand)
                    print(text)

