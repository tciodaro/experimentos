
import Tests



print('==== STARTING TESTS ====')

# Loop over default tests
n_success = 0
for test in Tests.test_pipeline:
        (status, solver) = test()
        n_success += status

if n_success != len(Tests.test_pipeline):
        print('=> Errors in some tests')
else:
        print('=> Successfully finished!')





# END OF FILE


