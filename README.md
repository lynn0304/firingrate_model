# firingrate_model

0303 update

* net.ouput -> can now choose specific neuron or group to output
    
    # output_population = 'AllPopulation' or 'group_name' or 'neuron_name'
    net.output(output_population='AllPopulation', filename_conf='network.conf', filename_pro='network.pro', group=True)
    
