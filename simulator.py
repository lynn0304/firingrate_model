import numpy as np
import matplotlib.pyplot as plt
from network import Network


if __name__ == "__main__":

    # 1 time unit here is 1s 

    # Network(weight_file='cx_model_table.xls', dt=1)
    net = Network(dt=1)


    net.add_neuron("E")
    net.add_neuron("I")

    # set "N" / "Taum"
    net.set_neuron_param("E", "Taum", 3)
    net.set_neuron_param("I", "Taum", 5)

    # add_target(pre, post, weight)
    net.add_target("E", "E", 1.0)
    net.add_target("E", "I", 0.5)
    net.add_target("I", "E", -0.5)

    # if using a connection table 
    # net.add_connection(sheet_name='EPG_EPG')

    # add_event(strat_time, end_time, target, weight)
    net.add_event(0, 30, "I", 0.4)
    net.add_event(0, 15, "E", 5.0)
    net.add_event(30, "EndTrial")

    # set_ltp(pre, post, learning_rate=5, pre_threshold=1, post_threshold=1)
    net.set_ltp("E", "I")

    # simulate(solver='EULER', active_func='sqrt', a=0, b=0, group=True)
    # ative_func = 'sqrt' or 'sigmoid' or 'ReLU'
    # group part unfinished
    # only euler by now
    net.simulate()


    print(np.max(net.neu["E"][1].firing_rate))
    print(np.max(net.neu["I"][1].firing_rate))
    plt.plot(net.neu["E"][1].firing_rate, label='E')
    plt.plot(net.neu["I"][1].firing_rate, label='I')
    plt.legend()
    plt.savefig('firing_rate.png')

    # output .conf .pro, just for debugging
    net.output()
