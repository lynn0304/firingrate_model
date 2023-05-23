from scipy.integrate import odeint
import numpy as np
import xlrd
import math

class Network():

    def __init__(self, name='', weight_file='cx_model_table.xls', dt=1):
        self.name = name
        self.dt = dt
        self.weight_file = weight_file
        self.reset()
    
    def reset(self):
        self.neu = {}
        self.group = {}
        self.synapse = {}
        self.eve = []
        self.EndTrial = 0

    def add_neuron(self, name, *args, **kwargs):
        neuron = NeuralPopulation(name, *args, **kwargs)
        self.neu[name] = (len(self.neu), neuron)
        return neuron

    def add_neuron_population(self, name, num, start=0, group=False):
        arr = []
        for i in range(start, start + num):
            neuron_name = f'{name}{i}'
            arr.append(self.add_neuron(neuron_name))
            if group:
                self.add_group(name, neuron_name)
        return arr

    def set_neuron_param(self, name, param, value):
        if param == 'Taum' and self.dt < 1:
            times = int(1/self.dt)
            value = value*times
        if name in self.group.keys():
            for neuron in self.group[name].member:
                setattr(self.neu[neuron][1], param, value)
        else:
            setattr(self.neu[name][1], param, value)

    def set_neuron_param_all(self, N, Taum):
        if self.dt < 1:
            times = int(1/self.dt)
        else:
            times = self.dt
        for neu in self.neu.values():
            neu[1].set_param(N, Taum*times)

    def add_target(self, pre_syn, post_syn, weight=0):
        if pre_syn in self.group.keys() and post_syn not in self.group.keys():
            for neu in self.group[pre_syn].member:
                if post_syn in self.neu[neu][1].target:
                    self.neu[neu][1].target[post_syn].weight = weight
                else:
                    self.neu[neu][1].add_target(post_syn, weight)
                if neu in self.neu[post_syn][1].upstream:
                    self.neu[post_syn][1].upstream[neu].weight = weight
                else:
                    self.neu[post_syn][1].add_upstream(neu, weight)
        elif post_syn in self.group.keys() and pre_syn not in self.group.keys():
            for neu in self.group[post_syn].member:
                if neu in self.neu[pre_syn][1].target:
                    self.neu[pre_syn][1].target[neu].weight = weight
                else:
                    self.neu[pre_syn][1].add_target(neu, weight)
                if pre_syn in self.neu[neu][1].upstream:
                    self.neu[neu][1].upstream[pre_syn].weight = weight
                else:
                    self.neu[neu][1].add_upstream(pre_syn, weight)
        elif post_syn in self.group.keys() and pre_syn in self.group.keys():
            for neu1 in self.group[pre_syn].member:         
                for neu2 in self.group[post_syn].member:        
                    if neu2 in self.neu[neu1][1].target:
                        self.neu[neu1][1].target[neu2].weight = weight
                    else:
                        self.neu[neu1][1].add_target(neu2, weight)
                    if neu1 in self.neu[neu2][1].upstream:
                        self.neu[neu2][1].upstream[neu1].weight = weight
                    else:
                        self.neu[neu2][1].add_upstream(neu1, weight) 
        else:
            if post_syn in self.neu[pre_syn][1].target:
                self.neu[pre_syn][1].target[post_syn].weight = weight
            else:
                self.neu[pre_syn][1].add_target(post_syn, weight)
            if pre_syn in self.neu[post_syn][1].upstream:
                self.neu[post_syn][1].upstream[pre_syn].weight = weight
            else:
                self.neu[post_syn][1].add_upstream(pre_syn, weight)

    def set_target_param(self, pre_syn, post_syn, param, value):
        setattr(self.neu[pre_syn][1].target[post_syn], param, value)

    def add_group(self, group_name, member_name, WTA=False):
        subgroup = [m for m in member_name if m in self.group.keys()]
        if subgroup:  # if there is sub_group_name in member_name, unpack the sub_group
            for g in subgroup:
                member_name = list(member_name)
                member_name.remove(g)
                member_name.extend(self.group[g].member)

        if group_name in self.group.keys():
            self.group[group_name].add_member_list(member_name)
        else:
            tmp_group = Group(group_name)
            tmp_group.add_member_list(member_name)
            self.group[group_name] = tmp_group
            
        if WTA==True:
            self.group[group_name].add_WTA()

    def add_event(self, time, event_type, *args):
        if self.dt < 1:
            times = int(1/self.dt)
        else:
            times = self.dt
        if type(event_type) == type(1):
            for i in range(time*times, event_type*times+1):
                if args[0] in self.group.keys():
                    for neu1 in self.group[args[0]].member:
                        event = Event(i, event_type*times, self.dt, neu1, args[1])
                        self.eve.append(event)
                else:
                    event = Event(i*times, event_type*times, self.dt, *args)
                    self.eve.append(event)
        else:
            event = Event(time*times, event_type*times, self.dt, *args)
            self.eve.append(event)
        if event_type == 'EndTrial':
            self.EndTrial = time*times
        return event

    def set_ltp(self, pre, post, learning_rate=5, pre_threshold=1, post_threshold=1):
        now_weight = self.neu[pre][1].target[post].weight
        now_time = len(self.neu[pre][1].firing_rate)
        pre_rate = self.neu[pre][1].firing_rate[now_time - 1]
        post_rate = self.neu[post][1].firing_rate[now_time - 1]
        new_weight = now_weight + learning_rate*(pre_rate - pre_threshold)*(post_rate - post_threshold)*self.dt
        self.neu[pre][1].target[post].weight = new_weight

    def output(self, output_population='AllPopulation', filename_conf='network.conf', filename_pro='network.pro', group=True):
        fout = open(filename_conf, 'w')
        for neu in self.neu.values():
            fout.write('%( '+ neu[1].name + ' --> ')
            idx = 0
            for post in neu[1].target:
                if neu[1].target[post].weight != 0:
                    if idx == len(neu[1].target)-1:
                        fout.write(post+':'+str(neu[1].target[post].weight)+' )\n')
                    else:
                        fout.write(post+':'+str(neu[1].target[post].weight)+', ')   
                idx += 1    
        if output_population == 'AllPopulation':    
            for neu in self.neu.values():
                neu[1].output(fout)
        elif output_population in self.neu.keys():
            self.neu[output_population][1].output(fout)
        elif output_population in self.group.keys():
            for neu in self.group[output_population].member:
                self.neu[neu][1].output(fout)

        if group:
            fout.write('-----------------------------------------------\n')
            for gro in self.group.values():
                gro.output(fout)
        
        fout = open(filename_pro, 'w')
        for event in self.eve:
            event.output(fout)

    def add_connection(self, sheet_name='EPG_EPG', type='excitation'):
        data = xlrd.open_workbook(self.weight_file)
        # table = data.sheets()[0], read by index
        table = data.sheet_by_name(sheet_name)
        nrows = table.nrows
        ncols = table.ncols
        for row in range(1, nrows):
            for col in range(1, ncols):
                if table.cell_value(row, col) != 0:
                    pre = str(table.cell_value(row, 0))
                    post = str(table.cell_value(0, col))
                    if type=='excitation':
                        weight = table.cell_value(row, col)
                    elif type=='inhibition':
                        weight = -table.cell_value(row, col)
                    self.add_target(pre, post, weight=weight)                        

    def simulate(self, solver='EULER', active_func='sqrt', a=0, b=0, group=True):
        def sort_event():
            l = len(self.eve)
            for i in range(l):
                for j in range(i, l):
                    if self.eve[j].time < self.eve[i].time:
                        temp = self.eve[j]
                        self.eve[j] = self.eve[i]
                        self.eve[i] = temp

        def func(h, mode, a=0, b=0):
            if mode == 'sqrt':
                if h<=0:
                    return 0
                else:
                    return math.sqrt(h)
            elif mode == 'sigmoid':
                return 1/(1+np.exp(-a*(h-b)))
            elif mode == 'ReLU':
                return max(0, h)
        
        def set_fire_arr():
            lengh = int(self.EndTrial)
            for neu in self.neu.values():
                neu[1].firing_rate = np.zeros((lengh+1))

        sort_event()
        set_fire_arr()
        eve_id = 0
        timestamp = 0
        event_happen = False
        for now in range(0, self.EndTrial):                
            loop = 0
            event_happen = False
            while self.eve[eve_id].time == now:
                now_event = self.eve[eve_id]
                eve_id += 1
                event_happen = True
            
                for neu in self.neu:
                    if loop!=0:
                        if neu == now_event.population:
                            pre_r = np.zeros((len(self.neu[neu][1].upstream), 1))
                            weight = np.zeros((len(self.neu[neu][1].upstream), len(self.neu[neu][1].upstream)))
                            up_idx = 0
                            for up1 in self.neu[neu][1].upstream:
                                pre_r[up_idx][0] = self.neu[up1][1].firing_rate[timestamp]
                                arr_col = 0
                                if up1 == neu:
                                    for up2 in self.neu[neu][1].upstream:
                                        if up1 in self.neu[up2][1].target:
                                            weight[up_idx][arr_col] = self.neu[up2][1].target[up1].weight
                                        else:
                                            weight[up_idx][arr_col] = 0
                                        arr_col += 1 
                                up_idx += 1
                            h_array = np.dot(weight, pre_r)
                            h = func(np.sum(h_array)+now_event.strength, active_func, a=a, b=b)
                            r_now = self.neu[neu][1].firing_rate[timestamp]
                            tau = self.neu[neu][1].Taum
                            r_next = r_now - r_now/tau + h/tau
                            self.neu[neu][1].firing_rate[timestamp+1] = (r_next)
                    else:
                        pre_r = np.zeros((len(self.neu[neu][1].upstream), 1))
                        weight = np.zeros((len(self.neu[neu][1].upstream), len(self.neu[neu][1].upstream)))
                        up_idx = 0
                        for up1 in self.neu[neu][1].upstream:
                            pre_r[up_idx][0] = self.neu[up1][1].firing_rate[timestamp]
                            arr_col = 0
                            if up1 == neu:
                                for up2 in self.neu[neu][1].upstream:
                                    if up1 in self.neu[up2][1].target:
                                        weight[up_idx][arr_col] = self.neu[up2][1].target[up1].weight
                                    else:
                                        weight[up_idx][arr_col] = 0
                                    arr_col += 1 
                            up_idx += 1
                        h_array = np.dot(weight, pre_r)
                        if event_happen == True and neu == now_event.population:
                            h = func(np.sum(h_array)+now_event.strength, active_func, a=a, b=b)
                        else:
                            h = func(np.sum(h_array), active_func, a=a, b=b)
                        r_now = self.neu[neu][1].firing_rate[timestamp]
                        tau = self.neu[neu][1].Taum
                        r_next = r_now - r_now/tau + h/tau
                        self.neu[neu][1].firing_rate[timestamp+1] = (r_next)
                loop += 1

            if event_happen!=True:
                for neu in self.neu:
                    pre_r = np.zeros((len(self.neu[neu][1].upstream), 1))
                    weight = np.zeros((len(self.neu[neu][1].upstream), len(self.neu[neu][1].upstream)))
                    up_idx = 0
                    for up1 in self.neu[neu][1].upstream:
                        pre_r[up_idx][0] = self.neu[up1][1].firing_rate[timestamp]
                        arr_col = 0
                        if up1 == neu:
                            for up2 in self.neu[neu][1].upstream:
                                if up1 in self.neu[up2][1].target:
                                    weight[up_idx][arr_col] = self.neu[up2][1].target[up1].weight
                                else:
                                    weight[up_idx][arr_col] = 0
                                arr_col += 1
                        up_idx+=1 
                    h_array = np.dot(weight, pre_r)
                    h = func(np.sum(h_array), active_func, a=a, b=b)
                    e = np.exp(-(now/self.neu[neu][1].Taum))
                    tau = self.neu[neu][1].Taum
                    r_now = self.neu[neu][1].firing_rate[timestamp]
                    r_next = r_now - r_now/tau + h/tau
                    self.neu[neu][1].firing_rate[timestamp+1] = (r_next)
            for grp in self.group.keys():
                if self.group[grp].WTA==True:
                    max_firerate=0
                    max_id=""
                    for neu in self.group[grp].member:
                        if self.neu[neu][1].firing_rate[timestamp+1]>=max_firerate:
                            max_firerate=self.neu[neu][1].firing_rate[timestamp+1]
                            max_id=self.neu[neu][1].name
                    for neu in self.group[grp].member:
                        if self.neu[neu][1].name!=max_id:
                            self.neu[neu][1].firing_rate[timestamp+1]=0
            timestamp+=1

    

                

    def ODE_simple(self, r0, t):
        def func_simple(self, parameters):
            r = parameters[0]
            dydt = [r, -r]
            return dydt

        sol = odeint(func_simple, r0, t)
        return sol

    # learning function !!!!!!!!!!!!!!!!!!!!!

class NeuralPopulation():
    
    def __init__(self, name, N=1, Taum=1, **kwargs):
        self.name = name
        self.N = N
        self.Taum = Taum
        self.upstream = {}
        self.target = {}
        self.firing_rate = [0]
        self.add_upstream(self.name)
        self.add_target(self.name)
        self.WTA = False
    
    def set_param(self, N, Taum):
        self.N = N
        self.Taum = Taum
    
    def add_target(self, name, weight=0):
        self.target[name] = Target(name, weight)
    
    def add_upstream(self, name, weight=0):
        self.upstream[name] = Upstream(name, weight)

    def output(self, fout):
        fout.write(
            f'----------NeuralPopulation: {self.name}-------------------\n'
            f'N={self.N}\n'
            f'Taum={self.Taum}\n\n\n'
        )

        # for upstream in self.upstream.values():
        #     if upstream.weight != 0:
        #         fout.write(
        #             f'UpstreamPopulation: {upstream.name}\n'
        #             f'weight={upstream.weight}\n'
        #             'EndUpstreamPopulation' + '\n\n\n'
        #         )

        for target in self.target.values():
            if target.weight != 0:
                fout.write(
                    f'TargetPopulation: {target.name}\n'
                    f'weight={target.weight}\n'
                    'EndTargetPopulation' + '\n\n\n'
                )

class Target():
    
    def __init__(self, name, weight=0):
        self.name = name
        self.weight = weight

    def set_param(self, weight):
        self.weight = weight

class Upstream():
    
    def __init__(self, name, weight=0):
        self.name = name
        self.weight = weight

    def set_param(self, weight):
        self.weight = weight


class Group():
    
    def __init__(self, name):
        self.name = name
        self.member = []
        self.WTA = False

    def add_member(self, member_name):
        self.member.append(member_name)

    def add_member_list(self, member_list):
        self.member.extend(member_list)
        
    def add_WTA(self):
        self.WTA=True

    def output(self, fout):
        fout.write(
                f'GroupName:{self.name}\n'
                f'WTA:{self.WTA}\n'
                'GroupMembers:')
        print(*self.member, sep=',', file=fout)
        fout.write('EndGroupMembers' + '\n\n')


class Event():
    
    def __init__(self, time, event_type, dt, *args):
        self.time = time
        self.type = event_type
        self.dt = dt
        self.timestamp = int(1/self.dt)
        if type(event_type) == type(1):
            self.end_time = event_type
            self.population = args[0]
            self.strength = args[1]
        elif event_type == 'EndTrial':    
            pass

    def output(self, fout):
        if type(self.type) == type(1) and self.time%self.timestamp == 0:
            fout.write(
                    f'EventTime {self.time/self.timestamp}\n'
                    'Type=Input\n'
                    f'Population: {self.population}\n'
                    f'Strength={self.strength}\n'
                    'EndEvent at ' + str(self.end_time/self.timestamp) + '\n\n'
                )
        elif self.type == 'EndTrial':
            fout.write(
                    f'EventTime {self.time/self.timestamp}\n'
                    'Type=EndTrial' + '\n'
                    'EndEvent' + '\n\n\n'
                )


            


    
