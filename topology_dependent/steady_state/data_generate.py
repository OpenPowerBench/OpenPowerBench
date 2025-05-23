import RL4Grid
from RL4Grid.utilize.form_action import form_action
import numpy as np
from pypower.api import *
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *
import time
from pypower.opf_args import opf_args2
from pypower.ppoption import ppoption
from pypower.isload import isload
from pypower.totcost import totcost
from pypower.fairmax import fairmax
from numpy import flatnonzero as find
import copy

network = 'SG126'
env = RL4Grid.make_gridsim(network=network)
# obs = env.reset()
# print(obs)
# num_gen = env.env.ppc['gen'].shape[0]
# adjust_gen_p = np.zeros(num_gen)
# adjust_gen_v = np.zeros(num_gen)
# obs, reward, done, info = env.step(form_action(adjust_gen_p, adjust_gen_v))
# import ipdb
# ipdb.set_trace()


def run_uopf(ppopt, ppc):
    open_hot = np.zeros(ppc['num_gen'] + 1)
    close_hot = np.zeros(ppc['num_gen'] + 1)
    t0 = time.time()  ## start timer

    ##-----  do combined unit commitment/optimal power flow  -----

    ## check for sum(Pmin) > total load, decommit as necessary
    on = find((ppc["gen"][:, GEN_STATUS] > 0) & ~isload(ppc["gen"]))  ## gens in service
    onld = find((ppc["gen"][:, GEN_STATUS] > 0) & isload(ppc["gen"]))  ## disp loads in serv
    load_capacity = sum(ppc["bus"][:, PD]) - sum(ppc["gen"][onld, PMIN])  ## total load capacity
    Pmin = ppc["gen"][on, PMIN]
    Pmax = ppc["gen"][on, PMAX]
    while sum(Pmin) > load_capacity:
        thermal_on = list(set(on) & set(ppc['thermal_ids']))
        if len(thermal_on) == 0:
            break
        ## shut down most expensive unit
        Pmin_thermal_on = ppc["gen"][thermal_on, PMIN]
        avgPmincost = totcost(ppc["gencost"][thermal_on, :], Pmin_thermal_on) / Pmin_thermal_on
        # _, i_to_close = fairmax(avgPmincost)  ## pick one with max avg cost at Pmin
        avgPmincost = list(avgPmincost)
        i_to_close = avgPmincost.index(max(avgPmincost))  ## pick one with max avg cost at Pmin
        i = thermal_on[i_to_close]  ## convert to generator index

        ## set generation to zero
        ppc["gen"][i, [PG, QG, GEN_STATUS, PMIN, PMAX]] = 0

        ## update minimum gen capacity
        on = find((ppc["gen"][:, GEN_STATUS] > 0) & ~isload(ppc["gen"]))  ## gens in service
        Pmin = ppc["gen"][on, PMIN]
        close_hot[i] = 1
        print('test, Shutting down generator %d.\n' % i)

    off = find((ppc["gen"][:, GEN_STATUS] == 0))
    while sum(Pmax) < load_capacity:
        thermal_off = list(set(off) & set(ppc['thermal_ids']))
        if len(thermal_off) == 0:
            break
        ## restart cheapest unit
        # Pmin_thermal_off = ppc["gen"][thermal_off, PMIN]
        # avgPmincost = totcost(ppc['gen'][thermal_off, :], Pmin_thermal_off) / Pmin_thermal_off
        avgPmincost = np.array(ppc['gencost'][:, STARTUP])[thermal_off]
        # _, i_to_restart = fairmax(-avgPmincost)
        avgPmincost = list(avgPmincost)
        i_to_restart = avgPmincost.index(min(avgPmincost))
        i = thermal_off[i_to_restart]

        # restart
        ppc['gen'][i, [PG, PMIN, PMAX]] = ppc['min_gen_p'][i]
        ppc['gen'][i, GEN_STATUS] = 1

        on = find((ppc["gen"][:, GEN_STATUS] > 0) & ~isload(ppc['gen']))
        off = find((ppc["gen"][:, GEN_STATUS] == 0))
        Pmax = ppc["gen"][on, PMAX]
        open_hot[i] = 1
        print('test, restarting generator %d.\n' % i)

    ## run initial opf
    results = opf(ppc, ppopt)

    ## compute elapsed time
    et = time.time() - t0

    return results, open_hot, close_hot


def rerun_opf(observation, ppc):
    bus_num = ppc['num_bus']
    gen_num = ppc['num_gen']
    load_num = ppc['num_load']
    renewable_num = len(ppc['renewable_ids'])

    gen_bus_lst = ppc['gen'][:, GEN_BUS].tolist()
    gen2busM = np.zeros((gen_num, bus_num))
    for i, bus in enumerate(gen_bus_lst):
        idx = ppc['bus'][:, BUS_I].tolist().index(bus)
        gen2busM[i, idx] = 1
    load_bus_lst = ppc['load_bus']
    ld2busM = np.zeros((load_num, bus_num))
    for i, bus in enumerate(load_bus_lst):
        ld2busM[i, int(bus)] = 1
    renewable2busM = np.zeros((renewable_num, bus_num))
    for i, gen in enumerate(ppc['renewable_ids']):
        renewable2busM[i] = gen2busM[gen]

    open_ids = np.where(observation.gen_status > 0)[0].tolist()
    close_ids = np.where(observation.gen_status == 0)[0].tolist()
    edge_ids = np.where(np.abs(observation.gen_p - np.asarray(ppc['min_gen_p'])) < 1e-3)[0].tolist()

    # BUS DATA
    load_p = np.asarray(observation.load_p)
    ppc['bus'][:, PD] = np.matmul(load_p, ld2busM)  # Pd
    load_q = np.asarray(observation.load_q)
    ppc['bus'][:, QD] = np.matmul(load_q, ld2busM)  # Qd
    Vm = np.asarray(observation.bus_v)
    ppc['bus'][:, VM] = Vm  # Vm
    # TODO: add bus angle information
    Va = np.asarray(observation.bus_ang)  # Va
    ppc['bus'][:, VA] = Va

    # GEN_DATA
    gen_p = np.asarray(observation.gen_p)
    ppc['gen'][:, PG] = gen_p  # Pg
    gen_q = np.asarray(observation.gen_q)
    ppc['gen'][:, QG] = gen_q  # Qg
    gen_v = np.asarray(observation.gen_v)
    ppc['gen'][:, VG] = gen_v  # Vg
    gen_status = np.asarray(observation.gen_status)
    ppc['gen'][:, GEN_STATUS] = gen_status  # status

    bal_gen_p_mid = (ppc['min_gen_p'][ppc['balanced_id']] + ppc['max_gen_p'][ppc['balanced_id']]) / 2
    redundancy = (ppc['max_gen_p'][ppc['balanced_id']] - ppc['min_gen_p'][ppc['balanced_id']]) / 2 * 0.7

    gen_p_upper = np.asarray(observation.gen_p) + observation.action_space['adjust_gen_p'].high
    gen_p_lower = np.asarray(observation.gen_p) + observation.action_space['adjust_gen_p'].low
    ppc['gen'][:, PMAX] = gen_p_upper
    ppc['gen'][ppc['balanced_id'], PMAX] = bal_gen_p_mid + redundancy
    ppc['gen'][close_ids, PMAX] = 0
    ratio = 0.98
    ppc['gen'][ppc['renewable_ids'], PMAX] = np.array(observation.nextstep_renewable_gen_p_max) * ratio
    ppc['gen'][:, PMIN] = gen_p_lower
    edge_n_thermal = list(set(edge_ids) & set(ppc['thermal_ids']))
    ppc['gen'][edge_n_thermal, PMIN] = ppc['gen'][edge_n_thermal, PMIN].clip(np.asarray(ppc['min_gen_p'])[edge_n_thermal], 1e6)
    ppc['gen'][ppc['balanced_id'], PMIN] = bal_gen_p_mid - redundancy
    ppc['gen'][close_ids, PMIN] = 0

    ppopt = ppoption(VERBOSE=0)
    # result = opf(ppc, ppopt)
    result, open_hot, close_hot = run_uopf(ppopt, ppc)
    result['gen'][:, 1] = result['gen'][:, 1].clip(ppc['gen'][:, 9], ppc['gen'][:, 8])
    new_gen_p = copy.deepcopy(result['gen'][:, 1])
    recover_ids = np.where(open_hot[:gen_num] > 0)[0].tolist()
    close_ids = np.where(close_hot[:gen_num] > 0)[0].tolist()
    return new_gen_p, recover_ids, close_ids, result


ppc_lst = []
target_dones = 0
for i in range(35132):
    obs = env.reset(start_sample_idx=i)
    action_high = obs.action_space['adjust_gen_p'].high
    action_low = obs.action_space['adjust_gen_p'].low
    ppc = copy.deepcopy(env.env.ppc)
    new_gen_p, _, _, result = rerun_opf(obs, ppc)
    best_a = new_gen_p - np.asarray(obs.gen_p)
    best_a[ppc['renewable_ids']] = best_a[ppc['renewable_ids']].clip(action_low[ppc['renewable_ids']], action_high[ppc['renewable_ids']])
    adjust_gen_v = np.zeros(ppc['num_gen'])
    adjust_gen_v = adjust_gen_v.clip(obs.action_space['adjust_gen_v'].low, obs.action_space['adjust_gen_v'].high)
    _, best_reward, target_done, info = env.step({'adjust_gen_p': best_a, 'adjust_gen_v': adjust_gen_v})
    if target_done:
        print(f'target done, {info}')
        target_dones += 1
        continue

    ppc_dict = {}
    ppc_dict['bus'] = env.env.ppc['bus']
    ppc_dict['branch'] = env.env.ppc['branch']
    ppc_dict['gen'] = env.env.ppc['gen']
    ppc_dict['target_gen_p'] = new_gen_p
    ppc_lst.append(ppc_dict)
    if i % 100 == 0:
        print("********************************************")
        print(f'{i}, network={network}, target_dones={target_dones}')
        print("--------------------------------------------")

import pickle
filehandler = open(f"ppc_lst_{network}.pkl", "wb")
pickle.dump(ppc_lst, filehandler)
filehandler = open(f'ppc_lst_{network}.pkl', "rb")
data = pickle.load(filehandler)
import ipdb
ipdb.set_trace()
