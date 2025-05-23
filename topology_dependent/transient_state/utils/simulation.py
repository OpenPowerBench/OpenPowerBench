from utils.network import Network
from utils.fault import set_random_fault, apply_fault
from utils.results import save_results, save_fault_info
from config.paths import case_T, ctg_filepath
from esa import SAW

saw = SAW(case_T)

def solve_power_flow():
    result = saw.SolvePowerFlow()
    return result

def run_dynamic_simulation(gen_p_profiles, gen_q_profiles, ld_p_profiles, ld_q_profiles, gen_bus, ld_bus, n_start):
    sim_time = 2.0
    time_step = 0.5
    fault_start = 0.2
    fault_end = 0.4

    idx_sc = n_start
    for idx, gen_p in enumerate(gen_p_profiles):
        gen_q, ld_p, ld_q = gen_q_profiles[idx], ld_p_profiles[idx], ld_q_profiles[idx]
        network = Network(case_T)
        network.print_network_information()
        network.set_profile(gen_p, gen_q, ld_p, ld_q, gen_bus, ld_bus)

        for i in range(len(gen_p)):
            network.change_generation(gen_bus[i], "1", gen_p[i], gen_q[i])
        for i in range(len(ld_p)):
            network.change_load(ld_bus[i], "1", ld_p[i], ld_q[i])

        solve_power_flow()

        fault_type, location, R, X = set_random_fault()
        apply_fault(fault_type, location, fault_start, fault_end, sim_time, time_step, R, X, idx_sc)

        saw.ProcessAuxFile(ctg_filepath)

        ctg_name = f"My Transient Contingency{idx_sc}"
        cmd = f'TSSolve("{ctg_name}",[0,{sim_time},{time_step},YES])'
        saw.RunScriptCommand(cmd)

        bus_list = saw.GetParametersMultipleElement("Bus", ["BusNum"], "")
        obj_fields = [f'"Bus {bus} | TSVpu"' for bus in bus_list["BusNum"]]
        obj_fields += [f'"Bus {bus} | TSVangle"' for bus in bus_list["BusNum"]]
        obj_fields += [f'"Bus {bus} | frequency"' for bus in bus_list["BusNum"]]

        metadata, result_df = saw.TSGetContingencyResults(ctg_name, obj_fields)
        result_df.columns = result_df.columns.astype(str)

        result_df.rename(columns={
            str(i): f"Bus_{row['PrimaryKey']}_{row['VariableName']}"
            for i, row in metadata.iterrows()
        }, inplace=True)

        save_results(idx_sc, result_df)
        save_fault_info(idx_sc, fault_type, fault_start, fault_end, location)

        saw.RunScriptCommand('TSClearResultsFromRAM(ALL)')
        idx_sc += 1
