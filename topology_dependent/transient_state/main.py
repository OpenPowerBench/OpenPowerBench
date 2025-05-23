from utils.profiles import load_profiles
from utils.simulation import run_dynamic_simulation
from config.paths import (
    gen_p_path_train, gen_q_path_train,
    ld_p_path_train, ld_q_path_train
)

def main():
    n_start = 0
    n_end = 35130
    specified_rows = list(range(n_start, n_end + 1))

    gen_p_profiles, gen_bus = load_profiles(gen_p_path_train, specified_rows)
    gen_q_profiles, _ = load_profiles(gen_q_path_train, specified_rows)
    ld_p_profiles, ld_bus = load_profiles(ld_p_path_train, specified_rows)
    ld_q_profiles, _ = load_profiles(ld_q_path_train, specified_rows)

    run_dynamic_simulation(gen_p_profiles, gen_q_profiles, ld_p_profiles, ld_q_profiles, gen_bus, ld_bus, n_start)

    print("Simulation completed.")

if __name__ == "__main__":
    main()
