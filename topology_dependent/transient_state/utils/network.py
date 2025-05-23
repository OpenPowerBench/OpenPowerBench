from esa import SAW

class Network:
    def __init__(self, case_path):
        self.saw = SAW(case_path)

    def print_network_information(self):
        self.buses = self.saw.GetParametersMultipleElement('Bus', ['BusNum'])
        self.generators = self.saw.GetParametersMultipleElement('Gen', ['BusNum', 'GenID'])
        self.loads = self.saw.GetParametersMultipleElement('Load', ['BusNum', 'LoadID'])
        self.branches = self.saw.GetParametersMultipleElement('Branch', ['BusNum', 'BusNum:1', 'LineCircuit'])

        load_data = self.saw.GetParametersMultipleElement('Load', ['BusNum', 'LoadID', 'LoadMW', 'LoadMVR'])
        self.load_p = load_data['LoadMW']
        self.load_q = load_data['LoadMVR']

    def set_profile(self, gen_p, gen_q, ld_p, ld_q, gen_bus, ld_bus):
        self.gen_p, self.gen_q = gen_p, gen_q
        self.load_p, self.load_q = ld_p, ld_q
        self.gen_prof_bus, self.ld_prof_bus = gen_bus, ld_bus

    def change_generation(self, bus_num, gen_id, p_gen, q_gen):
        self.saw.ChangeParameters('Gen', ['BusNum', 'GenID', 'GenMW', 'GenMVR'], [bus_num, gen_id, p_gen, q_gen])

    def change_load(self, bus_num, load_id, p_load, q_load):
        self.saw.ChangeParameters('Load', ['BusNum', 'LoadID', 'LoadMW', 'LoadMVR'], [bus_num, load_id, p_load, q_load])
