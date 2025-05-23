from config.paths import ctg_filepath

def write_aux(data_list, simu_time, time_step, idx):
    name = f"My Transient Contingency{idx}"
    filepath_aux = ctg_filepath
    TS_header = "TSContingency (Name,Category,TimeStart,TimeEnd,TimeStepUseCycles,TimeStep,...)"
    TS_dataheader = "TSContingencyElement (Contingency,Time,Object,Action,...)"
    savebus_header = "SetData(Bus, [TSSaveAll,TSSaveVpu,...], [NO, YES,...], All);"

    with open(filepath_aux, 'w') as f:
        f.write(f"{TS_header}\n{{\n")
        f.write(data_list[0] + "\n" + data_list[1] + "\n}")
        f.write(f"\n{TS_dataheader}\n{{\n")
        for val in data_list.values():
            f.write(val + "\n")
        f.write("}\nSCRIPT\n{\n" + savebus_header + "\n}\n")

    return filepath_aux
