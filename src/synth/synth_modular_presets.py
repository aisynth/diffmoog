from synth_architecture import SynthModularCell

BASIC_FLOW = [
    SynthModularCell(index=(0, 0), operation='osc', default_connection=True),
    SynthModularCell(index=(0, 1), operation='fm', default_connection=True),
    SynthModularCell(index=(1, 0), operation='osc', default_connection=True),
    SynthModularCell(index=(1, 1), operation='fm', input_list=[[1, 0]], output=[0, 2]),
    SynthModularCell(index=(1, 2), operation=None, input_list=None),
    SynthModularCell(index=(0, 2),
                     operation='mix',
                     input_list=[[0, 1], [1, 1]]),
    SynthModularCell(index=(0, 3), operation='filter', default_connection=True),
    SynthModularCell(index=(0, 4), operation='env_adsr', default_connection=True),
]