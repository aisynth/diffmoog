reverb folder contains room impulse responses generated in Audacity free DAW software.

Multiple room impulse responses are created by generating a delta function and applying reverb effect in the software
with different room sizes,

Currently, the ai_synth module does not support the reverb effect, and may be not used at all.
However, the files are saved in case a future use is needed

rir_for_reverb and rir_for_reverb_no_amp are dictionaries containing tensors with the room impulse responses.
they are generated at room_impulse_response.py