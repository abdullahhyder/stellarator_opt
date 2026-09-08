from main import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

# ARIES-CS (unscaled)
vmec_file = "./wout_ARIES-CS.nc"
title = "ARIES-CS"
stellgap_output = AlfvenSpecData.from_dir("./")
file_name = "ARIES-CS.pdf"

# Plot
fig_plt = plot_continuum_matplotlib(vmec_file=vmec_file, 
                modes=stellgap_output.get_modes(), ylims=[0,3], 
                insert_inset=False, normalized=True)

fig_plt[0].savefig(file_name, bbox_inches='tight')

fig_plt[0].show()