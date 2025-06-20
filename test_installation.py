import polyscope as ps
import mitsuba as mi

mi.set_variant("llvm_ad_rgb")

# Initialize Polyscope
ps.init()

# Start Polyscope
ps.show()
