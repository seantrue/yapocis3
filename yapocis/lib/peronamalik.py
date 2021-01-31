# http://mail.scipy.org/pipermail/scipy-user/2009-February/020125.html
# Anisotropic Diffusion, as per Perona and Malik's paper (see section V).
import yapocis.rpc.programs
from yapocis.rpc import engines
yapocis.rpc.programs.import_program("peronamalik", filterImage="_peronamalik", filterImage_res="_peronamalik_res")
