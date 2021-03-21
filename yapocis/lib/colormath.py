#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.lib.bound_kernels import BoundKernels

transform_names = ['adobe1998_d65.rgb2xyz', 'adobe1998_d65.xyz2rgb', 'apple_d65.rgb2xyz', 'apple_d65.xyz2rgb',
                   'best_d50.rgb2xyz', 'best_d50.xyz2rgb', 'beta_d50.rgb2xyz', 'beta_d50.xyz2rgb', 'bruce_d65.rgb2xyz',
                   'bruce_d65.xyz2rgb', 'cie_e.rgb2xyz', 'cie_e.xyz2rgb', 'colormatch_d50.rgb2xyz',
                   'colormatch_d50.xyz2rgb', 'donrgb3_d50.rgb2xyz', 'donrgb3_d50.xyz2rgb', 'eci_d50.rgb2xyz',
                   'eci_d50.xyz2rgb', 'ektaspaceps5_d50.rgb2xyz', 'ektaspaceps5_d50.xyz2rgb', 'ntsc_c.rgb2xyz',
                   'ntsc_c.xyz2rgb', 'palsecam_d65.rgb2xyz', 'palsecam_d65.xyz2rgb', 'prophoto_d50.rgb2xyz',
                   'prophoto_d50.xyz2rgb', 'smpte-c_d65.rgb2xyz', 'smpte-c_d65.xyz2rgb', 'srgb_d65.rgb2xyz',
                   'srgb_d65.xyz2rgb', 'wide_d50.rgb2xyz', 'wide_d50.xyz2rgb', 'adobe_d50.rgb2xyz', 'adobe_d50.xyz2rgb',
                   'applergb_d50.rgb2xyz', 'applergb_d50.xyz2rgb', 'bruce_d50.rgb2xyz', 'bruce_d50.xyz2rgb',
                   'cie_d50.rgb2xyz', 'cie_d50.xyz2rgb', 'ntsc_d50.rgb2xyz', 'ntsc_d50.xyz2rgb', 'palsecam_d50.rgb2xyz',
                   'palsecam_d50.xyz2rgb', 'smptec_d50.rgb2xyz', 'smptec_d50.xyz2rgb', 'srgb_d50.rgb2xyz',
                   'srgb_d50.xyz2rgb', "unity"]
names = {name: i for (i, name) in enumerate(transform_names)}


class Colormath(BoundKernels):
    INTERFACE = "colormath"

    def transform_named(self, name, array):
        return self.transform(array, names[name], global_size=array.size // 3)
    def xyz2xyY(self, input):
        return self.kmgr.xyz2xyY(input,global_size=input.size//3)
    def xyY2xyz(self, input):
        return self.kmgr.xyY2xyz(input,global_size=input.size//3)