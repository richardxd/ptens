import os
import torch
from setuptools import setup
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import time
from os.path import splitext
from os.path import basename
from glob import glob


def main():

    # --- User settings ------------------------------------------------------------------------------------------
    # os.environ['CUDA_HOME']='/usr/local/cuda'
    #os.environ["CC"] = "clang"

    # compile_with_cuda = False  
    compile_with_cuda = True

    copy_warnings = False
    torch_convert_warnings = False

    # ------------------------------------------------------------------------------------------------------------
    
    #if 'CUDAHOME' in os.environ:
        #print("CUDA found at "+os.environ['CUDAHOME'])

    if compile_with_cuda:
        print("Installing with CUDA support")
    else:
        print("No CUDA found, installing without GPU support.")
        compile_with_cuda=False

    cwd = os.getcwd()
    cnine_folder = "/../../cnine/"

    _include_dirs = [cwd + cnine_folder + '/include',
		     cwd + cnine_folder + '/combinatorial',
		     cwd + cnine_folder + '/containers',
		     cwd + cnine_folder + '/math',
		     cwd + cnine_folder + '/hpc',
		     cwd + cnine_folder + '/utility',
		     cwd + cnine_folder + '/wrappers',
                     cwd + cnine_folder + '/include/cmaps',
                     cwd + cnine_folder + '/objects/scalar',
                     cwd + cnine_folder + '/objects/matrix',
                     cwd + cnine_folder + '/objects/tensor',
                     cwd + cnine_folder + '/objects/backendA',
                     cwd + cnine_folder + '/objects/backendB',
                     cwd + cnine_folder + '/objects/ntensor',
                     cwd + cnine_folder + '/objects/labeled',
                     cwd + cnine_folder + '/objects/labeled2',
                     cwd + cnine_folder + '/objects/tensor_views',
                     cwd + cnine_folder + '/objects/tensor_array',
                     cwd + cnine_folder + '/objects/tensor_array/cell_maps',
                     cwd + cnine_folder + '/objects/tensor_array/cell_ops',
                     cwd + cnine_folder + '/objects/ntensor',
                     cwd + cnine_folder + '/objects/ntensor/functions',
                     cwd + '/../include',
                     cwd + '/../tensors',
                     cwd + '/../layers',
                     cwd + '/../layers/backend'
                     ]


    _cxx_compile_args = ['-std=c++17',
                         '-Wno-sign-compare',
                         '-Wno-deprecated-declarations',
                         '-Wno-unused-variable',
                         # '-Wno-unused-but-set-variable',
                         '-Wno-reorder',
                         '-Wno-reorder-ctor',
                         '-Wno-overloaded-virtual',
                         '-D_WITH_ATEN',
                         '-DCNINE_RANGE_CHECKING',
                         '-DCNINE_SIZE_CHECKING',
                         '-DCNINE_DEVICE_CHECKING',
                         '-DCNINE_FUNCTION_TRACING',
                         '-DWITH_FAKE_GRAD'
                         ]

    _nvcc_compile_args = ['-D_WITH_CUDA',
                          '-D_WITH_CUBLAS',
                          '-D_DEF_CGCMEM',
                          '-DWITH_FAKE_GRAD',
                          '-DCNINE_FUNCTION_TRACING',
                          '--default-stream=per-thread'
                          # '-rdc=true'
                          ]

    if copy_warnings:
        _cxx_compile_args.extend([
            '-std=c++17',
            '-DCNINE_COPY_WARNINGS',
            '-DCNINE_ASSIGN_WARNINGS',
            '-DCNINE_MOVE_WARNINGS',
            '-DCNINE_MOVEASSIGN_WARNINGS',
            '-DPTENS_COPY_WARNINGS',
            '-DPTENS_ASSIGN_WARNINGS',
            '-DPTENS_MOVE_WARNINGS',
        ])

    if torch_convert_warnings:
        _cxx_compile_args.extend([
            '-DCNINE_ATEN_CONVERT_WARNINGS'
        ])

    if compile_with_cuda:
        _cxx_compile_args.extend(['-D_WITH_CUDA', '-D_WITH_CUBLAS'])

    _depends = ['setup.py',
                'bindings/*'
                ]


    # ---- Compilation commands ----------------------------------------------------------------------------------

    if compile_with_cuda:
        ext_modules = [CUDAExtension('ptens_base', [
            '../../cnine/include/Cnine_base.cu',
            '../../cnine/cuda/TensorView_accumulators.cu',
            '../../cnine/cuda/BasicCtensorProducts.cu',
            '../../cnine/cuda/RtensorUtils.cu',
            '../../cnine/cuda/RtensorPackUtils.cu',
            '../../cnine/cuda/gatherRows.cu',
            '../cuda/Ptensors0.cu',
            '../cuda/Ptensors1.cu',
            '../cuda/Ptensors2.cu',
            '../cuda/NodeLayer.cu',
            'bindings/ptens_py.cpp'
        ],
            include_dirs=_include_dirs,
            extra_compile_args={
            'nvcc': _nvcc_compile_args,
            'cxx': _cxx_compile_args},
            depends=_depends
        )]
    else:
        ext_modules = [CppExtension('ptens_base', ['bindings/ptens_py.cpp'],
                                    include_dirs=_include_dirs,
                                    # sources=sources,
                                    extra_compile_args={
                                        'cxx': _cxx_compile_args},
                                    depends=_depends
                                    )]

    setup(name='ptens',
          ext_modules=ext_modules,
          packages=find_packages('src'),
          package_dir={'': 'src'},
          py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
          include_package_data=True,
          zip_safe=False,
          cmdclass={'build_ext': BuildExtension})


    # ------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()


print("Compilation finished:", time.ctime(time.time()))
