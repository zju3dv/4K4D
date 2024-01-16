import torch

cudart = torch.cuda.cudart()


def register_memory(x: torch.Tensor):
    """
    The implementation of registering memory for fast HtoD copy is ultimately quite tricky due to the fact that
    some of the API for cuda and pytorch are not so robust on Windows (after an Windows 11 Insider update)
    Previously we used torch.cuda.cudart().cudaHostRegister and torch.cuda.cudart().cudaHostUnregister
    Everything is breaking a part, emitting strange errors like FIND CuDNN for convolution & invalid arguments & etc.

    RuntimeError: CUDA error: invalid argument # ???
    RuntimeError: FIND was unable to find an engine to execute this computation # disable benchmarking
    CudaError: part or all of the requested memory range is already mapped (cudaError.???) # make memory contiguous

    However the naive x.pin_memory() is not working either, the registered memory are too large
    Pinning 128MB results in 256MB shared memory usage on Windows, which is not explained anywhere...

    Thus we manually create a new tensor with pin_memory set to True, this should call cudaHostMalloc instead of registering
    This code path is more thoroughly tested by the PyTorch community since the async dataloading involves this
    And as experiments shown, this combines the benefits of the previous two implementations

    And no, it doesn't work.
    """
    # 1:
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g564c32d0e6032a9383494b6e63de7bd0
    # x = x.contiguous()
    # torch.cuda.check_error(cudart.cudaHostRegister(x.data_ptr(), x.numel() * x.element_size(), 0x08))
    # torch.cuda.check_error(cudart.cudaHostRegister(x.data_ptr(), x.numel() * x.element_size(), 0x0))
    # torch.cuda.synchronize()  # ???
    # torch.cuda.empty_cache()  # ???
    # return x

    # 2:
    # y = torch.empty_like(x, pin_memory=True)  # why extra memory usage here?
    # y = y.copy_(x)
    # return y

    # 3:
    x = x.pin_memory()
    return x

    # TODO: SWITCH THIS BASED ON PLATFORM
    # 4:
    # y = torch.empty_like(x, memory_format=torch.contiguous_format, pin_memory=False)
    # torch.cuda.check_error(cudart.cudaHostRegister(y.data_ptr(), y.numel() * y.element_size(), 0x08))
    # y = y.copy_(x)
    # return y

    # 5:
    # from cuda import cudart
    # CHECK_CUDART_ERROR(cudart.cudaHostRegister(x.data_ptr(), x.numel() * x.element_size(), cudart.cudaHostRegisterReadOnly))
    # return x


def unregister_memory(x: torch.Tensor):
    # torch.cuda.check_error(cudart.cudaHostUnregister(x.data_ptr()))
    # return x

    # from cuda import cudart
    # CHECK_CUDART_ERROR(cudart.cudaHostUnregister(x.data_ptr()))
    # return x
    y = torch.empty_like(x, pin_memory=False)
    y.copy_(x)
    return x


def FORMAT_CUDART_ERROR(err):
    from cuda import cudart
    return (
        f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
        f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
    )


def CHECK_CUDART_ERROR(args):
    from cuda import cudart

    if isinstance(args, tuple):
        assert len(args) >= 1
        err = args[0]
        if len(args) == 1:
            ret = None
        elif len(args) == 2:
            ret = args[1]
        else:
            ret = args[1:]
    else:
        err = args
        ret = None

    assert isinstance(err, cudart.cudaError_t), type(err)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(FORMAT_CUDART_ERROR(err))

    return ret
