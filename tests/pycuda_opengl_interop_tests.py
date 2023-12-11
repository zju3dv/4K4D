from easyvolcap.utils.test_utils import my_tests

def test_context_creation():
    import pycuda.driver as drv
    drv.init()
    dev = drv.Device(0)
    print('Trying to create context....')
    ctx = dev.make_context()
    print(f'Context created on device: {dev.name()}')
    ctx.pop()
    del ctx
    print('Context removed.\nEnd of test')

    print('First test passed')
    import pycuda.gl
    ctx = pycuda.gl.make_context(dev)
    print(f'Context created on device: {dev.name()}')
    ctx.pop()
    del ctx
    print('Context removed.\nEnd of test')

if __name__ == '__main__':
    my_tests(globals())