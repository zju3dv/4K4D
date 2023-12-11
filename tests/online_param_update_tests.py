import torch
from easyvolcap.utils.test_utils import my_tests

REQUEST_GRAPH_DELETION = False  # if False, the error always occurs, if True, only sometimes
TEST_REPEAT_COUNT = 0  # 200


def test_parameter_online_upate():
    def run_loop():
        x = torch.ones((32, 64, 64), requires_grad=True)
        out = None

        while x.shape[0] > 0:
            yield x.shape  # for printing

            if REQUEST_GRAPH_DELETION:
                del out  # time.sleep(0.0001) helps, but shoudn't be relied on

            out = x.sum()
            assert out.grad_fn(out).shape == x.shape  # correct gradient shape
            out.backward()  # incorrect gradient shape because some metadata is not updated

            with torch.no_grad():  # change the shape
                N = x.shape[0] // 2

                grad = x.grad
                # https://discuss.pytorch.org/t/tensor-set-seems-not-to-correctly-update-metadata-if-the-shape-changes/67361
                # x.data = x.data[:N] will error out
                x.set_(x.data[:N])
                x.grad = grad[:N]

    if TEST_REPEAT_COUNT > 0:  # repeats
        error_count, succ_count = 0, 0
        for i in range(TEST_REPEAT_COUNT):
            shapes = []
            for shape in run_loop():
                shapes.append(shape[0])
            succ_count += 1


if __name__ == '__main__':
    my_tests(globals())
