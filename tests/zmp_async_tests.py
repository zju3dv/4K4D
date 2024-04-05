import zmq
import time
import threading
from queue import Queue

# 线程安全的队列，用于发布者和订阅者之间的数据共享
data_queue = Queue()


def publisher(context, data_queue):
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5556")
    while True:
        # 检查是否有新的数据要发布
        if not data_queue.empty():
            data = data_queue.get()
            socket.send_pyobj(data)
        time.sleep(1)  # 模拟工作负载


def subscriber(context, data_queue):
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5556")
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    while True:
        data = socket.recv_pyobj()
        # 处理接收到的数据...
        print(f"Received data: {data}")
        # 可能基于接收到的数据更新队列或执行其他操作
        # 注意：直接修改共享数据应当小心使用锁来避免竞态条件
        data_queue.put(data)


if __name__ == "__main__":
    context = zmq.Context()
    # 创建并启动发布者线程
    pub_thread = threading.Thread(target=publisher, args=(context, data_queue))
    pub_thread.start()

    # 创建并启动订阅者线程
    sub_thread = threading.Thread(target=subscriber, args=(context, data_queue))
    sub_thread.start()

    data_queue.put('data')


    pub_thread.join()
    sub_thread.join()
