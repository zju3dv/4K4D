import asyncio
import websockets


async def websocket_client():
    uri = "ws://localhost:1024"  # 根据实际地址和端口调整
    async with websockets.connect(uri) as websocket:
        # 等待并打印接收到的消息
        response = await websocket.recv()
        print(f"Received from server: {response}")

# 运行客户端
asyncio.get_event_loop().run_until_complete(websocket_client())
