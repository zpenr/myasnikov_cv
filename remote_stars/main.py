import socket
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label

def recvall(sock, nbytes):
    data = bytearray()
    while len(data) < nbytes:
        package = sock.recv(nbytes - len(data))
        if not package:
            return None
        data.extend(package)
    return data

def get_brigthest_pixel(image: np.ndarray, labeled: np.ndarray, label: int):
    pixel = np.unravel_index(np.argmax(image * (labeled == label)), image.shape)
    return pixel

def dist(center1: list[int, int], center2: list[int,int]) -> int:
    return ((center1[1] - center2[1])**2 + (center1[0] - center2[0])**2)**0.5

def solve(image: np.ndarray) -> float:
    labeled = label(image > 0)
    center1 = get_brigthest_pixel(image, labeled, 1)
    center2 = get_brigthest_pixel(image, labeled, 2)
    return dist(center1,center2)

host = "84.237.21.36"
port = 5152

def main() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        sock.send(b"124ras1")
        print(sock.recv(10))

        beat = b"nope"
        while beat != b"yep":
            sock.send(b"get")
            bts: None | bytearray = recvall(sock, 40002)

            image = np.frombuffer(bts[2:40002], dtype="uint8").reshape(bts[0], bts[1])
            answer: float = round(solve(image),1)

            print("my ans", answer)
            sock.send(str(answer).encode())
            print(sock.recv(10))
            sock.send(b"beat")
            beat: bytes = sock.recv(10)

main()