import random
import matplotlib.pyplot as plt
import numpy as np
import time


def sashapetyaslava_distribution_function(i: int, d: int, max_idx: int) -> float:
    """
    p(i) == 0
    sum([p(i) for i in range(0, max_idx)]) == 1

    p(i) ^   .....
         |   .   .
         |....   .
         |........
         ------------> i
             ^   ^
             |   |
     max_idx-d   max_idx
    """
    h1 = 0.5 / (max_idx - d + 1)
    h2 = 0.5 / d
    if not (0 <= i <= max_idx):
        raise ValueError("i is out of distribution range")
    if i >= max_idx - d:
        return h2
    else:
        return h1


def sample_from_sashapetyaslava(d: int, max_idx: int) -> int:
    ws = range(0, max_idx + 1)
    probas = [
        sashapetyaslava_distribution_function(i, d=d, max_idx=max_idx) for i in ws
    ]
    return random.choices(ws, probas)[0]


# lst = []
# start = time.time()
# for i in range(10000):
#     lst.append(sample_from_sashapetyaslava(d=500, max_idx=1999))
# print(time.time() - start)


def get_sps_probs_v(d, max_idx):
    buf = np.ones(max_idx + 1)
    h1 = 0.5 / (max_idx - d + 1)
    h2 = 0.5 / d
    h1_v = buf * h1
    h2_v = buf * h2
    cond = np.full(max_idx + 1, (np.arange(max_idx + 1) >= max_idx - d + 1))
    return np.where(cond, h2_v, h1_v)


def get_sps_probs(d, max_idx):
    buf = np.ones(max_idx + 1)
    h1 = 0.5 / (max_idx - d + 1)
    h2 = 0.5 / d
    buf[: max_idx - d + 1] = h1
    buf[max_idx - d + 1 :] = h2
    return buf


probas = get_sps_probs(d=500, max_idx=1999)
print(probas[0])
print(f"{sum(probas)}")
probas_v = get_sps_probs_v(d=500, max_idx=1999)
print(f"{sum(probas_v)}")


def sample_from_sashapetyaslava(d: int, max_idx: int) -> int:
    probas = get_sps_probs(d=d, max_idx=max_idx)
    return np.random.choice(max_idx + 1, p=probas, replace=True, size=1)


lst = []
start = time.time()
for i in range(10000):
    lst.append(sample_from_sashapetyaslava(d=500, max_idx=1999))
print(time.time() - start)


def sample_from_sashapetyaslava(d: int, max_idx: int) -> int:
    probas = get_sps_probs_v(d=d, max_idx=max_idx)
    return np.random.choice(max_idx + 1, p=probas, replace=True, size=1)


lst = []
start = time.time()
for i in range(10000):
    lst.append(sample_from_sashapetyaslava(d=500, max_idx=1999))
print(time.time() - start)


def sample_from_sashapetyaslava(d: int, max_idx: int) -> int:
    probas = get_sps_probs(d=d, max_idx=max_idx)
    return np.random.choice(max_idx + 1, p=probas, replace=True, size=1)


lst = []
start = time.time()
for i in range(10000):
    lst.append(sample_from_sashapetyaslava(d=500, max_idx=10000 - 1))
print(time.time() - start)


def sample_from_sashapetyaslava(d: int, max_idx: int) -> int:
    probas = get_sps_probs_v(d=d, max_idx=max_idx)
    return np.random.choice(max_idx + 1, p=probas, replace=True, size=1)


lst = []
start = time.time()
for i in range(10000):
    lst.append(sample_from_sashapetyaslava(d=500, max_idx=10000 - 1))
print(time.time() - start)
