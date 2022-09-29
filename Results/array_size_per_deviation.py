import numpy as np
from matplotlib import pyplot as plt

array_size = [0.40000000000000013, 0.400197794112005, 0.4005142660005831, 0.4007244331365938, 0.4022208501546524,
              0.4030970981384796, 0.40652347678125833, 0.40975900863680337, 0.41463572908364843, 0.423665940676059,
              0.4321736230495311]

original = 0.4
array_growth = [i/original for i in array_size]
# array_fac = []
# for k, val in enumerate(array_growth):
#     if k != 0:
#         array_fac.append(val++1/k)

fac = []
for idx in range(len(array_growth)-1):
    fac.append(array_growth[idx]/array_growth[idx-1])

print(fac)

plt.plot(array_growth)
plt.show()