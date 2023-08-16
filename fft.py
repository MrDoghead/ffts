import random
import math
from math import pi, sin, cos
import copy
import numpy as np

J = complex(0,1)

def fft_radix2_inplace(a, is_forward=True):
    n = len(a)
    nbit = math.ceil(math.log2(n))
    rev = list(range(n))
    y = a 
    # bit reverse, e.g.,
    # p = [0(000), 1(001),   2(010), 3(011),   4(100), 5(101),   6(110), 7(111)]
    #   ->[0(000), 2(010),   4(100), 6(110)], [1(001), 3(011),   5(101), 7(111)]
    # q = [0(000), 4(100)], [2(010), 6(110)], [1(001), 5(101)], [3(011), 7(111)]
    # find p and q are just reversed
    for i in range(n):
        rev[i] = rev[i >> 1] >> 1 | (rev[i] & 1) << nbit - 1
        if i < rev[i]:
            y[i], y[rev[i]] = y[rev[i]], y[i]

    for step in map(lambda x: 1 << x, range(1, nbit + 1)): # determine the length of blocks for each level
        phi = 2 * pi / step if is_forward else -2 * pi / step
        w = complex(cos(phi), sin(phi))
        for i in range(0, n, step): # loop over blocks
            for j in range(step//2): # loop over half of the block
                ye = y[i + j]
                yo = y[i + j + step//2]
                y[i + j]           = ye + w**j * yo 
                y[i + j + step//2] = ye - w**j * yo
    if not is_forward:
        y = [v / n for v in y]
    return y

def fft_radix4_inplace(a, is_forward=True):
    n = len(a)
    nbit = math.ceil(math.log2(n))
    rev = list(range(n))
    y = a 
    # bit reverse, e.g.,
    # p = [0(0000),1(0001),2(0010), 3(0011),  4(0100),5(0101),6(0110), 7(0111),  8(1000),9(1001),10(1010),11(1011), 12(1100),13(1101),14(1110),15(1111)]
    # q = [0(0000),4(0100),8(1000),12(1100)],[1(0001),5(0101),9(1001),13(1101)],[2(0010),6(0110),10(1010),14(1110)],[3(0011), 7(0111),11(1011),15(1111)]
    # find p and q are just reversed in 2-bits
    for i in range(n):
        rev[i] = rev[i >> 2] >> 2 | (rev[i] & 3) << nbit - 2 
        if i < rev[i]:
            y[i], y[rev[i]] = y[rev[i]], y[i]

    for step in map(lambda x: 1 << x, range(2, nbit + 1, 2)):
        phi = 2 * pi / step if is_forward else -2 * pi / step
        w = complex(cos(phi), sin(phi))
        for i in range(0, n, step):
            for j in range(step//4):
                if is_forward:
                    y0 = y[i+j] + 1 * w**j * y[i+j+step*1//4] + 1 * w**(2*j) * y[i+j+step*2//4] + 1 * w**(3*j) * y[i+j+step*3//4]
                    y1 = y[i+j] - J * w**j * y[i+j+step*1//4] - 1 * w**(2*j) * y[i+j+step*2//4] + J * w**(3*j) * y[i+j+step*3//4]
                    y2 = y[i+j] - 1 * w**j * y[i+j+step*1//4] + 1 * w**(2*j) * y[i+j+step*2//4] - 1 * w**(3*j) * y[i+j+step*3//4]
                    y3 = y[i+j] + J * w**j * y[i+j+step*1//4] - 1 * w**(2*j) * y[i+j+step*2//4] - J * w**(3*j) * y[i+j+step*3//4]
                else:
                    y0 = y[i+j] + 1 * w**j * y[i+j+step*1//4] + 1 * w**(2*j) * y[i+j+step*2//4] + 1 * w**(3*j) * y[i+j+step*3//4]
                    y1 = y[i+j] + J * w**j * y[i+j+step*1//4] - 1 * w**(2*j) * y[i+j+step*2//4] - J * w**(3*j) * y[i+j+step*3//4]
                    y2 = y[i+j] - 1 * w**j * y[i+j+step*1//4] + 1 * w**(2*j) * y[i+j+step*2//4] - 1 * w**(3*j) * y[i+j+step*3//4]
                    y3 = y[i+j] - J * w**j * y[i+j+step*1//4] - 1 * w**(2*j) * y[i+j+step*2//4] + J * w**(3*j) * y[i+j+step*3//4]
                y[i+j]           = y0
                y[i+j+step*1//4] = y1
                y[i+j+step*2//4] = y2
                y[i+j+step*3//4] = y3
    if not is_forward:
        y = [v / n for v in y]
    return y

def FFT_Radix2(P):
    n = len(P)
    if n == 1:
        return P
    phi = -2 * pi / n
    w = complex(cos(phi), sin(phi))
    Pe, Po = P[::2], P[1::2]
    ye, yo = FFT_Radix2(Pe), FFT_Radix2(Po)
    y = [0] * n 
    for j in range(n//2):
        y[j] = ye[j] + w**j * yo[j]
        y[j + n//2] = ye[j] - w**j * yo[j]
    return y

def IFFT_Radix2(P):
    """note: divide the final result by n"""
    n = len(P)
    if n == 1:
        return P
    phi = 2 * pi / n
    w = complex(cos(phi), sin(phi))
    Pe, Po = P[::2], P[1::2]
    ye, yo = IFFT_Radix2(Pe), IFFT_Radix2(Po)
    y = [0] * n 
    for j in range(n//2):
        y[j] = ye[j] + w**j * yo[j]
        y[j + n//2] = ye[j] - w**j * yo[j]
    return y

def FFT_Radix4(P):
    n = len(P)
    if n == 1:
        return P
    phi = -2 * pi / n
    w = complex(cos(phi), sin(phi))
    P0, P1, P2, P3 = P[::4], P[1::4], P[2::4], P[3::4]
    y0, y1, y2, y3 = FFT_Radix4(P0), FFT_Radix4(P1), FFT_Radix4(P2), FFT_Radix4(P3)
    y = [0] * n
    for j in range(n//4):
        y[j]          = y0[j] + 1 * w**j * y1[j] + 1 * w**(2*j) * y2[j] + 1 * w**(3*j) * y3[j]
        y[j + n*1//4] = y0[j] - J * w**j * y1[j] - 1 * w**(2*j) * y2[j] + J * w**(3*j) * y3[j]
        y[j + n*2//4] = y0[j] - 1 * w**j * y1[j] + 1 * w**(2*j) * y2[j] - 1 * w**(3*j) * y3[j]
        y[j + n*3//4] = y0[j] + J * w**j * y1[j] - 1 * w**(2*j) * y2[j] - J * w**(3*j) * y3[j]
    return y

# do not forget to divide the IFFT result by N
def IFFT_Radix4(P):
    n = len(P)
    if n == 1:
        return P
    phi = 2 * pi / n
    w = complex(cos(phi), sin(phi))
    P0, P1, P2, P3 = P[::4], P[1::4], P[2::4], P[3::4]
    y0, y1, y2, y3 = IFFT_Radix4(P0), IFFT_Radix4(P1), IFFT_Radix4(P2), IFFT_Radix4(P3)
    y = [0] * n
    for j in range(n//4):
        y[j]          = y0[j] + 1 * w**j * y1[j] + 1 * w**(2*j) * y2[j] + 1 * w**(3*j) * y3[j]
        y[j + n*1//4] = y0[j] + J * w**j * y1[j] - 1 * w**(2*j) * y2[j] - J * w**(3*j) * y3[j]
        y[j + n*2//4] = y0[j] - 1 * w**j * y1[j] + 1 * w**(2*j) * y2[j] - 1 * w**(3*j) * y3[j]
        y[j + n*3//4] = y0[j] - J * w**j * y1[j] - 1 * w**(2*j) * y2[j] + J * w**(3*j) * y3[j]
    return y

def test1():
    print("##### test fft and ifft #####")
    N = 16
    p = [random.randint(-8, 7) for i in range(N)]
    print("p: ", p)
    print("np fft:", np.fft.fft(p))
    fft_out = FFT_Radix2(p)
    print("fft: ", fft_out)
    ifft_out = IFFT_Radix2(fft_out)
    ifft_out = [round(v.real / N) for v in ifft_out]
    print("ifft: ", ifft_out)
    err_rate = sum([1 if p[i]!=ifft_out[i] else 0 for i in range(N)]) / N
    print(f"Error rate: {err_rate}")

def test1_2():
    print("##### test fft and ifft #####")
    N = 16
    p = [random.randint(-8, 7) for i in range(N)]
    print("p: ", p)
    fft_out = fft_radix2_inplace(copy.deepcopy(p), True)
    print("fft: ", fft_out)
    ifft_out = fft_radix2_inplace(copy.deepcopy(fft_out), False)
    ifft_out = [round(v.real) for v in ifft_out]
    print("ifft: ", ifft_out)
    err_rate = sum([1 if p[i]!=ifft_out[i] else 0 for i in range(N)]) / N
    print(f"Error rate: {err_rate}")

def test2():
    print("##### test radix-4 fft and ifft #####")
    N = 64
    p = [random.randint(-8, 7) for i in range(N)]
    # p = [random.randint(0, 1) for i in range(N)]
    print("p: ", p)
    fft_out = FFT_Radix4(p)
    print("fft: ", fft_out)
    ifft_out = IFFT_Radix4(fft_out)
    ifft_out = [round(v.real/N) for v in ifft_out]
    print("ifft: ", ifft_out)
    err_rate = sum([1 if p[i]!=ifft_out[i] else 0 for i in range(N)]) / N
    print(f"Error rate: {err_rate}")

def test2_2():
    print("##### test radix-4 fft and ifft #####")
    N = 16
    p = [random.randint(-8, 7) for i in range(N)]
    # p = [random.randint(0, 1) for i in range(N)]
    print("p: ", p)
    fft_out = fft_radix4_inplace(copy.deepcopy(p), True)
    print("fft: ", fft_out)
    ifft_out = fft_radix4_inplace(copy.deepcopy(fft_out), False)
    ifft_out = [round(v.real) for v in ifft_out]
    print("ifft: ", ifft_out)
    err_rate = sum([1 if p[i]!=ifft_out[i] else 0 for i in range(N)]) / N
    print(f"Error rate: {err_rate}")

def test3():
    print("##### compute 12345*67890=838102050 #####")
    N = 16
    p1 = [5, 4, 3, 2, 1] + [0] * (N-5)
    p2 = [0, 9, 8, 7, 6] + [0] * (N-5)
    print(f"p1:\n{p1}\np2:\n{p2}")
    y1 = FFT_Radix2(p1)
    y2 = FFT_Radix2(p2)
    print(f"FFT(p1):\n{y1}")
    print(f"FFT(p2):\n{y2}")
    y3 = [y1[i]*y2[i] for i in range(N)]
    print(f"FFT(p1) · FFT(p2):\n{y3}")
    p3 = IFFT_Radix2(y3)
    p3 = [v/N for v in p3]
    print(f"p3:\n{p3}")
    result = 0
    for i in range(N):
        result += round(p3[i].real) * 10**i
    print(f"the product is {result}")

if __name__ == "__main__":
    test1()
    # test1_2()
    # test2()
    # test2_2()
    # test3()

