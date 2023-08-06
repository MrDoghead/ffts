import random
from math import pi, sin, cos
import numpy as np

K = 16 # exponent of 2^K
N = 4**(K//2) # polynomial
input_bits = 1
output_bits = input_bits + K
J = complex(0,1)

def to_intN(x, n):
    y = round(x)
    if y < -2**(n-1):
        return -2**(n-1)
    elif y > 2**(n-1)-1:
        return 2**(n-1)-1
    else:
        return y

def complex_round(v, n):
    rounded_real = to_intN(v.real, n)
    rounded_imag = to_intN(v.imag, n)
    return complex(rounded_real, rounded_imag)

def OFT_Radix4(P):
    n = len(P)
    if n == 1:
        return P
    phi = -2 * pi / n
    w = complex(cos(phi), sin(phi))
    P0, P1, P2, P3 = P[::4], P[1::4], P[2::4], P[3::4]
    y0, y1, y2, y3 = OFT_Radix4(P0), OFT_Radix4(P1), OFT_Radix4(P2), OFT_Radix4(P3)
    y = [0] * n
    for j in range(n//4):
        # print(f"coeff: 1 {w**j} {w**(2*j)} {w**(3*j)}")
        y[j]          = y0[j] + 1 * w**j * y1[j] + 1 * w**(2*j) * y2[j] + 1 * w**(3*j) * y3[j]
        y[j + n*1//4] = y0[j] - J * w**j * y1[j] - 1 * w**(2*j) * y2[j] + J * w**(3*j) * y3[j]
        y[j + n*2//4] = y0[j] - 1 * w**j * y1[j] + 1 * w**(2*j) * y2[j] - 1 * w**(3*j) * y3[j]
        y[j + n*3//4] = y0[j] + J * w**j * y1[j] - 1 * w**(2*j) * y2[j] - J * w**(3*j) * y3[j]
    if n >= 4:
        # print(f"[OFT] P-{n} outputs before rounding: \n {y}")
        y = [complex_round(v, output_bits) for v in y]
        # print(f"[OFT] P-{n} outputs after rounding:\n {y}")
    return y

def IOFT_Radix4(P):
    """note: divide the final result by n"""
    n = len(P)
    if n == 1:
        return P
    phi = 2 * pi / n
    w = complex(cos(phi), sin(phi))
    P0, P1, P2, P3 = P[::4], P[1::4], P[2::4], P[3::4]
    y0, y1, y2, y3 = IOFT_Radix4(P0), IOFT_Radix4(P1), IOFT_Radix4(P2), IOFT_Radix4(P3)
    y = [0] * n
    for j in range(n//4):
        y[j]          = y0[j] + 1 * w**j * y1[j] + 1 * w**(2*j) * y2[j] + 1 * w**(3*j) * y3[j]
        y[j + n*1//4] = y0[j] + J * w**j * y1[j] - 1 * w**(2*j) * y2[j] - J * w**(3*j) * y3[j]
        y[j + n*2//4] = y0[j] - 1 * w**j * y1[j] + 1 * w**(2*j) * y2[j] - 1 * w**(3*j) * y3[j]
        y[j + n*3//4] = y0[j] - J * w**j * y1[j] - 1 * w**(2*j) * y2[j] + J * w**(3*j) * y3[j]
    if n >= 4:
        y = [complex_round(v, output_bits) for v in y]
    return y

def OFT_Radix2(P):
    n = len(P)
    if n == 1:
        print(f"n={n}  y={P}")
        return P
    phi = 2 * pi / n
    w = complex(cos(phi), sin(phi))
    Pe, Po = P[::2], P[1::2]
    ye, yo = OFT_Radix2(Pe), OFT_Radix2(Po)
    y = [0] * n 
    for j in range(n//2):
        y[j] = ye[j] + w**j * yo[j]
        y[j + n//2] = ye[j] - w**j * yo[j]
    # if n >= 2:
        # y = [complex_round(v, output_bits) for v in y]
    # print(f"n={n}  y={y}")
    return y

def IOFT_Radix2(P):
    """note: divide the final result by n"""
    n = len(P)
    if n == 1:
        print(f"n={n}  y={P}")
        return P
    phi = -2 * pi / n
    w = complex(cos(phi), sin(phi))
    Pe, Po = P[::2], P[1::2]
    ye, yo = IOFT_Radix2(Pe), IOFT_Radix2(Po)
    y = [0] * n 
    for j in range(n//2):
        y[j] = ye[j] + w**j * yo[j]
        y[j + n//2] = ye[j] - w**j * yo[j]
    if n >= 2:
        y = [complex_round(v, output_bits) for v in y]
    print(f"n={n}  y={y}")
    return y

def test1():
    print("***** Test Radix-4 OFT and IOFT *****")
    print(f"N: {N}  inputs: {input_bits} bits  output: {output_bits} bits")
    if input_bits == 1:
        lo, hi = 0, 1
    else:
        lo, hi = -2**(input_bits-1), 2**(input_bits-1) - 1
    p = [random.randint(lo, hi) for i in range(N)]
    print("Input p:\n", p)
    ## FFT
    y = OFT_Radix4(p)
    print("OFT(p):\n", y)
    # check FFT results when rounding is not applied
    # correct = True
    # y2 = np.fft.fft(p)
    # for i in range(N):
        # err = y[i] - y2[i]
        # if err.real > 0.01 or err.imag > 0.01:
            # correct = False
            # break
    # if correct:
        # print("*** Same FFT results as numpy ***")
    # else:
        # print("*** Result may be wrong! ***")
        # exit()
    # IFFT
    p2 = IOFT_Radix4(y)
    p2 = [round(v.real/N) for v in p2]
    print("IOFT(OFT(p)):\n", p2)
    err_rate = sum([1 if p[i]!=p2[i] else 0 for i in range(N)]) / N
    print(f"Error rate: {err_rate}")

if __name__ == "__main__":
    test1()

