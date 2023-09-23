import random
import math
from math import pi, sin, cos

K = 4 # exponent of 2^K
N = 2**K # polynomial
input_bits = 4
J = complex(0,1)
R = 4 # simu Radix-R
Rs = [R**z for z in range(1,16)]
NRUN = 10000
oft_in_bits = 5 # int-X
oft_out_bits = oft_in_bits + 2

"""
Assumption:
    1. omac input precision uint4 [0,15]
    2. omac weight precision w [-1,1]
"""

def to_intN(x, n, sign=True):
    y = round(x)
    lo = 1 - (1 << n-1) if sign else 0
    hi = (1 << n-1) - 1 if sign else (1 << n) - 1
    if y < lo:
        return lo
    elif y > hi:
        return hi
    else:
        return y

def complex_round(v, n, sign=True):
    rounded_real = to_intN(v.real, n, sign)
    rounded_imag = to_intN(v.imag, n, sign)
    return complex(rounded_real, rounded_imag)

def check_bitwidth(x, n, sign=True):
    lo = 1 - (1 << n-1) if sign else 0
    hi = (1 << n-1) - 1 if sign else (1 << n) - 1
    if x < lo or x > hi:
        print("check_bitwidth", x)
        print(lo, hi)
        return False
    return True

def check_complex_bitwidth(c, n, sign=True):
    if check_bitwidth(c.real, n, sign) and check_bitwidth(c.imag, n, sign):
        return True
    print("check_complex_bitwidth", False)
    return False

def check_complex_list_bitwidth(a, n, sign=True):
    for c in a:
        if not check_complex_bitwidth(c, n, sign):
            print("check_complex_list_bitwidth", c)
            return False
    return True

def get_bitwidth(x):
    n_bits = len(bin(x).split('b')[-1])
    if x < 0:
        n_bits += 1
    return n_bits

def get_complex_bitwidth(c):
    return get_bitwidth(c.real), get_bitwidth(c.imag)

def get_max_bitwidth(a):
    max_bitwidth = 0
    for c in a:
        v1, v2 = get_complex_bitwidth(c)
        if max(v1,v2) > max_bitwidth:
            max_bitwidth = max(v1,v2)
    return max_bitwidth

def oft_r4(a, w, forward=True):
    """
    A 4-in-4-out oft unit
    a: an input vector with size 4, `input_bits`
    return: an output vector with size 4, `output_bits`
    """
    assert len(a)==4
    assert check_complex_list_bitwidth(a, oft_in_bits, sign=True), f"oft_r4 input {a} exceeds {oft_in_bits}-bit range"
    z = [0] * 4
    if forward:
        z[0] = a[0] + 1 * w * a[1] + 1 * w**2 * a[2] + 1 * w**3 * a[3]
        z[1] = a[0] - J * w * a[1] - 1 * w**2 * a[2] + J * w**3 * a[3]
        z[2] = a[0] - 1 * w * a[1] + 1 * w**2 * a[2] - 1 * w**3 * a[3]
        z[3] = a[0] + J * w * a[1] - 1 * w**2 * a[2] - J * w**3 * a[3]
    else:
        z[0] = a[0] + 1 * w * a[1] + 1 * w**2 * a[2] + 1 * w**3 * a[3]
        z[1] = a[0] + J * w * a[1] - 1 * w**2 * a[2] - J * w**3 * a[3]
        z[2] = a[0] - 1 * w * a[1] + 1 * w**2 * a[2] - 1 * w**3 * a[3]
        z[3] = a[0] - J * w * a[1] - 1 * w**2 * a[2] + J * w**3 * a[3]
    z = [complex_round(v, oft_out_bits) for v in z]
    assert check_complex_list_bitwidth(z, oft_out_bits, sign=True), f"oft_r4 output {z} exceeds {oft_out_bits}-bit range"
    return z

"""
Radix-4, poly-N (2^2 ~ 2^16), oft uint4
level   n       in      out     n_oft
0       65536   60      76      15 * 16384 * N/65536
1       16384   46      60      12 * 4096  * N/16384
2       4096    34      46      9  * 1024  * N/4096  
3       1024    24      34      6  * 256   * N/1024
4       256     16      24      4  * 64    * N/256
5       64      10      16      3  * 16    * N/64
6       16      6       10      2  * 4     * N/16
7       4       4       6       1  * 1     * N/4
"""
oft_r4_map = {4:1, 16:2, 64:3, 256:4, 1024:6, 4096:9, 16384:12, 65536:15}
ioft_r4_map = {4:2, 16:3, 64:4, 256:6, 1024:9, 4096:12, 16384:15, 65536:19}

def get_complex_slot(p, k, l):
    """
    get k-th slot with l bitwidth from the list p 
    """
    pk = []
    for c in p:
        re, im = c.real, c.imag
        re_ = (int(re) >> k*l) & (1<<l)-1
        im_ = (int(im) >> k*l) & (1<<l)-1
        pk.append(complex(re_, im_))
    return pk

def split_complex_pos_and_eng(a):
    a_pos = []
    a_neg = []
    for c in a:
        re, im = c.real, c.imag
        if re<0 and im>=0:
            re_pos = 0
            re_neg = -re
            im_pos = im
            im_neg = 0
        elif re>=0 and im<0:
            re_pos = re
            re_neg = 0
            im_pos = 0
            im_neg = -im
        elif re<0 and im<0:
            re_pos = 0
            re_neg = -re
            im_pos = 0
            im_neg = -im
        else:
            re_pos = re
            re_neg = 0
            im_pos = im
            im_neg = 0
        a_pos.append(complex(re_pos, im_pos))
        a_neg.append(complex(re_neg, im_neg))
    return a_pos, a_neg


def OFT_Radix4(P, forward=True):
    n = len(P)
    if n == 1:
        return P
    P0, P1, P2, P3 = P[::4], P[1::4], P[2::4], P[3::4]
    y0, y1, y2, y3 = OFT_Radix4(P0, forward), OFT_Radix4(P1, forward), OFT_Radix4(P2, forward), OFT_Radix4(P3, forward)
    y = [0] * n
    n_slots = oft_r4_map[n] if forward else ioft_r4_map[n] 
    phi = -2 * pi / n if forward else 2 * pi / n 
    w = complex(cos(phi), sin(phi))
    for j in range(n//4):
        a = [y0[j], y1[j], y2[j], y3[j]]
        a_pos, a_neg = split_complex_pos_and_eng(a)

        for k in range(n_slots):
            a_pos_ = get_complex_slot(a_pos, k, 4)
            a_neg_ = get_complex_slot(a_neg, k, 4)
            y_pos0, y_pos1, y_pos2, y_pos3 = oft_r4(a_pos_, w**j, forward)
            y_neg0, y_neg1, y_neg2, y_neg3 = oft_r4(a_neg_, w**j, forward)
            y[j]          += (y_pos0 - y_neg0) * (16**k)
            y[j + n*1//4] += (y_pos1 - y_neg1) * (16**k)
            y[j + n*2//4] += (y_pos2 - y_neg2) * (16**k)
            y[j + n*3//4] += (y_pos3 - y_neg3) * (16**k)

    return y

def OFT_Radix2(P):
    n = len(P)
    if n == 1:
        return P
    phi = 2 * pi / n
    w = complex(cos(phi), sin(phi))
    Pe, Po = P[::2], P[1::2]
    ye, yo = OFT_Radix2(Pe), OFT_Radix2(Po)
    y = [0] * n 
    for j in range(n//2):
        y[j]        = ye[j] + w**j * yo[j]
        y[j + n//2] = ye[j] - w**j * yo[j]
    # if n in Rs:
        # y = [complex_round(v, output_bits) for v in y]
    # print(f"n={n}  y={y}")
    return y

def IOFT_Radix2(P):
    """note: divide the final result by n"""
    n = len(P)
    if n == 1:
        return P
    phi = -2 * pi / n
    w = complex(cos(phi), sin(phi))
    Pe, Po = P[::2], P[1::2]
    ye, yo = IOFT_Radix2(Pe), IOFT_Radix2(Po)
    y = [0] * n 
    for j in range(n//2):
        y[j]        = ye[j] + w**j * yo[j]
        y[j + n//2] = ye[j] - w**j * yo[j]
    if n in Rs:
        y = [complex_round(v, output_bits) for v in y]
    return y

def check_fft_result(p, y, eps=0.01):
    y2 = np.fft.fft(p)
    for i in range(N):
        err = y[i] - y2[i]
        if err.real > eps or err.imag > eps:
            print("*** Result may be wrong! ***")
            return False
    print("*** Same FFT results as numpy ***")
    return True

def random_init_poly():
    if input_bits == 1:
        lo, hi = 0, 1
    else:
        lo, hi = 1-(1<<input_bits-1), (1<<input_bits-1)-1
    print(f"random init poly from [{lo}, {hi}]")
    p = [random.randint(lo, hi) for i in range(N)]
    return p

def test1():
    print("***** Test Radix-4 OFT and IOFT *****")
    print(f"N: {N}  inputs: {input_bits} bits")
    p = random_init_poly()
    print("Input p:\n", p)
    # print("np fft:\n", np.fft.fft(p))
    ## FFT
    y = OFT_Radix4(p, True)
    print("OFT(p):\n", y)
    # IFFT
    p2 = OFT_Radix4(y, False)
    p2 = [round(v.real/N) for v in p2]
    print("IOFT(OFT(p)):\n", p2)
    err_rate = sum([1 if p[i]!=p2[i] else 0 for i in range(N)]) / N
    print(f"Error rate: {err_rate}")

def test3():
    nrun = NRUN
    cnt_err = 0
    print(f"***** Test Radix-4 OFT {nrun} times, N: 2^{K} in: {input_bits} R: {R} *****")
    for i in range(nrun):
        if input_bits == 1:
            lo, hi = 0, 1
        else:
            lo, hi = -2**(input_bits-1), 2**(input_bits-1) - 1
        p = [random.randint(lo, hi) for i in range(N)]
        # y = OFT_Radix4(p)
        # p2 = IOFT_Radix4(y)
        y = OFT_Radix2(p)
        p2 = IOFT_Radix2(y)
        p2 = [round(v.real/N) for v in p2]
        err = sum([1 if p[i]!=p2[i] else 0 for i in range(N)]) / N
        if err != 0:
            cnt_err += 1
        print(f"Run-{i} err: {err}")
    err_rate = cnt_err / nrun
    print(f"Final Error Rate: {err_rate}")

if __name__ == "__main__":
    test1()
    # test3()

