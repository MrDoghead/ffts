import random
import math
import numpy as np
# define some constants
P = 998244353
G = 3
GI = 332748118 # G * GI mod P = 1 

class BarrettReducer:
    """https://www.nayuki.io/page/barrett-reduction-algorithm"""
    modulus: int
    shift: int
    factor: int
    def __init__(self, mod: int):
        if mod <= 0:
            raise ValueError("Modulus must be positive")
        if mod & (mod - 1) == 0:
            raise ValueError("Modulus must not be a power of 2")
        self.modulus = mod
        self.shift = mod.bit_length() * 2
        self.factor = (1 << self.shift) // mod

    # For x in [0, mod^2), this returns x % mod.
    def reduce(self, x: int) -> int:
        mod = self.modulus
        assert 0 <= x < mod**2
        t = (x - ((x * self.factor) >> self.shift) * mod)
        return t if (t < mod) else (t - mod)

def qpow(a, n, p):
    """a^n mod p"""
    ans = 1
    a = (a % p + p) % p
    while n:
        if n & 1:
            ans = (a * ans) % p
        a = (a * a) % p
        n >>= 1
    return ans

def get_ntt_matrix(n, forward=True):
    V = np.zeros((n,n), dtype=np.int64)
    if forward:
        gn = qpow(G, (P-1)//n, P)
    else:
        gn = qpow(GI, (P-1)//n, P)
    for i in range(n):
        for j in range(n):
            V[i,j] = qpow(gn, i*j, P)
    return V

def NTT_Radix2(A):
    n = len(A)
    if n == 1:
        return A
    gn = qpow(G, (P-1)//n, P)
    Ae, Ao = A[::2], A[1::2]
    ye, yo = NTT_Radix2(Ae), NTT_Radix2(Ao)
    y = [0] * n
    g = 1
    for j in range(n//2):
        r = qpow(gn, j, P) * yo[j]
        y[j]        = (ye[j] + r) % P
        y[j + n//2] = (ye[j] - r) % P
    return y

def INTT_Radix2(A):
    n = len(A)
    if n == 1:
        return A
    gn = qpow(GI, (P-1)//n, P)
    Ae, Ao = A[::2], A[1::2]
    ye, yo = INTT_Radix2(Ae), INTT_Radix2(Ao)
    y = [0] * n
    g = 1
    for j in range(n//2):
        r = qpow(gn, j, P) * yo[j]
        y[j]        = (ye[j] + r) % P
        y[j + n//2] = (ye[j] - r) % P
    return y

g1_4  = qpow(G, (P-1)//4, P) # g^{N/4}_N = g^1_4 = G^{(P-1)/4}
gi1_4 = qpow(GI, (P-1)//4, P) # gi^{N/4}_N = gi^1_4 = GI^{(P-1)/4} 

def NTT_Radix4(A):
    n = len(A)
    if n == 1:
        return A
    gn   = qpow(G, (P-1)//n, P)
    A0, A1, A2, A3 = A[::4], A[1::4], A[2::4], A[3::4]
    y0, y1, y2, y3 = NTT_Radix4(A0), NTT_Radix4(A1), NTT_Radix4(A2), NTT_Radix4(A3)
    y = [0] * n
    for j in range(n//4):
        r1 = qpow(gn, 1*j, P) * y1[j]
        r2 = qpow(gn, 2*j, P) * y2[j]
        r3 = qpow(gn, 3*j, P) * y3[j]
        y[j]          = (y0[j] +        r1 + r2 +        r3) % P
        y[j + n*1//4] = (y0[j] + g1_4 * r1 - r2 - g1_4 * r3) % P
        y[j + n*2//4] = (y0[j] -        r1 + r2 -        r3) % P
        y[j + n*3//4] = (y0[j] - g1_4 * r1 - r2 + g1_4 * r3) % P
    return y

def INTT_Radix4(A):
    n = len(A)
    if n == 1:
        return A
    gn   = qpow(GI, (P-1)//n, P)
    A0, A1, A2, A3 = A[::4], A[1::4], A[2::4], A[3::4]
    y0, y1, y2, y3 = INTT_Radix4(A0), INTT_Radix4(A1), INTT_Radix4(A2), INTT_Radix4(A3)
    y = [0] * n
    for j in range(n//4):
        r1 = qpow(gn, 1*j, P) * y1[j]
        r2 = qpow(gn, 2*j, P) * y2[j]
        r3 = qpow(gn, 3*j, P) * y3[j]
        y[j]          = (y0[j] +         r1 + r2 +         r3) % P
        y[j + n*1//4] = (y0[j] + gi1_4 * r1 - r2 - gi1_4 * r3) % P
        y[j + n*2//4] = (y0[j] -         r1 + r2 -         r3) % P
        y[j + n*3//4] = (y0[j] - gi1_4 * r1 - r2 + gi1_4 * r3) % P
    return y

"""another way of ntt implementation, i dont like it"""
def NTT(a, is_forward=True):
    n = len(a)
    nbit = math.ceil(math.log2(n))
    rev = list(range(n))
    y = a
    for i in range(n):
        rev[i] = rev[i >> 1] >> 1 | (rev[i] & 1) << nbit - 1
        if i < rev[i]:
            y[i], y[rev[i]] = y[rev[i]], y[i]

    for step in map(lambda x: 1 << x, range(1, nbit + 1)):
        gn = qpow(G if is_forward else GI, (P - 1) // step, P)
        for i in range(0, n, step):
            g = 1
            for j in range(step >> 1):
                r = g * y[i + j + (step >> 1)]
                y[i + j], y[i + j + (step >> 1)] = (y[i + j] + r) % P, (y[i + j] - r) % P
                g = g * gn % P
    if not is_forward:
        inv = qpow(n, P - 2, P)
        for i in range(n):
            y[i] = y[i] * inv % P
    return y

def bitslicing(p, n):
    """slice a polynomial p into n bit slices"""
    bit_slices = []
    for i in range(n):
        bit_slices.append([0] * len(p))
    for i in range(len(p)):
        x = p[i]
        cnt = 0
        while x > 0 and cnt < n:
            if x & 1:
                bit_slices[cnt][i] = 1
            x = x >> 1
            cnt += 1
    return bit_slices

def test1():
    print("##### test ntt and intt #####")
    N = 16
    # p = [random.randint(0, 15) for i in range(N)]
    p = list(range(16)) * (N-15)
    print("p: ", p)
    ntt_out = NTT_Radix4(p)
    print("ntt(p):", ntt_out)
    intt_out = INTT_Radix4(ntt_out)
    intt_out = [v * qpow(N, P-2, P) % P for v in intt_out]
    print("intt(ntt(p)): ", intt_out)

def test2():
    print("##### compute 12345*67890=838102050 #####")
    N = 16
    p1 = [5, 4, 3, 2, 1] + [0] * (N-5)
    p2 = [0, 9, 8, 7, 6] + [0] * (N-5)
    print(f"p1:\n{p1}\np2:\n{p2}")
    ntt_out1 = NTT_Radix2(p1)
    ntt_out2 = NTT_Radix2(p2)
    print(f"NTT(p1):\n{ntt_out1}")
    print(f"NTT(p2):\n{ntt_out2}")
    prod = [ntt_out1[i] * ntt_out2[i] for i in range(N)]
    print(f"NTT(p1) · NTT(p2):\n{prod}")
    intt_out = INTT_Radix2(prod)
    intt_out = [v * qpow(N, P-2, P) % P for v in intt_out]
    print(f"INTT(NTT(p1) · NTT(p2)):\n{intt_out}")
    result = 0
    for i in range(N):
        result += round(intt_out[i]) * 10**i
    print(f"Recover the product: {result}")

def test3():
    print("##### test bitslicing ntt #####")
    N = 16
    k = 4
    p1 = [random.randint(0, 2**k-1) for i in range(N)]
    p2 = [random.randint(0, 2**k-1) for i in range(N)]
    print(f"p1 = {p1}\np2 = {p2}")

    # bit slicing
    print("\n***** bit slicing *****")
    bit_slices1 = bitslicing(p1, k)
    bit_slices2 = bitslicing(p2, k)
    print("Bit slices of p1: ")
    for each in bit_slices1:
        print(each)
    print("Bit slices of p2: ")
    for each in bit_slices2:
        print(each)

    # ntt(p1)
    print("\n***** ntt(p1) *****")
    ntt_out_bp1 = []
    for bp in bit_slices1:
        ntt_out_tmp = NTT_Radix2(bp)
        print("bitslice ntt result:", ntt_out_tmp)
        ntt_out_bp1.append(ntt_out_tmp)
    ntt_out1 = [0] * N
    for i in range(N):
        for j in range(k):
            ntt_out1[i] = (ntt_out1[i]  + (ntt_out_bp1[j][i] << j)) % P
    print("ntt_out1:", ntt_out1)
    # ans for p1
    ntt_ans1 = NTT_Radix2(p1)
    print("ntt_ans1:", ntt_ans1)

    # ntt(p2)
    print("\n***** ntt(p2) *****")
    ntt_out_bp2 = []
    for bp in bit_slices2:
        ntt_out_tmp = NTT_Radix2(bp)
        print("bitslice ntt result:", ntt_out_tmp)
        ntt_out_bp2.append(ntt_out_tmp)
    ntt_out2 = [0] * N
    for i in range(N):
        for j in range(k):
            ntt_out2[i] = (ntt_out2[i]  + (ntt_out_bp2[j][i] << j)) % P
    print("ntt_out2:", ntt_out2)
    # ans for p2
    ntt_ans2 = NTT_Radix2(p2)
    print("ntt_ans2:", ntt_ans2)

    # ntt(p1) * ntt(p2) and bitslice the product
    print("\n***** product and bit slicing *****")
    prod = [ntt_out1[i] * ntt_out2[i] for i in range(N)]
    print("ntt(p1) * ntt(p2):", prod)
    k2 = 32*2
    prod_bit_slices = bitslicing(prod, k2)
    print("Bit slices of prod: ")
    for each in prod_bit_slices:
        print(each)

    # intt(prod)
    print("\n***** intt(prod) *****")
    intt_out_bp = []
    for bp in prod_bit_slices:
        intt_out_tmp = INTT_Radix2(bp)
        intt_out_bp.append(intt_out_tmp)
    intt_out = [0] * N
    for i in range(N):
        for j in range(k2):
            intt_out[i] = (intt_out[i]  + (intt_out_bp[j][i] << j)) % P
    print("intt_out:", intt_out)
    # ans for intt
    prod_ans = [ntt_ans1[i] * ntt_ans2[i] for i in range(N)]
    intt_ans = INTT_Radix2(prod_ans)
    print("intt_ans:", intt_ans)

def test4():
    N = 16

    V = get_ntt_matrix(N, True)
    print("V:", V)

    p = np.arange(N)
    print("p:", p)

    ntt_out = NTT_Radix2(p)
    print("ntt_out:", ntt_out)

    ntt_out2 = np.dot(V, p)
    ntt_out2 = ntt_out2 % P
    print("ntt_out2:", ntt_out2)

if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    # print(primitive_root(26, get_min=False))

    N = 16
    V = get_ntt_matrix(N, True)
    np.savetxt(f"V_{N}_ntt.csv", V, delimiter=",", fmt="%d")




