import random
import math
# define some constants
P = 998244353
G = 3
GI = 332748118

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

def NTT(a, is_forward=True):
    """another way of ntt implementation, i dont like it"""
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

def test1():
    print("##### test ntt and intt #####")
    N = 16
    p = [random.randint(0, 15) for i in range(N)]
    # p = list(range(16))
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

if __name__ == "__main__":
    test1()
    # test2()


