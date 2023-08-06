import random
from math import pi, sin, cos

J = complex(0,1)

def FFT_Radix2(P):
    n = len(P)
    if n == 1:
        return P
    phi = 2 * pi / n
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
    phi = -2 * pi / n
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
    y = FFT_Radix2(p)
    print("y: ", y)
    p2 = IFFT_Radix2(y)
    p2 = [v/N for v in p2]
    print("pp: ", p2)

def test2():
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

def test3():
    print("##### test radix-4 fft and ifft #####")
    N = 64
    p = [random.randint(-8, 7) for i in range(N)]
    # p = [random.randint(0, 1) for i in range(N)]
    print("p: ", p)
    y = FFT_Radix4(p)
    print("y: ", y)
    p2 = IFFT_Radix4(y)
    p2 = [round(v.real/N) for v in p2]
    print("pp: ", p2)
    err_rate = sum([1 if p[i]!=p2[i] else 0 for i in range(N)]) / N
    print(f"Error rate: {err_rate}")


if __name__ == "__main__":
    # test1()
    # test2()
    test3()

