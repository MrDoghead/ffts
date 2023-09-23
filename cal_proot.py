"""
An efficient solution of finding Primitive Root is based on the below facts. 
If the multiplicative order of a number r modulo n is equal to Euler Totient Function ?(n) 
( note that the Euler Totient Function for a prime n is n-1), then it is a primitive root. 
e.g.,
P = 7
Euler_Totient_Function(7) = 6
we have 3^6 = 1 (mod 7)
"""
import math

def is_prime(num):
    print("* checking prime")
    if num > 1:
       for i in range(2, int(math.sqrt(num))):
           if (num % i) == 0:
               print(num, "is not a prime")
               print(i,"乘于",num//i,"是",num)
               return False
       else:
           print(num, "is a prime")
           return True
    else:
        return False

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

# 用辗转相除求最大公因子
def gcd(a, b):
    r = a % b
    while(r != 0):
        a = b
        b = r
        r = a % b
    return b

# 欧拉函数-暴力循环版
def euler(a):
    print("* cal euler")
    if is_prime(a):
        return a-1
    count = 0
    for i in range(1, a):
        if gcd(a, i) == 1:
            count += 1
    return count

def get_min_order(g, p, n):
    for i in range(1, n+1):
        if i % 1000000 == 0:
            print(f"** cal order {i} ...")
        if qpow(g, i, p) == 1:
            return i
    return 0

def primitive_root(p, get_min=True):
    """order(g, p) == euler(p)"""
    n = euler(p)
    print("Eular:", n)
    primitive_roots = []
    for g in range(2, p):
        print("g:", g)
        if gcd(g, p) != 1:
            continue
        if get_min_order(g, p, n) == n:
            primitive_roots.append(g)
            if get_min:
                break
    return primitive_roots

if __name__ == "__main__":
    P = 998244353 
    root = primitive_root(P, get_min=True)
    print(root)
