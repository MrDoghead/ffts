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
    # print("* checking prime")
    if num <= 1:
        return False
    
    for i in range(2, int(math.sqrt(num))+1):
        if (num % i) == 0:
        #    print(num, "is not a prime")
        #    print(i,"乘于",num//i,"是",num)
            return False
    # print(num, "is a prime")
    return True


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
            print("find proot:", g)
            primitive_roots.append(g)
            if get_min:
                break
        exit()
    return primitive_roots

def get_prime_factors(n):
    factors = []
    x = 2
    next_prime = False
    while(x<=n):
        # print(n, x)
        if n % x == 0:
            if x not in factors:
                factors.append(x)
            n = n // x
        else:
            next_prime = True
        # get next prime
        while(next_prime or not is_prime(x)):
            x += 1
            next_prime = False

    return factors

def fast_min_primitive_root(p):
    """only work for prime number"""
    assert(is_prime(p))

    s = p - 1
    factors = get_prime_factors(s)
    ts = [s//v for v in factors]
    for g in range(2, p):
        # actually you can skip g=g_prev^2
        cnt = 0
        for t in ts:
            if qpow(g, t, p) == 1:
                break
            else:
                cnt += 1
        if cnt == len(ts):
            return g
    return 0

if __name__ == "__main__":
    P = 998244353
    # P = 68719403009
    # P = 68719230977
    # P = 137438822401 
    # root = primitive_root(P, get_min=True)
    root = fast_min_primitive_root(P)
    print(root)

