/* 
C++ program to find the primitive root of p
If the multiplicative order of a number r modulo n is equal to Euler Totient Function ?(n) 
( note that the Euler Totient Function for a prime n is n-1), then it is a primitive root. 
e.g.,
P = 7
Euler_Totient_Function(7) = 6
we have 3^6 = 1 (mod 7)
*/
#include <iostream>
#include <math.h>
#include <vector>
using namespace std;

typedef __int128_t int128;
typedef unsigned long long uint64;

inline void printInt128(int128 x)
{
    if (x < 0)
    {
        putchar('-');
        x = -x;
    }
    if (x > 9)
        printInt128(x / 10);
    putchar(x % 10 + '0');
}

// Returns true if n is prime
template <class T>
T isPrime(T n)
{
    cout << "* Checking prime" << endl;
    // Corner cases
    if (n <= 1)
        return false;
    if (n <= 3)
        return true;

    // This is checked so that we can skip
    // middle five numbers in below loop
    if (n % 2 == 0 || n % 3 == 0)
        return false;

    for (T i = 5; i * i <= n; i = i + 6)
    {
        if (n % i == 0 || n % (i + 2) == 0)
        {
            // cout << i << " or " << (i+2) << endl;
            return false;
        }
    }

    return true;
}

/* Iterative Function to calculate (x^y)%p in O(logy) */
template <class T>
T qpow(T x, T y, T p)
{
    T res = 1; // Initialize result
    x = x % p;

    while (y != 0)
    {
        // If y is odd, multiply x with result
        if (y & 1)
            res = (res * x) % p;

        y = y >> 1;
        x = (x * x) % p;
    }
    return res;
}

template <class T>
T gcd(T a, T b)
{
    T r = a % b;
    while (r != 0)
    {
        a = b;
        b = r;
        r = a % b;
    }
    return b;
}

template <class T>
T eular(T x)
{
    if (isPrime(x))
    {
        cout << "a prime" << endl;
        return x - 1;
    }
    cout << "not a prime" << endl;

    T cnt = 0;
    for (T i = 1; i < x; i++)
    {
        if (gcd(i, x) == 1)
        {
            cnt += 1;
        }
    }
    return cnt;
}

template <class T>
T getMinOrder(T g, T n, T p)
{
    for (T i = 1; i < n + 1; i++)
    {
        if (i % 1000000000 == 0)
        {
            printInt128(i);
            cout << " ..." << endl;
            // cout << "** cal order " << i << "..." << endl;
        }
        if (qpow(g, i, p) == 1)
            return i;
    }
    return 0;
}

template <class T>
vector<T> findPrimitiveRoot(T x, bool get_min = true)
{
    T n = eular(x);
    // cout << "Eular: "<< n << endl;
    vector<T> roots;
    for (T g = 2; g < x; g++)
    {
        cout << "g: ";
        printInt128(g);
        cout << endl;
        if (gcd(g, x) != 1)
            continue;
        if (getMinOrder(g, n, x) == n)
        {
            roots.push_back(g);
            if (get_min)
                return roots;
        }
    }

    return roots;
}

template <class T>
void printResult(T P, vector<T> roots)
{
    cout << "Get primitive root of " << P << ":" << endl;
    for (T x : roots)
        cout << x << " ";
    cout << endl;
}

void printResultInt128(int128 P, vector<int128> roots)
{
    cout << "Get primitive root of ";
    printInt128(P);
    cout << ":" << endl;
    for (int128 x : roots)
    {
        printInt128(x);
        cout << endl;
    }
}

int main()
{
    // uint64 P = 31;
    // uint64 P = 998244353; // 30bits
    // vector<uint64> roots = findPrimitiveRoot(P, true);
    // printResult(P, roots);

    int128 P = 68719403009; // 36bits
    // int128 P = 68719230977; // 36bits
    // int128 P = 137438822401; // 37bits
    vector<int128> roots = findPrimitiveRoot(P, true);
    printResultInt128(P, roots);

    return 0;
}