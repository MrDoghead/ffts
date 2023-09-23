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

// Returns true if n is prime
bool isPrime(long n)
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

    for (long i = 5; i * i <= n; i = i + 6)
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
long qpow(long x, long y, long p)
{
    long res = 1; // Initialize result

    x = x % p; // Update x if it is more than or
    // equal to p

    while (y > 0)
    {
        // If y is odd, multiply x with result
        if (y & 1)
            res = (res * x) % p;

        // y must be even now
        y = y >> 1; // y = y/2
        x = (x * x) % p;
    }
    return res;
}

long gcd(long a, long b)
{
    long r = a % b;
    while (r != 0)
    {
        a = b;
        b = r;
        r = a % b;
    }
    return b;
}

long eular(long x)
{
    if (isPrime(x))
    {
        cout << x << " is a prime" << endl;
        return x - 1;
    }
    cout << x << " is not a prime" << endl;

    long cnt = 0;
    for (long i = 1; i < x; i++)
    {
        if (gcd(i, x) == 1)
        {
            cnt += 1;
        }
        
    }
    return cnt;
}

long getMinOrder(long g, long n, long p)
{
    for (long i = 1; i < n+1; i++)
    {
        if (i % 1000000 == 0)
            cout << "** cal order " << i << "..." << endl;
        if (qpow(g, i, p) == 1)
            return i;
    }
    return 0;
}

vector<long> findPrimitiveRoot(long x, bool get_min=true)
{
    long n = eular(x);
    cout << "Eular: "<< n << endl;
    vector<long> roots;
    for (long g = 2; g < x; g++)
    {
        cout << "g: " << g << endl;
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

// Driver code
int main()
{
    // long P = 998244353;
    long P = 68719403009;
    vector<long> roots = findPrimitiveRoot(P, true);

    cout << "Get primitive root of " << P << ":" << endl;;
    for (long x : roots)
        cout << x << " ";
    cout << endl;

    return 0;
}