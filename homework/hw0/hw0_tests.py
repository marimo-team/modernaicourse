import mugrade


### Re-inclusion of the reference implementation for Polynomial

class Polynomial:
    """
    This class represents a polynomial as a list of coefficients.  Each item in list
    at position i (zero-indexed), represents the coefficient corresponding to the x^i
    term of the polynomial.  For instance, the list:

    [1, 0, 4, 3]
    would represent the polynomial
    3x^3 + 4x^2 + 1
    """

    def __init__(self, coefficients):
        """ Initialize the coefficients, and make the largest degree coefficient is not zero """
        self.coefficients = coefficients
        while self.coefficients[-1] == 0 and len(self.coefficients) > 1:
            self.coefficients.pop()

    def __eq__(self, value):
        """ Check if two polynomials are equal """
        return self.coefficients == value.coefficients

    def degree(self):
        return len(self.coefficients)-1
                
    def __repr__(self):
        """ Returns a string representation of the polynomial"""
        if len(self.coefficients) == 0:
            return "0"
        terms = []
        for i,c in enumerate(self.coefficients):
            if c != 0:
                if i == 0:
                    terms.append(f"{c}")
                elif i == 1:
                    terms.append(f"{c}x")
                else:
                    terms.append(f"{c}x^{i}")
        if len(terms) == 0:
            terms.append("0")
        return " + ".join(reversed(terms))


### Test/submit cases

def test_add(add):
    assert(add(5,6) == 11)
    assert(add(2.1,2.3) == 4.4)
    assert(isinstance(add(4,2.1), float))

def submit_add(add):
    mugrade.submit(add(-1,5))
    mugrade.submit(add(1,2.5))
    mugrade.submit(type(add(1,3)))


def test_primes(primes):
    p = primes(10)
    assert(isinstance(p,list))
    assert(p == [2,3,5,7])


def submit_primes(primes):
    p = primes(100)
    mugrade.submit(p)
    p = primes(10000)
    mugrade.submit(len(p))
    mugrade.submit(p[-1])

def test_poly_add(poly_add):
    p1 = Polynomial([1, 5, 0, 5])
    p2 = Polynomial([0, 2])
    p3 = Polynomial([-1, 6, 7, -5])
    p4 = Polynomial([0.3, 0.4, 1.6, 1.9])
    assert(poly_add(p1, p2) == Polynomial([1, 7, 0, 5]))
    assert(poly_add(p1, p3) == Polynomial([0, 11, 7]))
    assert(poly_add(p1, Polynomial([0])) == p1)
    assert(poly_add(p2, p4) == Polynomial([0.3, 2.4, 1.6, 1.9]))


def submit_poly_add(poly_add):
    p1 = Polynomial([2, 4, -1, 3])
    p2 = Polynomial([1, 8, -2])
    p3 = Polynomial([-5, 2, 0, 4])
    p4 = Polynomial([0.9, 0.4, -1.7, 2.5])
    mugrade.submit(poly_add(p1, p2).coefficients)
    mugrade.submit(poly_add(p1, p3).coefficients)
    mugrade.submit(poly_add(p1, Polynomial([0])).coefficients)
    mugrade.submit(poly_add(p2, p4).coefficients)


def test_poly_mul(poly_mul):
    p1 = Polynomial([1, 5, 0, 5])
    p2 = Polynomial([0, 2])
    p3 = Polynomial([-1, 6, 7, -5])
    p4 = Polynomial([0.3, 0.4, 1.6, 1.9])
    assert(poly_mul(p1, p2) == Polynomial([0, 2, 10, 0, 10]))
    assert(poly_mul(p1, p3) == Polynomial([-1, 1, 37, 25, 5, 35, -25]))
    assert(poly_mul(p1, Polynomial([1])) == p1)
    assert(poly_mul(p1, p4) == Polynomial([0.3, 1.9, 3.6, 11.4, 11.5, 8.0, 9.5]))


def submit_poly_mul(poly_mul):
    p1 = Polynomial([2, 4, -1, 3])
    p2 = Polynomial([1, 8, -2])
    p3 = Polynomial([-5, 2, 0, 4])
    p4 = Polynomial([0.9, 0.4, -1.7, 2.5])
    mugrade.submit(poly_mul(p1, p2).coefficients)
    mugrade.submit(poly_mul(p1, p3).coefficients)
    mugrade.submit(poly_mul(p1, Polynomial([0])).coefficients)
    mugrade.submit(poly_mul(p2, p4).coefficients)


def test_poly_derivative(poly_derivative):
    p1 = Polynomial([1, 5, 0, 5])
    p2 = Polynomial([0.3, 0.4, 1.6])
    assert(poly_derivative(p1) == Polynomial([5, 0, 15]))
    assert(poly_derivative(p2) == Polynomial([0.4, 3.2]))
    assert(poly_derivative(Polynomial([0])) == Polynomial([0]))


def submit_poly_derivative(poly_derivative):
    p1 = Polynomial([2, 4, -1, 3])
    p2 = Polynomial([1, 8, -2])
    p3 = Polynomial([0.9, 0.4, -1.7, 2.5])
    mugrade.submit(poly_derivative(p1).coefficients)
    mugrade.submit(poly_derivative(p2).coefficients)
    mugrade.submit(poly_derivative(p3).coefficients)
    mugrade.submit(poly_derivative(poly_derivative(p1)).coefficients)















