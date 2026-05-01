R = QQ[r1, r2, a, x1, x2,
  A11, A12, A21, A22,
  B111, B112, B122, B211, B212, B222]

f1 = r1 + A11 * x1 + A12 * x2
 + B111*x1*x1 + B112*x1*x2 + B122*x2*x2
f2 = r2 + A21 * x1 + A22 * x2
 + B211*x1*x1 + B212*x1*x2 + B222*x2*x2

g1 = x1*x2
g2 = diff(x1, f1)*diff(x2,f2)
 -diff(x1,f2)*diff(x2,f1)
I1 = ideal(f1, f2, g1)
I2 = ideal(f1, f2, g2)

L = {A11 => (1-a)*A11, A12 => (1-a)*A12,
 A21 => (1-a)*A21, A22 => (1-a)*A22,
 B111 => a*B111, B112 => a*B112,
 B122 => a*B122, B211 => a*B211,
 B212 => a*B212, B222 => a*B222}

-- zero boundary
J1 = eliminate({x1, x2}, I1)
H1 = J1_0
h1 = sub(H1, L)
Delta_zero = factor(h1)

-- complex boundary
J2 = eliminate({x1,x2}, I2)
H2 = J2_0
h2 = sub(H2, L)
Delta_complex = factor(h2)
