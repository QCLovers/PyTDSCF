/**
 * @brief primints_cls.py functions in c++. Compile with gcc and c++17.
 *        ex.) For Linux, "g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) $1.cpp -o $1$(python3-config --extension-suffix)"
 *             For MacOS, "g++ -O3 -Wall -shared -std=c++17 -undefined dynamic_lookup $(python3.8 -m pybind11 --includes) $1.cpp -o $1$(python3-config --extension-suffix)"
 *        Please do not forget fixing PYTHONPATH to import this library.
 */

#include <pybind11/pybind11.h>
#include <iostream>
#include <cmath>

const double au_in_cm1 = 219474.6313631965;

void print(const auto s) {std::cout << s << std::endl;return;}

/**
 * COMBINATION
 * @brief see https://caprest.hatenablog.com/entry/2016/05/29/181102
 * @param n
 * @return std::vector<std::vector<long long>>
 */
std::vector<std::vector<long long>> calc_comb(int n) {
  std::vector<std::vector<long long>> v(n+1,std::vector<long long>(n+1,0));
  for (int i = 0; i < n+1; i++) {
    v[i][0] = 1;
    v[i][i] = 1;
  }
  for (int j = 1; j < n+1; j++) {
    for (int k = 1; k < j; k++) {
      v[j][k] = (v[j-1][k-1] + v[j-1][k]);
    }
  }
  return v;
}

std::vector<std::vector<double>> d_comb(int n) {
    const std::vector<std::vector<long long>> lref = calc_comb(n);
    std::vector<std::vector<double>> ret(n+1, std::vector<double>(n+1, 0.0));
    for (int i = 0; i <= n; i++)
        for (int j = 0; j <= n; j++)
            ret[i][j] += double(lref[i][j]);
    return ret;
}

unsigned long long fact(int n) {
    unsigned long long ret = 1;
    long long n_ = n;
    while(n_ > 0) {ret *= n_; n_--;}
    return ret;
}

unsigned long long fact2(int n) {
    unsigned long long ret = 1;
    long long n_ = n;
    while(n_ > 0) {ret *= n_; n_-=2;}
    return ret;
}

/**
 * @fn
 * Get HO overlap integral matrix element (scalar !!) <HO_v0(omega)|HO_v1(omega')>
 * @brief     See also J.-L. Chang (2005) J.Mol.Spec. 232, 102-104 \
            https://doi.org/10.1016/j.jms.2005.03.004
 * @param v0 int : The order of bra HO primitive.
 * @param v1 int : The order of bra HO primitive.
 * @param freq_cm1_bra double : pbas_bra.freq_cm1.
 * @param freq_cm1_ket double : pbas_ket.freq_cm1.
 * @param origin_bra double : pbas_bra.origin.
 * @param origin_ket double : pbas_ket.origin.
 * @return double : The overlap integral (density) matrix element of HO primitive basis.
 * @
 */
double ovi_HO_FBR_cpp(int v0, int v1,
                      double freq_cm1_bra, double freq_cm1_ket,
                      double origin_bra,   double origin_ket) {

    const double a0 = freq_cm1_bra / au_in_cm1 / 1.0; //(pbas_bra.freq_cm1 / const.au_in_cm1) / 1.0 # omega / hbar
    const double a1 = freq_cm1_ket / au_in_cm1 / 1.0; //(pbas_ket.freq_cm1 / const.au_in_cm1) / 1.0 # omega / hbar
    //x0 = pbas_bra.origin # normal coordinate (i.e. m.w.c. sqrt(m_e) * bohr)
    //x1 = pbas_ket.origin # normal coordinate (i.e. m.w.c. sqrt(m_e) * bohr)
    const double x0 = origin_bra / std::sqrt(a0); //pbas_bra.origin / math.sqrt(a0) // zeta
    const double x1 = origin_ket / std::sqrt(a1); //pbas_ket.origin / math.sqrt(a1) // zeta

    const double d = x1 - x0;
    const double b0 = - a1 * std::sqrt(a0) * d / (a0 + a1);
    const double b1 = + a0 * std::sqrt(a1) * d / (a0 + a1);
    const std::vector<std::vector<double>> comb = d_comb(std::max(v0,v1));

    double val = 0.0;
    for (int k0 = 0; k0 <= v0; k0++) {
        for (int k1 = 0; k1 <= v1; k1++) {
            if ( k0+k1 % 2 == 0 ) {
                const int K = ( k0+k1 ) / 2;
                val +=  comb[v0][k0] * comb[v1][k1]
                      * std::hermite(v0-k0, b0) * std::hermite(v1-k1, b1)
                      * std::pow(2*std::sqrt(a0), k0) * std::pow(2*std::sqrt(a1), k1)
                      * double(fact2(2*K-1)) / std::pow(a0+a1, K);
            }
        }
    }
    const double S = (a0 * a1 * d * d) / (a0 + a1);
    const double A = 2.0 * std::sqrt(a0*a1) / (a0 + a1);
    const double C = std::sqrt((A * std::exp(-S)) / (std::pow(2,v0+v1) * double(fact(v0)*fact(v1))));

    return C * val;
}

/**
 * @fn
 * Get HO q^n integral matrix element (scalar !!) <HO_v0(omega)|q^n|HO_v1(omega')>
 * @brief     See also J.-L. Chang (2005) J.Mol.Spec. 232, 102-104 \
            https://doi.org/10.1016/j.jms.2005.03.004
 * @param v0 int : The order of bra HO primitive.
 * @param v1 int : The order of bra HO primitive.
 * @param freq_cm1_bra double : pbas_bra.freq_cm1.
 * @param freq_cm1_ket double : pbas_ket.freq_cm1.
 * @param origin_bra double : pbas_bra.origin.
 * @param origin_ket double : pbas_ket.origin.
 * @param norder int : The order of operator q^n.
 * @return double : The integral (density) of HO primitive with q^n <v0|q^n|v1>.
 * @
 */
double poly_HO_FBR_cpp(int v0, int v1,
                       double freq_cm1_bra, double freq_cm1_ket,
                       double origin_bra,   double origin_ket,
                       int norder) {

    const double a0 = freq_cm1_bra / au_in_cm1 / 1.0;
    const double a1 = freq_cm1_ket / au_in_cm1 / 1.0;
    const double x0 = origin_bra / std::sqrt(a0);
    const double x1 = origin_ket / std::sqrt(a1);

    const double d = x1 - x0;
    const double b0 = - a1 * std::sqrt(a0) * d / (a0 + a1);
    const double b1 = + a0 * std::sqrt(a1) * d / (a0 + a1);
    const double r = - a1 * d / (a0 + a1);
    const std::vector<std::vector<double>> comb = d_comb(std::max(norder,std::max(v0,v1)));

    double val = 0.0e+0;
    for (int k2 = 0; k2 <= norder; k2++) {
        for (int k0 = 0; k0 <= v0; k0++) {
            for (int k1 = 0; k1 <= v1; k1++) {
                if ( (k0+k1+k2) % 2 == 0) {
                    const int K = ( k0+k1+k2 ) / 2;
                    val +=  comb[v0][k0] * comb[v1][k1] * comb[norder][k2]
                          * std::hermite(double(v0-k0), b0) * std::hermite(double(v1-k1), b1)
                          * std::pow(r, double(norder-k2)) * std::pow(2.0*std::sqrt(a0), double(k0)) * std::pow(2.0*std::sqrt(a1), double(k1))
                          * double(fact2(2*K-1)) / std::pow(a0+a1, double(K));
                }
            }
        }
    }

    const double S = (a0 * a1 * d * d) / (a0 + a1);
    const double A = 2.0 * std::sqrt(a0*a1) / (a0 + a1);
    const double C = std::sqrt((A * std::exp(-S)) / (std::pow(2.0, double(v0+v1)) * double(fact(v0)*fact(v1))));

    return C * val;
}

PYBIND11_MODULE(_primints, m)
{
    m.doc() = "primints_cls.py functions in c++";
    m.def("ovi_HO_FBR_cpp", &ovi_HO_FBR_cpp, "similar to ovi_HO_FBR in _primints_cls.py");
    m.def("poly_HO_FBR_cpp", &poly_HO_FBR_cpp, "similar to poly_HO_FBR in _primints_cls.py");
}
