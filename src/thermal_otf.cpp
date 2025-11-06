#include "openmc/thermal_otf.h"

#include "openmc/hdf5_interface.h"
#include "openmc/random_lcg.h"

// Constants 
constexpr double boltz = 0.00008617385;
constexpr double ref_temp_ev = 0.0253;
constexpr double ref_temp_k = ref_temp_ev/boltz;
constexpr double PI {3.141592653589793238462643383279502884};

std::vector<double> eval_chebyshev_orders(double const &x, int const &n)
{
    std::vector<double> T(n + 1);
    T[0] = 1.0;
    if (n > 0) {T[1] = x;}
    for (int i = 2; i <= n; ++i){
        T[i] = 2.0 * x * T[i-1] - T[i-2];
    }
    return T;
}

std::vector<double> eval_cosine_orders(double const &x, int const &n)
{
    std::vector<double> f(n+1);
    double two_pi_x = 2.0*PI*x;
    for (int i = 0; i <= n; ++i){
        f[i] = cos(i*two_pi_x);
    }
    return f;
}

std::vector<double> eval_exponential_orders(double const &x, int const &n)
{
    std::vector<double> f(n + 1);
    f[0] = 1;
    double f1_x = std::exp(x);
    if (n > 0) {f[1] = f1_x;}
    for (int i = 2; i <= n; ++i){
        f[i] = f1_x * f[i-1];
    }
    return f;
}

std::vector<double> eval_inverse_exponential_orders(double const &x, int const &n)
{
    std::vector<double> f(n + 1);
    f[0] = 1;
    double f1_x = std::exp(-x);
    if (n > 0) {f[1] = f1_x;}
    for (int i = 2; i <= n; ++i){
        f[i] = f1_x * f[i-1];
    }
    return f;
}

std::vector<double> eval_inverse_log_power_orders(double const &x, int const &n)
{
    std::vector<double> f(n + 1);
    f[0] = 1;
    double f1_x = 1.0 / std::log(x);
    if (n > 0) {f[1] = f1_x;}
    for (int i = 2; i <= n; ++i){
        f[i] = f1_x * f[i-1];
    }
    return f;
}

std::vector<double> eval_inverse_power_orders(double const &x, int const &n)
{
    std::vector<double> f(n + 1);
    f[0] = 1;
    double f1_x = 1.0 / x;
    if (n > 0) {f[1] = f1_x;}
    for (int i = 2; i <= n; ++i){
        f[i] = f1_x * f[i-1];
    }
    return f;
}

std::vector<double> eval_inverse_sqrt_log_power_orders(double const &x, int const &n)
{
    std::vector<double> f(n + 1);
    f[0] = 1;
    double f1_x = 1.0 / std::sqrt(std::log(x));
    if (n > 0) {f[1] = f1_x;}
    for (int i = 2; i <= n; ++i){
        f[i] = f1_x * f[i-1];
    }
    return f;
}

std::vector<double> eval_inverse_sqrt_power_orders(double const &x, int const &n)
{
    std::vector<double> f(n + 1);
    f[0] = 1;
    double f1_x = 1.0 / std::sqrt(x);
    if (n > 0) {f[1] = f1_x;}
    for (int i = 2; i <= n; ++i){
        f[i] = f1_x * f[i-1];
    }
    return f;
}

double legendre_alpha_recurrence__(double const & x, int const & k){
    return  ((2.0*k + 1.0)/(k + 1.0))*x;
}

double legendre_beta_recurrence__(double const & x, int const & k){
    return - k/(k + 1.0);
}

std::vector<double> eval_legendre_orders(double const &x, int const &n)
{
    std::vector<double> P(n + 1);
    P[0] = 1;
    if (n > 0) {P[1] = x;}
    for (int i = 2; i <= n; ++i){
        P[i] = legendre_alpha_recurrence__(x,i-1)*P[i-1] + legendre_beta_recurrence__(x,i-1)*P[i-2];
    }
    return P;
}

std::vector<double> eval_log_power_orders(double const &x, int const &n)
{
    std::vector<double> f(n + 1);
    f[0] = 1;
    double f1_x = std::log(x);
    if (n > 0) {f[1] = f1_x;}
    for (int i = 2; i <= n; ++i){
        f[i] = f1_x * f[i-1];
    }
    return f;
}

std::vector<double> eval_power_orders(double const &x, int const &n)
{
    std::vector<double> f(n + 1);
    f[0] = 1;
    double f1_x = x;
    if (n > 0) {f[1] = f1_x;}
    for (int i = 2; i <= n; ++i){
        f[i] = f1_x * f[i-1];
    }
    return f;
}

std::vector<double> eval_sine_cosine_orders(double const &x, int const &n)
{
    std::vector<double> f(n+1);
    double two_pi_x = 2*PI*x;
    f[0] = 1;
    for (int i = 1; i <= n; ++i){
        if (i%2 == 0) {f[i] = sin(int(i/2)*two_pi_x);}
        else {f[i] = cos(int(i/2)*two_pi_x);}
    }
    return f;
}

std::vector<double> eval_sine_orders(double const &x, int const &n)
{
    std::vector<double> f(n+1);
    double two_pi_x = 2.0*PI*x;
    f[0] = 1.0;
    for (int i = 1; i <= n; ++i){
        f[i] = sin(i*two_pi_x);
    }
    return f;
}

std::vector<double> eval_sqrt_log_power_orders(double const &x, int const &n)
{
    std::vector<double> f(n + 1);
    f[0] = 1;
    double f1_x = std::sqrt(std::log(x));
    if (n > 0) {f[1] = f1_x;}
    for (int i = 2; i <= n; ++i){
        f[i] = f1_x * f[i-1];
    }
    return f;
}

std::vector<double> eval_sqrt_power_orders(double const &x, int const &n)
{
    std::vector<double> f(n + 1);
    f[0] = 1;
    double f1_x = std::sqrt(x);
    if (n > 0) {f[1] = f1_x;}
    for (int i = 2; i <= n; ++i){
        f[i] = f1_x * f[i-1];
    }
    return f;
}

// Support

double scale_value(double const & x, double const & old_min, double const & old_max, double const & new_min, double const & new_max){
    return ((x - old_min)/(old_max - old_min))*(new_max - new_min) + new_min;
}

double ENDF_interp_scheme_1(double const & x1, double const & x2, double const & y1, double const & y2, double const& x){
    return y1;
}

double ENDF_interp_scheme_2(double const & x1, double const & x2, double const & y1, double const & y2, double const& x){
    return y1 + (y2 - y1) * ((x - x1) / (x2 - x1));
}

double ENDF_interp_scheme_3(double const & x1, double const & x2, double const & y1, double const & y2, double const& x){
    return y1 + (y2 - y1)*(log(x/x1)/log(x2/x1));
}

double ENDF_interp_scheme_4(double const & x1, double const & x2, double const & y1, double const & y2, double const& x){
    return y1*exp(log(y2/y1)*((x - x1)/(x2 - x1)));
}

double ENDF_interp_scheme_5(double const & x1, double const & x2, double const & y1, double const & y2, double const& x){
    return y1*exp(log(x/x1)*(log(y2/y1)/log(x2/x1)));
}

double ENDF_interp(double const & x1, double const & x2, double const & y1, double const & y2, double const & x, int const scheme){
    switch (scheme)
    {
    case 1:
        return ENDF_interp_scheme_1(x1, x2, y1, y2, x);
    case 2:
        return ENDF_interp_scheme_2(x1, x2, y1, y2, x);        
    case 3:
        return ENDF_interp_scheme_3(x1, x2, y1, y2, x);
    case 4:
        return ENDF_interp_scheme_4(x1, x2, y1, y2, x);
    case 5:
        return ENDF_interp_scheme_5(x1, x2, y1, y2, x);
    default:
        throw std::invalid_argument("Scheme type was not recognized.");
    }
}

double bi_interp(double const & x1, double const & x2, double const & y1, double const & y2, 
                 double const & f11, double const & f12, double const & f21, double const & f22, 
                 double const & x, double const & y, 
                 int const x_scheme, int const y_scheme){
    //     | y1  |  y  | y2
    //  --------------------
    //  x1 | f11 |  -  | f12
    //  --------------------
    //  x  |  -  | fxy |  -
    //  --------------------
    //  x2 | f21 |  -  | f22
    double fxy1 = ENDF_interp(x1, x2, f11, f21, x, x_scheme);
    double fxy2 = ENDF_interp(x1, x2, f12, f22, x, x_scheme);
    return ENDF_interp(y1, y2, fxy1, fxy2, y, y_scheme);
}

/** 
 * @brief Calculates the inner product (dot product) of two vectors. 
 * * This function is optimized for contiguous memory and gives a strong hint 
 * to the compiler to use SIMD instructions. 
 * @param iter1 Const iterator to the beginning of the first vector. 
 * @param iter2 Const iterator to the beginning of the second vector. 
 * @param length The number of elements to process. 
 * @return The scalar dot product (double). 
 */ 
inline double inner_product(const std::vector<double>::const_iterator &iter1, 
                     const std::vector<double>::const_iterator &iter2, 
                     const size_t length) { 
    double sum = 0.0; 
    #pragma omp simd reduction(+:sum) 
    for (size_t i = 0; i < length; ++i) { 
        sum += iter1[i] * iter2[i]; 
    } 
    return sum; 
}

typedef std::pair<size_t, size_t> InterpolationIndices;
typedef std::vector<double>::const_iterator Iter;

/**
 * @brief Finds the indices of list of elements that would bracket the search value val
 * @param begin Iterator pointing to the first element of the search range
 * @param end Iterator pointing to the last element of the search range
 * If you use x.end(), you need to subtract one from the iterator
 * @param val Double of the value that is desired to be found
 * @return InterpolationIndicies type (std::pair(size_t, size_t))
 */
InterpolationIndices findSampleInterpolationIndices(const Iter &begin, 
                                                    const Iter &end, 
                                                    const double &val) 
{
    Iter lo = begin + 1; // +1 handles if below grid
    Iter hi = end;
    int len = std::distance(lo, hi);
    while (len > 0) {
        int half = len / 2;
        Iter mid = lo;
        std::advance(mid, half);
        if (*mid < val) {
            lo = mid;
            ++lo;
            len = len - half - 1;
        } else {
            len = half;
        }
    }
    size_t hi_index = std::distance(begin, lo);
    size_t lo_index = hi_index - 1;
    return std::make_pair(lo_index, hi_index);
}

/**
 * @brief Finds the indices of list of elements that would bracket the search value val when linear interpolation is applied between the two list
 * @param begin1 Iterator pointing to the first element of the search range for the lower list
 * @param begin2 Iterator pointing to the first element of the search range for the upper list
 * @param end Iterator pointing to the last element of the search range
 * If you use x.end(), you need to subtract one from the iterator
 * @param val Double of the value that is desired to be found
 * @param scheme Interpolation scheme
 * @return InterpolationIndicies type (std::pair(size_t, size_t))
 */
InterpolationIndices findSampleInterpolationIndices(const Iter &begin1, 
                                                    const Iter &begin2, 
                                                    const Iter &end, 
                                                    const double& x1, 
                                                    const double& x2, 
                                                    const double& x, 
                                                    const double &val, 
                                                    const int scheme) 
{
    Iter lo = begin1 + 1; // +1 handles if below grid
    Iter hi = end;
    int len = std::distance(lo, hi);
    while (len > 0) {
        int half = len / 2;
        Iter mid = lo;
        std::advance(mid, half);
        int dist = std::distance(begin1, mid);
        double interp_val = ENDF_interp(x1, x2, *(begin1+dist), *(begin2+dist), x, scheme);
        if (interp_val < val) {
            lo = mid;
            ++lo;
            len = len - half - 1;
        } else {
            len = half;
        }
    }
    size_t hi_index = std::distance(begin1, lo);
    size_t lo_index = hi_index - 1;
    return std::make_pair(lo_index, hi_index);
}

/**
 * @brief Finds the indices of list of coefficients that would bracket the search value val.
 * This method uses std::inner_product to evaluate the coefficients.
 * @param begin Iterator pointing to the first coefficient in the first set of coefficients of the search range
 * @param end Iterator pointing to the first coefficient in the last set of coefficients of the search range
 * See test_sample_search.cpp to see how to set begin and end properly
 * @param val Double of the value that is desired to be found\
 * @param evaled_basis_points Evaluated f(x) values of the basis functions at the desired x value.  
 * This should be the returned vector from an Evaluation Function
 * @return InterpolationIndicies type (std::pair(size_t, size_t))
 */
InterpolationIndices findSampleCoeffInterpolationIndices(const Iter &begin, 
                                                         const Iter &end, 
                                                         const double &val, 
                                                         const std::vector<double> &evaled_basis_points)
{
    int num_coeffs = evaled_basis_points.size();
    Iter lo = begin + num_coeffs; // +num_coeffs handles if below grid
    Iter hi = end;
    int len = std::distance(lo, hi) / num_coeffs;
    while (len > 0) {
        int half = len / 2;
        Iter mid = lo;
        std::advance(mid, half * num_coeffs);
        double func_val = std::inner_product(mid, mid + num_coeffs, evaled_basis_points.begin(), 0.0);
        // double func_val = inner_product(mid, evaled_basis_points.begin(), num_coeffs);
        if (func_val < val) {
            lo = mid;
            lo += num_coeffs;
            len = len - half - 1;
        } else {
            len = half;
        }
    }
    size_t hi_index = std::distance(begin, lo) / num_coeffs;
    size_t lo_index = hi_index - 1;
    return std::make_pair(lo_index, hi_index);
}

void set_fit_func(ThermalScatteringOTF::FuncPointer& fitting_function, const std::string& fit_func_str)
{
  if (fit_func_str == "Chebyshev"){fitting_function = eval_chebyshev_orders;}
  else if (fit_func_str == "Cosine"){fitting_function = eval_cosine_orders;}
  else if (fit_func_str == "Exponential"){fitting_function = eval_exponential_orders;}
  else if (fit_func_str == "InverseExponential"){fitting_function = eval_inverse_exponential_orders;}
  else if (fit_func_str == "InverseLogPower"){fitting_function = eval_inverse_log_power_orders;}
  else if (fit_func_str == "InversePower"){fitting_function = eval_inverse_power_orders;}
  else if (fit_func_str == "InverseSqrtLogPower"){fitting_function = eval_inverse_sqrt_log_power_orders;}
  else if (fit_func_str == "InverseSqrtPower"){fitting_function = eval_inverse_sqrt_power_orders;}
  else if (fit_func_str == "Legendre"){fitting_function = eval_legendre_orders;}
  else if (fit_func_str == "LogPower"){fitting_function = eval_log_power_orders;}
  else if (fit_func_str == "Power"){fitting_function = eval_power_orders;}
  else if (fit_func_str == "SineCosine"){fitting_function = eval_sine_cosine_orders;}
  else if (fit_func_str == "Sine"){fitting_function = eval_sine_orders;}
  else if (fit_func_str == "SqrtLogPower"){fitting_function = eval_sqrt_log_power_orders;}
  else if (fit_func_str == "SqrtPower"){fitting_function = eval_sqrt_power_orders;}
  else {throw std::out_of_range("Unknown basis function.");}
} 

ThermalScatteringOTF::Inelastic_Fit_1D ThermalScatteringOTF::read_inelastic_fit_1D(const hid_t& group, const char* x_name){
    Inelastic_Fit_1D data;
    std::string fit_func_str;
    openmc::read_dataset(group, x_name, data.x);
    openmc::read_dataset(group, "COEFFS", data.coeffs);
    data.num_coeffs = data.coeffs.size()/data.x.size();
    openmc::read_attribute(group, "MIN_SCALE", data.min_scale);
    openmc::read_attribute(group, "MAX_SCALE", data.max_scale);
    openmc::read_attribute(group, "FITTING_FUNCTION", fit_func_str);
    set_fit_func(data.fit_function, fit_func_str);
    return data;
}

ThermalScatteringOTF::Inelastic_Fit_2D ThermalScatteringOTF::read_inelastic_fit_2D(const hid_t& group, const char* x_name, const char* y_name){
    Inelastic_Fit_2D data;
    std::string fit_func_str;
    openmc::read_dataset(group, x_name, data.x);
    openmc::read_dataset(group, y_name, data.y);
    openmc::read_dataset(group, "COEFFS", data.coeffs);
    data.num_coeffs = data.coeffs.size()/(data.x.size()*data.y.size());
    openmc::read_attribute(group, "MIN_SCALE", data.min_scale);
    openmc::read_attribute(group, "MAX_SCALE", data.max_scale);
    openmc::read_attribute(group, "FITTING_FUNCTION", fit_func_str);
    set_fit_func(data.fit_function, fit_func_str);
    return data;
}

ThermalScatteringOTF::Inelastic_Data ThermalScatteringOTF::read_inelastic_data(const hid_t& group){
    Inelastic_Data data;
    openmc::read_attribute(group, "MIN_T", data.min_t);
    openmc::read_attribute(group, "MAX_T", data.max_t);
    data.xs = read_inelastic_fit_1D(openmc::open_group(group, "XS"), "ENERGY_GRID");
    data.beta = read_inelastic_fit_2D(openmc::open_group(group, "BETA"), "ENERGY_GRID", "CDF_GRID");
    data.alpha = read_inelastic_fit_2D(openmc::open_group(group, "ALPHA"), "BETA_GRID", "CDF_GRID");
    return data;
}

ThermalScatteringOTF::Coherent_Elastic_Data ThermalScatteringOTF::read_coherent_elastic_data(const hid_t& group){
    Coherent_Elastic_Data data;
    openmc::read_dataset(group, "ENERGIES", data.energies);
    openmc::read_dataset(group, "S_VALS", data.s_vals);
    openmc::read_dataset(group, "TEMPERATURES", data.temperatures);
    openmc::read_dataset(group, "INTERPOLATION_LAWS", data.interp_laws);
    return data;
}

ThermalScatteringOTF::Incoherent_Elastic_Data ThermalScatteringOTF::read_incoherent_elastic_data(const hid_t& group){
    Incoherent_Elastic_Data data;
    openmc::read_dataset(group, "DEBYE_WALLERS", data.debye_wallers);
    openmc::read_dataset(group, "TEMPERATURES", data.temperatures);
    openmc::read_attribute(group, "INTERPOLATION_LAW", data.interp_law);
    return data;
}

ThermalScatteringOTF::Elastic_Data ThermalScatteringOTF::read_elastic_data(const hid_t& group){
    Elastic_Data elastic;
    if (openmc::object_exists(group, "Coherent")){
        elastic.coherent = read_coherent_elastic_data(openmc::open_group(group, "Coherent"));
    }
    if (openmc::object_exists(group, "Incoherent")){
        elastic.incoherent = read_incoherent_elastic_data(openmc::open_group(group, "Incoherent"));
    }
    return elastic;
}

ThermalScatteringOTF::OTF_Data ThermalScatteringOTF::read_data(const hid_t& group){
    ThermalScatteringOTF::OTF_Data data;
    openmc::read_attribute(group, "A0", data.A0);
    openmc::read_attribute(group, "BOUND_XS", data.bound_xs);
    if (openmc::object_exists(group, "Elastic")){
        data.elastic = read_elastic_data(openmc::open_group(group, "Elastic"));
    }
    if (openmc::object_exists(group, "Inelastic")){
        data.inelastic = read_inelastic_data(openmc::open_group(group, "Inelastic")); 
    }
    return data;
}

ThermalScatteringOTF::ThermalScatteringOTF(const hid_t& group){
    data = read_data(group);
}
        
ThermalScatteringOTF::Thermal_Cross_Sections ThermalScatteringOTF::calculate_xs(const double& energy, const double& sqrtkT){
    double temperature = (sqrtkT*sqrtkT)/boltz;
    Thermal_Cross_Sections XS;
    XS.inelastic = calculate_inelastic_xs(energy, temperature);
    XS.elastic_coherent = calculate_elastic_coherent_xs(energy, temperature);
    XS.elastic_incoherent = calculate_elastic_incoherent_xs(energy, temperature);
    return XS;
}

void ThermalScatteringOTF::sample_collision(const openmc::NuclideMicroXS& xs_data,
                        const double& inc_energy, const double& sqrtkT, 
                        double& E_out, double& mu,
                        uint64_t* seed){
    double rand_num = openmc::prn(seed)*xs_data.thermal;
    double temperature = (sqrtkT*sqrtkT)/boltz;
    if (rand_num >= xs_data.thermal_elastic){
        sample_inelastic_collision(inc_energy, temperature, E_out, mu, seed);
    }
    else if (rand_num >= xs_data.thermal_elastic_coherent){
        sample_elastic_incoherent_collision(inc_energy, temperature, E_out, mu, seed);
    }
    else {
        sample_elastic_coherent_collision(inc_energy, temperature, E_out, mu, seed);
    }
}

double ThermalScatteringOTF::calculate_elastic_coherent_xs(const double& energy, const double& temperature){
    double xs = 0;
    if (data.elastic && data.elastic->coherent){
        const Coherent_Elastic_Data& coh_data = data.elastic->coherent.value();
        /// NOTE: If the energy is below the first bragg edge, the XS is zero
        if (energy < coh_data.energies[0]){
            xs = 0;
        }
        else{
            InterpolationIndices t_b = findSampleInterpolationIndices(coh_data.temperatures.begin(),
                                                                        coh_data.temperatures.end()-1,
                                                                        temperature);
            InterpolationIndices e_b = findSampleInterpolationIndices(coh_data.energies.begin(),
                                                                        coh_data.energies.end()-1,
                                                                        energy);
            double s_l = coh_data.s_vals[coh_data.energies.size()*t_b.first  + e_b.first];
            double s_u = coh_data.s_vals[coh_data.energies.size()*t_b.second + e_b.first];
            double s =  ENDF_interp(coh_data.temperatures[t_b.first], coh_data.temperatures[t_b.second],
                                    s_l, s_u, temperature, coh_data.interp_laws[t_b.first]);
            xs = s/energy;
        }
    }
    return xs;
}    

double ThermalScatteringOTF::calculate_elastic_incoherent_xs(const double& energy, const double& temperature){
    double xs = 0;
    if (data.elastic && data.elastic->incoherent){
        const Incoherent_Elastic_Data& incoh_data = data.elastic->incoherent.value();
        InterpolationIndices t_b = findSampleInterpolationIndices(incoh_data.temperatures.begin(),
                                                                    incoh_data.temperatures.end()-1,
                                                                    temperature);

        double debye_waller = ENDF_interp(incoh_data.temperatures[t_b.first],
                                            incoh_data.temperatures[t_b.second],
                                            incoh_data.debye_wallers[t_b.first],
                                            incoh_data.debye_wallers[t_b.second],
                                            temperature,
                                            incoh_data.interp_law);
        double ew2 = 2*energy*debye_waller;
        xs = data.bound_xs/2*((1-std::exp(-2*ew2))/ew2);
    }
    return xs;
}

double ThermalScatteringOTF::calculate_inelastic_xs(const double& energy, const double& temperature){
    double xs = 0;
    if (data.inelastic){
        const Inelastic_Data& inel_data = data.inelastic.value();
        double eval_point = scale_value(temperature, 
                                        inel_data.min_t, inel_data.max_t, 
                                        inel_data.xs.min_scale, inel_data.xs.max_scale);
        std::vector<double> basis_points = inel_data.xs.fit_function(eval_point, inel_data.xs.num_coeffs - 1);
        InterpolationIndices e_b = findSampleInterpolationIndices(inel_data.xs.x.begin(), inel_data.xs.x.end()-1, energy);
        double xs_l = std::inner_product(basis_points.begin(), basis_points.end(), 
                                            inel_data.xs.coeffs.begin() + e_b.first*inel_data.xs.num_coeffs, 0.0);
        double xs_u = std::inner_product(basis_points.begin(), basis_points.end(), 
                                            inel_data.xs.coeffs.begin() + e_b.second*inel_data.xs.num_coeffs, 0.0);
        xs = ENDF_interp(inel_data.xs.x[e_b.first], inel_data.xs.x[e_b.second],
                            xs_l, xs_u, energy, 2);
    }
    return xs;
}      

void ThermalScatteringOTF::sample_elastic_coherent_collision(const double& inc_energy, const double& temperature, 
                                        double& out_energy, double& out_angle,
                                        uint64_t* seed){
    out_energy = inc_energy; // No energy change
    const Coherent_Elastic_Data& coh_data = data.elastic->coherent.value();
    double sampled_bragg = 0;
    /// NOTE: If the energy is below the lowest stored energy, the xs should be zero so this collision should not be sampled
    if (inc_energy < coh_data.energies[1]){
        sampled_bragg = coh_data.energies[0];
    }                                    
    else{
        InterpolationIndices e_b = findSampleInterpolationIndices(coh_data.energies.begin(),
                                                                    coh_data.energies.end()-1,
                                                                    inc_energy);
        InterpolationIndices t_b = findSampleInterpolationIndices(coh_data.temperatures.begin(),
                                                                    coh_data.temperatures.end()-1,
                                                                    temperature);
        /// NOTE: Finds the maximum S value based on the energy and then interpolates between the temperatures.
        /// NOTE: The search S value is then just the random number times the maximum S value
        double s_search = openmc::prn(seed) * ENDF_interp(coh_data.temperatures[t_b.first], coh_data.temperatures[t_b.second],
                                                coh_data.s_vals[coh_data.temperatures.size()*t_b.first + e_b.first],
                                                coh_data.s_vals[coh_data.temperatures.size()*t_b.second + e_b.first],
                                                temperature, coh_data.interp_laws[t_b.first]);
        InterpolationIndices s_b = findSampleInterpolationIndices(coh_data.s_vals.begin() + coh_data.temperatures.size()*t_b.first,
                                                                    coh_data.s_vals.begin() + coh_data.temperatures.size()*t_b.second,
                                                                    s_search);
        sampled_bragg = coh_data.energies[s_b.first];
    }
    out_angle = 1 - ((2*sampled_bragg)/inc_energy);
}

void ThermalScatteringOTF::sample_elastic_incoherent_collision(const double& inc_energy, const double& temperature, 
                                            double& out_energy, double& out_angle,
                                            uint64_t* seed){
    out_energy = inc_energy; // No energy change
    const Incoherent_Elastic_Data& incoh_data = data.elastic->incoherent.value();
    InterpolationIndices t_b = findSampleInterpolationIndices(incoh_data.temperatures.begin(), incoh_data.temperatures.end()-1, temperature);
    double debye_waller = ENDF_interp(incoh_data.temperatures[t_b.first], incoh_data.temperatures[t_b.second],
                                        incoh_data.debye_wallers[t_b.first], incoh_data.debye_wallers[t_b.second],
                                        temperature, incoh_data.interp_law);
    double c = 2*inc_energy*debye_waller;
    double exp_neg_c = std::exp(-c);
    double sinh_c = (std::exp(c) - exp_neg_c) / 2;
    double numerator = 2*openmc::prn(seed)*sinh_c + exp_neg_c;
    out_angle = std::log(numerator)/c;
}

void ThermalScatteringOTF::sample_inelastic_collision(const double& inc_energy, const double& temperature, 
                                double& out_energy, double& out_angle,
                                uint64_t* seed){
    double sampled_beta = sample_beta(inc_energy, temperature, openmc::prn(seed));
    double sampled_alpha = sample_alpha(inc_energy, temperature, sampled_beta, openmc::prn(seed));
    out_energy = calculate_scattering_energy(inc_energy, temperature, sampled_beta);
    out_angle = calculate_scattering_angle(inc_energy, out_energy, temperature, sampled_alpha);
}

double ThermalScatteringOTF::sample_beta(const double& inc_energy, const double& temperature, const double& random){
    const Inelastic_Data& inel_data = data.inelastic.value();
    const Inelastic_Fit_2D& beta_data = inel_data.beta;
    double eval_point = scale_value(temperature, inel_data.min_t, inel_data.max_t, beta_data.min_scale, beta_data.max_scale);
    std::vector<double> evaled_basis_points = beta_data.fit_function(eval_point, beta_data.num_coeffs-1);
    InterpolationIndices e_b = findSampleInterpolationIndices(beta_data.x.begin(), beta_data.x.end()-1,  inc_energy);
    InterpolationIndices c_b = findSampleInterpolationIndices(beta_data.y.begin(), beta_data.y.end()-1,  random);
    vec_iter f11_i = beta_data.coeffs.begin() + beta_data.num_coeffs*(e_b.first*beta_data.y.size() + c_b.first);
    vec_iter f12_i = beta_data.coeffs.begin() + beta_data.num_coeffs*(e_b.first*beta_data.y.size() + c_b.second);
    vec_iter f21_i = beta_data.coeffs.begin() + beta_data.num_coeffs*(e_b.second*beta_data.y.size() + c_b.first);
    vec_iter f22_i = beta_data.coeffs.begin() + beta_data.num_coeffs*(e_b.second*beta_data.y.size() + c_b.second);
    return bi_interp(
        beta_data.x[e_b.first], //x1
        beta_data.x[e_b.second],//x2
        beta_data.y[c_b.first], //y1
        beta_data.y[c_b.second],//y2
        std::inner_product(f11_i, f11_i+beta_data.num_coeffs, evaled_basis_points.begin(), 0.0), //f11
        std::inner_product(f12_i, f12_i+beta_data.num_coeffs, evaled_basis_points.begin(), 0.0), //f12
        std::inner_product(f21_i, f21_i+beta_data.num_coeffs, evaled_basis_points.begin(), 0.0), //f21
        std::inner_product(f22_i, f22_i+beta_data.num_coeffs, evaled_basis_points.begin(), 0.0), //f22
        // inner_product(f11_i, evaled_basis_points.begin(), beta_data.num_coeffs), //f11
        // inner_product(f12_i, evaled_basis_points.begin(), beta_data.num_coeffs), //f12
        // inner_product(f21_i, evaled_basis_points.begin(), beta_data.num_coeffs), //f21
        // inner_product(f22_i, evaled_basis_points.begin(), beta_data.num_coeffs), //f22
        inc_energy, //x
        random,     //y
        2, //x-interp scheme
        2  //y-interp scheme
    );
}

double ThermalScatteringOTF::sample_alpha(const double& inc_ener, const double& temperature, const double& beta, const double& random){
    const Inelastic_Data& inel_data = data.inelastic.value();
    const Inelastic_Fit_2D& alpha_data = inel_data.alpha;
    double eval_point = scale_value(temperature, inel_data.min_t, inel_data.max_t, alpha_data.min_scale, alpha_data.max_scale);
    std::vector<double> evaled_basis_points = alpha_data.fit_function(eval_point, alpha_data.num_coeffs-1);
    Alpha_Extrema alpha_extrema = calculate_alpha_extrema(inc_ener, temperature, beta);
    /// NOTE: Alpha data is stored using the original beta grid provided to LEAPR and is unscaled.
    /// NOTE: In addition, stored alpha CDFs are symmetric about beta=0 so only need the positive half.
    double grid_beta = std::abs(beta*temperature/ref_temp_k);
    InterpolationIndices b_b = findSampleInterpolationIndices(alpha_data.x.begin(), alpha_data.x.end()-1, grid_beta);
    /// NOTE: Due to the manner in which the OTF data is stored, large energy losses/gains fall off the stored beta grid.
    /// This causes large extrapolation errors which can lead to nonsense scattering angles.
    /// To avoid this, use the last stored beta value of the grid beta.
    if (grid_beta > alpha_data.x[b_b.second]){grid_beta = alpha_data.x[b_b.second];}
    double l_alpha = sample_bounding_alpha(b_b.first, alpha_extrema, random, evaled_basis_points);
    double u_alpha = sample_bounding_alpha(b_b.second, alpha_extrema, random, evaled_basis_points);
    double alpha = ENDF_interp(alpha_data.x[b_b.first], alpha_data.x[b_b.second],
                                l_alpha, u_alpha, grid_beta, 2);
    return alpha;
}

double ThermalScatteringOTF::sample_bounding_alpha(const int& beta_ind, const Alpha_Extrema& alpha_extrema, const double& random, const std::vector<double>& evaled_basis_points){
    const Inelastic_Fit_2D& alpha_data = data.inelastic.value().alpha;
    double random_prime = rescale_alpha_random_number(beta_ind, alpha_extrema, evaled_basis_points, random);
    InterpolationIndices c_b = findSampleInterpolationIndices(alpha_data.y.begin(), alpha_data.y.end()-1, random_prime);
    vec_iter a_l_iter = alpha_data.coeffs.begin() + alpha_data.num_coeffs*(beta_ind*alpha_data.y.size()+c_b.first);
    vec_iter a_u_iter = alpha_data.coeffs.begin() + alpha_data.num_coeffs*(beta_ind*alpha_data.y.size()+c_b.second);
    double alpha_l = std::inner_product(a_l_iter, a_l_iter+alpha_data.num_coeffs, evaled_basis_points.begin(), 0.0);
    double alpha_u = std::inner_product(a_u_iter, a_u_iter+alpha_data.num_coeffs, evaled_basis_points.begin(), 0.0);
    // double alpha_l = inner_product(a_l_iter, evaled_basis_points.begin(), alpha_data.num_coeffs);
    // double alpha_u = inner_product(a_u_iter, evaled_basis_points.begin(), alpha_data.num_coeffs);
    double alpha = ENDF_interp(alpha_data.y[c_b.first], alpha_data.y[c_b.second], alpha_l, alpha_u, random_prime, 2);
    return alpha;
}

double ThermalScatteringOTF::rescale_alpha_random_number(const int& beta_ind, const Alpha_Extrema& alpha_extrema, const std::vector<double>& evaled_basis_points, const double& random){
    const Inelastic_Fit_2D& alpha_data = data.inelastic.value().alpha;
    vec_iter alpha_start = alpha_data.coeffs.begin() + beta_ind*alpha_data.y.size()*alpha_data.num_coeffs; // Point to the beginning of the first set of coefficients
    vec_iter alpha_end = alpha_start + alpha_data.y.size()*alpha_data.num_coeffs - alpha_data.num_coeffs; // Point to the beginning of the last set of coefficients
    double alpha_min_cdf = reverse_search_alpha_cdf(beta_ind, alpha_start, alpha_end, alpha_extrema.minimum, evaled_basis_points);
    double alpha_max_cdf = reverse_search_alpha_cdf(beta_ind, alpha_start, alpha_end, alpha_extrema.maximum, evaled_basis_points);
    double random_prime = scale_value(random, 0, 1, alpha_min_cdf, alpha_max_cdf);
    return random_prime;
}

double ThermalScatteringOTF::reverse_search_alpha_cdf(const int& beta_ind, const vec_iter& alpha_start, const vec_iter& alpha_end, const double& alpha_search, const std::vector<double>& evaled_basis_points){
    const Inelastic_Fit_2D& alpha_data = data.inelastic.value().alpha;
    InterpolationIndices a_b = findSampleCoeffInterpolationIndices(alpha_start, alpha_end, alpha_search, evaled_basis_points);
    vec_iter a_l_iter = alpha_data.coeffs.begin() + alpha_data.num_coeffs*(beta_ind*alpha_data.y.size() + a_b.first);
    vec_iter a_u_iter = alpha_data.coeffs.begin() + alpha_data.num_coeffs*(beta_ind*alpha_data.y.size() + a_b.second);
    double a_l = std::inner_product(a_l_iter, a_l_iter + alpha_data.num_coeffs, evaled_basis_points.begin(), 0.0);
    double a_u = std::inner_product(a_u_iter, a_u_iter + alpha_data.num_coeffs, evaled_basis_points.begin(), 0.0);
    // double a_l = inner_product(a_l_iter, evaled_basis_points.begin(), alpha_data.num_coeffs);
    // double a_u = inner_product(a_u_iter, evaled_basis_points.begin(), alpha_data.num_coeffs);
    double cdf = ENDF_interp(a_l, a_u, alpha_data.y[a_b.first], alpha_data.y[a_b.second], alpha_search, 2);
    return cdf;
}

ThermalScatteringOTF::Alpha_Extrema ThermalScatteringOTF::calculate_alpha_extrema(const double& inc_energy, const double& temperature, const double& beta){
    Alpha_Extrema extrema;
    double t1 = std::sqrt(inc_energy);
    double t2 = data.A0*boltz*temperature;
    double t3 = std::sqrt(std::abs(inc_energy + beta*boltz*temperature));
    double t4 = t1 - t3;
    double t5 = t1 + t3;
    extrema.minimum = (t4*t4)/t2;
    extrema.maximum = (t5*t5)/t2;
    return extrema;
}

double ThermalScatteringOTF::calculate_scattering_energy(const double& inc_energy, const double& temperature, const double& beta){
    return inc_energy + beta*boltz*temperature;
}

double ThermalScatteringOTF::calculate_scattering_angle(const double inc_energy, const double& scat_energy, const double& temperature, const double& alpha){
    return ((inc_energy + scat_energy) - alpha*data.A0*boltz*temperature)/(2*std::sqrt(inc_energy*scat_energy));
}