#include "openmc/thermal.h"

#include <algorithm> // for sort, move, min, max, find
#include <cmath>     // for round, sqrt, abs
#include <numeric>   // for innerproduct

#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include <fmt/core.h>

#include "openmc/constants.h"
#include "openmc/endf.h"
#include "openmc/error.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"
#include "openmc/secondary_correlated.h"
#include "openmc/secondary_thermal.h"
#include "openmc/settings.h"
#include "openmc/string_utils.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace data {
std::unordered_map<std::string, int> thermal_scatt_map;
vector<unique_ptr<ThermalScattering>> thermal_scatt;
} // namespace data

//==============================================================================
// Support Functions for OTF treatment
//==============================================================================

// Eval basis Functions

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

typedef std::pair<size_t, size_t> InterpolationIndices;

/**
 * @brief Finds the indices of list of elements that would bracket the search value val
 * @tparam Iter Constant iterator
 * @param begin Iterator pointing to the first element of the search range
 * @param end Iterator pointing to the last element of the search range
 * If you use x.end(), you need to subtract one from the iterator
 * @param val Double of the value that is desired to be found
 * @return InterpolationIndicies type (std::pair(size_t, size_t))
 */
template <typename Iter>
InterpolationIndices findSampleInterpolationIndices(const Iter &begin, const Iter &end, const double &val) 
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
 * @tparam Iter Constant iterator
 * @param begin1 Iterator pointing to the first element of the search range for the lower list
 * @param begin2 Iterator pointing to the first element of the search range for the upper list
 * @param end Iterator pointing to the last element of the search range
 * If you use x.end(), you need to subtract one from the iterator
 * @param val Double of the value that is desired to be found
 * @param scheme Interpolation scheme
 * @return InterpolationIndicies type (std::pair(size_t, size_t))
 */
template <typename Iter>
InterpolationIndices findSampleInterpolationIndices(const Iter &begin1, const Iter &begin2, const Iter &end, const double& x1, const double& x2, const double& x, const double &val, const int scheme) 
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
 * @tparam Iter Constant iterator
 * @param begin Iterator pointing to the first coefficient in the first set of coefficients of the search range
 * @param end Iterator pointing to the first coefficient in the last set of coefficients of the search range
 * See test_sample_search.cpp to see how to set begin and end properly
 * @param val Double of the value that is desired to be found\
 * @param evaled_basis_points Evaluated f(x) values of the basis functions at the desired x value.  
 * This should be the returned vector from an Evaluation Function
 * @return InterpolationIndicies type (std::pair(size_t, size_t))
 */
template <typename Iter>
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

typedef std::vector<double>(*FuncPointer)(const double &, const int &);
void set_fit_func__(FuncPointer& fitting_function, const std::string& fit_func_str)
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

// Constants 
constexpr double boltz = 0.00008617385;
constexpr double ref_temp_ev = 0.0253;
constexpr double ref_temp_k = ref_temp_ev/boltz;

//==============================================================================
// ThermalScattering implementation
//==============================================================================

ThermalScattering::ThermalScattering(
  hid_t group, const vector<double>& temperature)
{
  if (attribute_exists(group, "OTF")){
    is_otf_ = true;
  }
  // Get name of table from group
  name_ = object_name(group);

  // Get rid of leading '/'
  name_ = name_.substr(1);

  read_attribute(group, "atomic_weight_ratio", awr_);
  read_attribute(group, "energy_max", energy_max_);
  read_attribute(group, "nuclides", nuclides_);

  if (is_otf_)
  {
    // Read in Elastic Data
    if (object_exists(group, "Elastic")){
      hid_t elastic_group = open_group(group, "Elastic");
      // Read in Coherent Data
      if (object_exists(elastic_group, "Coherent")){
        hid_t coherent_group = open_group(elastic_group, "Coherent");
        otf_has_elastic_coherent = true;
        read_dataset(coherent_group, "ENERGIES", otf_elastic_coherent_energies);
        read_dataset(coherent_group, "INTERPOLATION_LAWS", otf_elastic_coherent_interp_laws);
        read_dataset(coherent_group, "S_VALUES", otf_elastic_coherent_s_vals);
        read_dataset(coherent_group, "TEMPERATURES", otf_elastic_coherent_temperatures);
        close_group(coherent_group);
      }
      // Read in Incoherent Data
      if (object_exists(elastic_group, "Incoherent")){
        hid_t incoherent_group = open_group(elastic_group, "Incoherent");
        otf_has_elastic_incoherent = true;
        read_dataset(incoherent_group, "DEBYE_WALLERS", otf_elastic_incoherent_debye_wallers);
        read_dataset(incoherent_group, "TEMPERATURES", otf_elastic_incoherent_temperatures);
        read_int(incoherent_group, "INTERPOLATION_LAW", &otf_elastic_incoherent_interp_law, true);
        close_group(incoherent_group);
      }
      close_group(elastic_group);
    }

    // Read in Inelastic Data
    if (object_exists(group, "Inelastic")){
      hid_t inelastic_group = open_group(group, "Inelastic");
      otf_has_inelastic = true;
      read_double(inelastic_group, "@A0", &otf_inelastic_A0, true);
      read_double(inelastic_group, "@BOUND_XS", &otf_inelastic_bound_xs, true);
      read_double(inelastic_group, "@E_MAX", &otf_inelastic_e_max, true);
      read_double(inelastic_group, "@FREE_XS", &otf_inelastic_free_xs, true);
      read_double(inelastic_group, "@M0", &otf_inelastic_m0, true);
      read_double(inelastic_group, "@MAT", &otf_inelastic_mat, true);
      read_double(inelastic_group, "@MAX_T", &otf_inelastic_max_t, true);
      read_double(inelastic_group, "@MIN_T", &otf_inelastic_min_t, true);
      read_double(inelastic_group, "@ZA", &otf_inelastic_za, true);
      // Read in ALPHA Data
      if (object_exists(inelastic_group, "ALPHA")){
        hid_t alpha_group = open_group(inelastic_group, "ALPHA");
        read_attribute(alpha_group, "FITTING_FUNCTION", otf_inelastic_alpha_fitting_function_str);
        set_fit_func__(otf_inelastic_alpha_fitting_function, otf_inelastic_alpha_fitting_function_str);
        read_dataset(alpha_group, "BETA_GRID", otf_inelastic_alpha_beta_grid);
        read_dataset(alpha_group, "CDF_GRID", otf_inelastic_alpha_cdf_grid);
        read_dataset(alpha_group, "COEFFS", otf_inelastic_alpha_coeffs);
        num_alpha_coeffs = otf_inelastic_alpha_coeffs.size() / (otf_inelastic_alpha_beta_grid.size() * otf_inelastic_alpha_cdf_grid.size());
        read_double(alpha_group, "MAX_SCALE", &otf_inelastic_alpha_max_scale, true);
        read_double(alpha_group, "MIN_SCALE", &otf_inelastic_alpha_min_scale, true);
        close_group(alpha_group);
      }
      // Read in BETA Data
      if (object_exists(inelastic_group, "BETA")){
        hid_t beta_group = open_group(inelastic_group, "BETA");
        read_attribute(beta_group, "FITTING_FUNCTION", otf_inelastic_beta_fitting_function_str);
        set_fit_func__(otf_inelastic_beta_fitting_function, otf_inelastic_beta_fitting_function_str);
        read_dataset(beta_group, "ENERGY_GRID", otf_inelastic_beta_energy_grid);
        read_dataset(beta_group, "CDF_GRID", otf_inelastic_beta_cdf_grid);
        read_dataset(beta_group, "COEFFS", otf_inelastic_beta_coeffs);
        num_beta_coeffs = otf_inelastic_beta_coeffs.size() / (otf_inelastic_beta_energy_grid.size() * otf_inelastic_beta_cdf_grid.size());
        read_double(beta_group, "MAX_SCALE", &otf_inelastic_beta_max_scale, true);
        read_double(beta_group, "MIN_SCALE", &otf_inelastic_beta_min_scale, true);
        close_group(beta_group);
      }
      // Read in XS data
      if (object_exists(inelastic_group, "XS")){
        hid_t xs_group = open_group(inelastic_group, "XS");
        read_attribute(xs_group, "FITTING_FUNCTION", otf_inelastic_xs_fitting_function_str);
        set_fit_func__(otf_inelastic_xs_fitting_function, otf_inelastic_xs_fitting_function_str);
        read_dataset(xs_group, "ENERGY_GRID", otf_inelastic_xs_energy_grid);
        read_dataset(xs_group, "COEFFS", otf_inelastic_xs_coeffs);
        num_xs_coeffs = otf_inelastic_xs_coeffs.size() / otf_inelastic_xs_energy_grid.size();
        read_double(xs_group, "MAX_SCALE", &otf_inelastic_xs_max_scale, true);
        read_double(xs_group, "MIN_SCALE", &otf_inelastic_xs_min_scale, true);
        close_group(xs_group);
      }
      close_group(inelastic_group);
    }
  }

  else{
    // Read temperatures
    hid_t kT_group = open_group(group, "kTs");

    // Determine temperatures available
    auto dset_names = dataset_names(kT_group);
    auto n = dset_names.size();
    auto temps_available = xt::empty<double>({n});
    for (int i = 0; i < dset_names.size(); ++i) {
      // Read temperature value
      double T;
      read_dataset(kT_group, dset_names[i].data(), T);
      temps_available[i] = std::round(T / K_BOLTZMANN);
    }
    std::sort(temps_available.begin(), temps_available.end());

    // Determine actual temperatures to read -- start by checking whether a
    // temperature range was given, in which case all temperatures in the range
    // are loaded irrespective of what temperatures actually appear in the model
    vector<int> temps_to_read;
    if (settings::temperature_range[1] > 0.0) {
      for (const auto& T : temps_available) {
        if (settings::temperature_range[0] <= T &&
            T <= settings::temperature_range[1]) {
          temps_to_read.push_back(std::round(T));
        }
      }
    }

    switch (settings::temperature_method) {
    case TemperatureMethod::NEAREST:
      // Determine actual temperatures to read
      for (const auto& T : temperature) {

        auto i_closest = xt::argmin(xt::abs(temps_available - T))[0];
        auto temp_actual = temps_available[i_closest];
        if (std::abs(temp_actual - T) < settings::temperature_tolerance) {
          if (std::find(temps_to_read.begin(), temps_to_read.end(),
                std::round(temp_actual)) == temps_to_read.end()) {
            temps_to_read.push_back(std::round(temp_actual));
          }
        } else {
          fatal_error(fmt::format(
            "Nuclear data library does not contain cross sections "
            "for {}  at or near {} K. Available temperatures "
            "are {} K. Consider making use of openmc.Settings.temperature "
            "to specify how intermediate temperatures are treated.",
            name_, std::round(T), concatenate(temps_available)));
        }
      }
      break;

    case TemperatureMethod::INTERPOLATION:
      // If temperature interpolation or multipole is selected, get a list of
      // bounding temperatures for each actual temperature present in the model
      for (const auto& T : temperature) {
        bool found = false;
        for (int j = 0; j < temps_available.size() - 1; ++j) {
          if (temps_available[j] <= T && T < temps_available[j + 1]) {
            int T_j = temps_available[j];
            int T_j1 = temps_available[j + 1];
            if (std::find(temps_to_read.begin(), temps_to_read.end(), T_j) ==
                temps_to_read.end()) {
              temps_to_read.push_back(T_j);
            }
            if (std::find(temps_to_read.begin(), temps_to_read.end(), T_j1) ==
                temps_to_read.end()) {
              temps_to_read.push_back(T_j1);
            }
            found = true;
          }
        }
        if (!found) {
          // If no pairs found, check if the desired temperature falls within
          // bounds' tolerance
          if (std::abs(T - temps_available[0]) <=
              settings::temperature_tolerance) {
            if (std::find(temps_to_read.begin(), temps_to_read.end(),
                  temps_available[0]) == temps_to_read.end()) {
              temps_to_read.push_back(temps_available[0]);
            }
          } else if (std::abs(T - temps_available[n - 1]) <=
                    settings::temperature_tolerance) {
            if (std::find(temps_to_read.begin(), temps_to_read.end(),
                  temps_available[n - 1]) == temps_to_read.end()) {
              temps_to_read.push_back(temps_available[n - 1]);
            }
          } else {
            fatal_error(
              fmt::format("Nuclear data library does not contain cross "
                          "sections for {} at temperatures that bound {} K.",
                name_, std::round(T)));
          }
        }
      }
    }

    // Sort temperatures to read
    std::sort(temps_to_read.begin(), temps_to_read.end());

    auto n_temperature = temps_to_read.size();
    kTs_.reserve(n_temperature);
    data_.reserve(n_temperature);

    for (auto T : temps_to_read) {
      // Get temperature as a string
      std::string temp_str = fmt::format("{}K", T);

      // Read exact temperature value
      double kT;
      read_dataset(kT_group, temp_str.data(), kT);
      kTs_.push_back(kT);

      // Open group for this temperature
      hid_t T_group = open_group(group, temp_str.data());
      data_.emplace_back(T_group);
      close_group(T_group);
    }

    close_group(kT_group);
  }
}

void ThermalScattering::calculate_xs(double E, double sqrtkT, int* i_temp,
  double* elastic, double* inelastic, uint64_t* seed) const
{
  if (is_otf_)
  {
    double temperature = (sqrtkT * sqrtkT) / boltz;
    double elastic_xs = 0;
    double inelastic_xs = 0;
    // Add in coherent elastic xs
    if (otf_has_elastic_coherent){
        // If the incident energy is below the first bragg edge, the cross section is 0
        if (E >= otf_elastic_coherent_energies[0]){
            elastic_xs += calculate_elastic_coherent_xs(E, temperature);
        }
    }
    // Add in incoherent elastic xs
    if (otf_has_elastic_incoherent){
        elastic_xs += calculate_elastic_incoherent_xs(E, temperature);
    }
    // Add in inelastic xs#include <math.h>
    if (otf_has_inelastic){
        inelastic_xs += calculate_inelastic_xs(E, temperature);
    }
    *elastic = elastic_xs;
    *inelastic = inelastic_xs;
  }
  else{
    // Determine temperature for S(a,b) table
    double kT = sqrtkT * sqrtkT;
    int i = 0;

    auto n = kTs_.size();
    if (n > 1) {
      if (settings::temperature_method == TemperatureMethod::NEAREST) {
        while (kTs_[i + 1] < kT && i + 1 < n - 1)
          ++i;
        // Pick closer of two bounding temperatures
        if (kT - kTs_[i] > kTs_[i + 1] - kT)
          ++i;
      } else {
        // If current kT outside of the bounds of available, snap to the bound
        if (kT < kTs_.front()) {
          i = 0;
        } else if (kT > kTs_.back()) {
          i = kTs_.size() - 1;
        } else {
          // Find temperatures that bound the actual temperature
          while (kTs_[i + 1] < kT && i + 1 < n - 1)
            ++i;
          // Randomly sample between temperature i and i+1
          double f = (kT - kTs_[i]) / (kTs_[i + 1] - kTs_[i]);
          if (f > prn(seed))
            ++i;
        }
      }
    }

    // Set temperature index
    *i_temp = i;

    // Calculate cross sections for ith temperature
    data_[i].calculate_xs(E, elastic, inelastic);
  }
}

bool ThermalScattering::has_nuclide(const char* name) const
{
  std::cout << "Hello from has_nuclide" << std::endl;
  std::string nuc {name};
  return std::find(nuclides_.begin(), nuclides_.end(), nuc) != nuclides_.end();
}

// OTF thermal scattering functions

double ThermalScattering::calculate_elastic_coherent_xs(double E, double temperature) const{
    InterpolationIndices t_b = findSampleInterpolationIndices(otf_elastic_coherent_temperatures.begin(), 
                                                                      otf_elastic_coherent_temperatures.end() - 1, 
                                                                      temperature);
            InterpolationIndices e_b = findSampleInterpolationIndices(otf_elastic_coherent_energies.begin(),
                                                                      otf_elastic_coherent_energies.end() - 1,
                                                                      E);
            int interp_scheme = otf_elastic_coherent_interp_laws[t_b.first];
            double s_l = otf_elastic_coherent_s_vals[otf_elastic_coherent_energies.size()*t_b.first  + e_b.first];
            double s_u = otf_elastic_coherent_s_vals[otf_elastic_coherent_energies.size()*t_b.second + e_b.first];
            return ENDF_interp(otf_elastic_coherent_temperatures[t_b.first],
                             otf_elastic_coherent_temperatures[t_b.second],
                             s_l,
                             s_u,
                             temperature,
                             interp_scheme);
}

double ThermalScattering::calculate_elastic_incoherent_xs(double E, double temperature) const{
    InterpolationIndices t_b = findSampleInterpolationIndices(otf_elastic_incoherent_temperatures.begin(),
                                                                  otf_elastic_incoherent_temperatures.end() - 1,
                                                                  temperature);
        double debye_waller = ENDF_interp(otf_elastic_incoherent_temperatures[t_b.first],
                                          otf_elastic_incoherent_temperatures[t_b.second],
                                          otf_elastic_incoherent_debye_wallers[t_b.first],
                                          otf_elastic_incoherent_debye_wallers[t_b.second],
                                          temperature,
                                          otf_elastic_incoherent_interp_law);
        double ew2 = 2*E*debye_waller;
        return otf_inelastic_bound_xs/2*((1-exp(-2*ew2))/ew2);
}

double ThermalScattering::calculate_inelastic_xs(double E, double temperature) const{
    double eval_point = scale_value(temperature, otf_inelastic_min_t, otf_inelastic_max_t, otf_inelastic_xs_min_scale, otf_inelastic_xs_max_scale);
        int num_coeffs = otf_inelastic_xs_coeffs.size()/otf_inelastic_xs_energy_grid.size();
        std::vector<double> basis_points = otf_inelastic_xs_fitting_function(eval_point, num_coeffs);
        InterpolationIndices e_b = findSampleInterpolationIndices(otf_inelastic_xs_energy_grid.begin(),
                                                                  otf_inelastic_xs_energy_grid.end() - 1,
                                                                  E);
        
        double xs_l = std::inner_product(basis_points.begin(), basis_points.end(), otf_inelastic_xs_coeffs.begin() + e_b.first *num_coeffs, 0.0);
        double xs_u = std::inner_product(basis_points.begin(), basis_points.end(), otf_inelastic_xs_coeffs.begin() + e_b.second*num_coeffs, 0.0);
        return ENDF_interp(otf_inelastic_xs_energy_grid[e_b.first],
                         otf_inelastic_xs_energy_grid[e_b.second],
                         xs_l,
                         xs_u,
                         E,
                         2);
}

void ThermalScattering::sample_otf(double sqrtkT, double E_in, double* E_out, double* mu, uint64_t* seed)
{
    double temperature = (sqrtkT * sqrtkT) / boltz;
    // Determine what type of reaction to sample
    double elastic_coherent_xs = 0;
    double elastic_incoherent_xs = 0;
    double elastic_xs;
    double inelastic_xs = 0;
    if (otf_has_elastic_coherent){elastic_coherent_xs = calculate_elastic_coherent_xs(E_in, temperature);}
    if (otf_has_elastic_incoherent){elastic_incoherent_xs = calculate_elastic_incoherent_xs(E_in, temperature);}
    if (otf_has_inelastic){inelastic_xs = calculate_inelastic_xs(E_in, temperature);}
    elastic_xs = elastic_coherent_xs + elastic_incoherent_xs;
    // Elastic vs Inelastic : Elastic Choosen
    if (prn(seed) * (inelastic_xs + elastic_xs) < elastic_xs){
        // Coherent vs Incoherent : Coherent Choosen
        if(prn(seed) * (elastic_coherent_xs + elastic_incoherent_xs) < elastic_coherent_xs){
            sample_otf_elastic_coherent(temperature, E_in, E_out, mu, seed);
        }
        // Coherent vs Incoherent : Incoherent Choosen
        else{
            sample_otf_elastic_incoherent(temperature, E_in, E_out, mu, seed);
        }
    }
    // Elastic vs Inelastic : Inelastic Choosen
    else{
        sample_otf_inelastic(temperature, E_in, E_out, mu, seed);
    }
}

  void ThermalScattering::sample_otf_elastic_coherent(double temperature, double E_in, double* E_out, double* mu, uint64_t* seed) const
  {
    // No energy change
    *E_out = E_in;
    double sampled_bragg = 0;
    if (E_in < otf_elastic_coherent_energies[1]){
        sampled_bragg = otf_elastic_coherent_energies[0];
    }
    else{
        InterpolationIndices e_b = findSampleInterpolationIndices(otf_elastic_coherent_energies.begin(),
                                                                  otf_elastic_coherent_energies.end() - 1,
                                                                  E_in);
        InterpolationIndices t_b = findSampleInterpolationIndices(otf_elastic_coherent_temperatures.begin(),
                                                                  otf_elastic_coherent_temperatures.end() - 1,
                                                                  temperature);
        double s_search = prn(seed) * ENDF_interp(otf_elastic_coherent_temperatures[t_b.first],
                                      otf_elastic_coherent_temperatures[t_b.second],
                                      otf_elastic_coherent_s_vals[otf_elastic_coherent_temperatures.size()*t_b.first  + e_b.first],
                                      otf_elastic_coherent_s_vals[otf_elastic_coherent_temperatures.size()*t_b.second + e_b.first],
                                      temperature,
                                      otf_elastic_coherent_interp_laws[t_b.first]);
        InterpolationIndices s_b = findSampleInterpolationIndices(otf_elastic_coherent_s_vals.begin() + otf_elastic_coherent_temperatures.size()*t_b.first,
                                                         otf_elastic_coherent_s_vals.begin() + otf_elastic_coherent_temperatures.size()*t_b.second,
                                                         otf_elastic_coherent_s_vals.begin() + otf_elastic_coherent_temperatures.size()*t_b.first + otf_elastic_coherent_energies.size(),
                                                         otf_elastic_coherent_temperatures[t_b.first],
                                                         otf_elastic_coherent_temperatures[t_b.second],
                                                         temperature,
                                                         s_search,
                                                         otf_elastic_coherent_interp_laws[t_b.first]);
        sampled_bragg = otf_elastic_coherent_energies[s_b.first];
    }
    *mu = 1 - ((2*sampled_bragg)/E_in);
  }

  void ThermalScattering::sample_otf_elastic_incoherent(double temperature, double E_in, double* E_out, double* mu, uint64_t* seed) const
  {
    *E_out = E_in;
    InterpolationIndices t_b = findSampleInterpolationIndices(otf_elastic_incoherent_temperatures.begin(),
                                                                  otf_elastic_incoherent_temperatures.end() - 1,
                                                                  temperature);
    double debye_waller = ENDF_interp(otf_elastic_incoherent_temperatures[t_b.first],
                                        otf_elastic_incoherent_temperatures[t_b.second],
                                        otf_elastic_incoherent_debye_wallers[t_b.first],
                                        otf_elastic_incoherent_debye_wallers[t_b.second],
                                        temperature,
                                        otf_elastic_incoherent_interp_law);
    double c = 2*E_in*debye_waller;
    double exp_neg_c = std::exp(-c);
    double sinh_c = (std::exp(c) - exp_neg_c) / 2;
    double numerator = 2*prn(seed)*sinh_c + exp_neg_c;
    *mu = std::log(numerator)/c;
  }

  void ThermalScattering::sample_otf_inelastic(double temperature, double E_in, double* E_out, double* mu, uint64_t* seed) const
  {
    double sampled_beta = sample_beta__(temperature, E_in, prn(seed));
    if (sampled_beta > 20){sampled_beta = 20;} // OTF TODO: Get rid of this.
    double sampled_scattering_energy = calculate_secondary_energy__(temperature, E_in, sampled_beta);
    double sampled_alpha = sample_alpha__(temperature, E_in, sampled_beta, prn(seed));
    double sampled_scattering_cosine = calculate_scattering_cosine__(temperature, E_in, sampled_scattering_energy, sampled_alpha);
    if (std::abs(sampled_scattering_cosine) > 1.0){sampled_scattering_cosine = std::copysign(1.0, sampled_scattering_cosine);} // OTF TODO: Get rid of this.
    // std::cout << "Incident Energy           " << E_in << std::endl;
    // std::cout << "Sampled Scattering Energy " << sampled_scattering_energy << std::endl;
    // std::cout << "Sampled Scattering Cosine " << sampled_scattering_cosine << std::endl;
    // std::cout << std::endl;
    *E_out = sampled_scattering_energy;
    *mu = sampled_scattering_cosine;
    // *mu = prn(seed)*2 - 1;
    // *E_out = E_in;
    // *mu = 0;
  }

  double ThermalScattering::sample_beta__(const double &temp, const double &inc_ener, const double &xi) const
    {
    double eval_point = scale_value(temp, otf_inelastic_min_t, otf_inelastic_max_t, otf_inelastic_beta_min_scale, otf_inelastic_beta_max_scale);
    std::vector<double> evaled_basis_points = otf_inelastic_beta_fitting_function(eval_point, num_beta_coeffs - 1);
    auto [lo_inc_ener_ind, hi_inc_ener_ind] = findSampleInterpolationIndices(otf_inelastic_beta_energy_grid.begin(), otf_inelastic_beta_energy_grid.end() - 1, inc_ener);
    auto [lo_beta_cdf_ind, hi_beta_cdf_ind] = findSampleInterpolationIndices(otf_inelastic_beta_cdf_grid.begin(), otf_inelastic_beta_cdf_grid.end() - 1, xi);
    std::vector<double>::const_iterator f11_i = otf_inelastic_beta_coeffs.begin() + num_beta_coeffs*(lo_inc_ener_ind*otf_inelastic_beta_cdf_grid.size() + lo_beta_cdf_ind);
    std::vector<double>::const_iterator f12_i = otf_inelastic_beta_coeffs.begin() + num_beta_coeffs*(lo_inc_ener_ind*otf_inelastic_beta_cdf_grid.size() + hi_beta_cdf_ind);
    std::vector<double>::const_iterator f21_i = otf_inelastic_beta_coeffs.begin() + num_beta_coeffs*(hi_inc_ener_ind*otf_inelastic_beta_cdf_grid.size() + lo_beta_cdf_ind);
    std::vector<double>::const_iterator f22_i = otf_inelastic_beta_coeffs.begin() + num_beta_coeffs*(hi_inc_ener_ind*otf_inelastic_beta_cdf_grid.size() + hi_beta_cdf_ind);
    return bi_interp(otf_inelastic_beta_energy_grid[lo_inc_ener_ind], // x1
                     otf_inelastic_beta_energy_grid[hi_inc_ener_ind], // x2
                     otf_inelastic_beta_cdf_grid[lo_beta_cdf_ind], // y1
                     otf_inelastic_beta_cdf_grid[hi_beta_cdf_ind], // y2
                     std::inner_product(f11_i, f11_i + num_beta_coeffs, evaled_basis_points.begin(), 0.0), //f11
                     std::inner_product(f12_i, f12_i + num_beta_coeffs, evaled_basis_points.begin(), 0.0), //f12
                     std::inner_product(f21_i, f21_i + num_beta_coeffs, evaled_basis_points.begin(), 0.0), //f21
                     std::inner_product(f22_i, f22_i + num_beta_coeffs, evaled_basis_points.begin(), 0.0), //f22
                     inc_ener, // x
                     xi, // y
                     2, // x-interp scheme
                     2); // y-interp scheme
}

double ThermalScattering::calculate_secondary_energy__(const double &temp, const double &inc_ener, const double &beta) const
{
    return inc_ener + beta*boltz*temp;
}

std::pair<double, double> ThermalScattering::return_alpha_extrema__(const double & temp, const double &inc_ener, const double &beta) const
{
    double t1 = std::sqrt(inc_ener);
    double t2 = otf_inelastic_A0*boltz*temp;
    double t3 = std::sqrt(std::abs(inc_ener + beta*boltz*temp));
    double t4 = t1 - t3;
    double t5 = t1 + t3;
    return std::pair<double, double>((t4*t4)/t2,(t5*t5)/t2);
}

double ThermalScattering::sample_alpha__(const double &temp, const double &inc_ener, const double &beta, const double &xi) const
{
    // std::cout << "Entering sample_alpha__ with temp: " << temp 
            //   << ", inc_ener: " << inc_ener 
            //   << ", beta: " << beta 
            //   << ", xi: " << xi << std::endl;

    double eval_point = scale_value(temp, otf_inelastic_min_t, otf_inelastic_max_t, otf_inelastic_alpha_min_scale, otf_inelastic_alpha_max_scale);
    // std::cout << "Calculated eval_point: " << eval_point << std::endl;

    std::vector<double> evaled_alpha_points = otf_inelastic_alpha_fitting_function(eval_point, num_alpha_coeffs - 1);
    // std::cout << "Evaluated alpha points: ";
    // for (const auto &point : evaled_alpha_points) std::cout << point << " ";
    // std::cout << std::endl;

    std::pair<double, double> alpha_extrema = return_alpha_extrema__(temp, inc_ener, beta);
    // std::cout << "Alpha extrema: [" << alpha_extrema.first << ", " << alpha_extrema.second << "]" << std::endl;

    double grid_beta = std::abs(beta * temp / ref_temp_k);
    // std::cout << "Grid beta: " << grid_beta << std::endl;

    auto [lo_beta_ind, hi_beta_ind] = findSampleInterpolationIndices(otf_inelastic_alpha_beta_grid.begin(), otf_inelastic_alpha_beta_grid.end() - 1, grid_beta);
    // std::cout << "Beta indices: lo=" << lo_beta_ind << ", hi=" << hi_beta_ind << std::endl;

    double l_alpha = sample_bounding_alpha__(temp, lo_beta_ind, alpha_extrema, xi, evaled_alpha_points);
    double u_alpha = sample_bounding_alpha__(temp, hi_beta_ind, alpha_extrema, xi, evaled_alpha_points);
    // std::cout << "Sampled bounding alphas: l_alpha=" << l_alpha << ", u_alpha=" << u_alpha << std::endl;

    // std::cout << "x1 : " << otf_inelastic_alpha_beta_grid[lo_beta_ind] << std::endl;
    // std::cout << "x2 : " << otf_inelastic_alpha_beta_grid[hi_beta_ind] << std::endl;
    double alpha = ENDF_interp(otf_inelastic_alpha_beta_grid[lo_beta_ind],
                               otf_inelastic_alpha_beta_grid[hi_beta_ind],
                               l_alpha, 
                               u_alpha, 
                               grid_beta, 
                               2);
    // std::cout << "Interpolated alpha: " << alpha << std::endl;
    return alpha;
}

double ThermalScattering::sample_bounding_alpha__(const double &temp, const int &beta_ind, const std::pair<double, double> &alpha_extrema, const double &xi, const std::vector<double> &evaled_basis_points) const
{
    // std::cout << "Entering sample_bounding_alpha__ with temp: " << temp 
            //   << ", beta_ind: " << beta_ind 
            //   << ", xi: " << xi << std::endl;

    std::vector<double>::const_iterator alpha_start = otf_inelastic_alpha_coeffs.begin() + beta_ind * otf_inelastic_alpha_cdf_grid.size() * num_alpha_coeffs;
    std::vector<double>::const_iterator alpha_end = alpha_start + otf_inelastic_alpha_cdf_grid.size() * num_alpha_coeffs;

    auto [l_amin_ind, u_amin_ind] = findSampleCoeffInterpolationIndices(alpha_start, alpha_end - num_alpha_coeffs, alpha_extrema.first, evaled_basis_points);
    // std::cout << "Alpha min indices: l=" << l_amin_ind << ", u=" << u_amin_ind << std::endl;

    std::vector<double>::const_iterator x1_i = otf_inelastic_alpha_coeffs.begin() + num_alpha_coeffs * (beta_ind * otf_inelastic_alpha_cdf_grid.size() + l_amin_ind);
    std::vector<double>::const_iterator x2_i = otf_inelastic_alpha_coeffs.begin() + num_alpha_coeffs * (beta_ind * otf_inelastic_alpha_cdf_grid.size() + u_amin_ind);

    double x1 = std::inner_product(x1_i, x1_i + num_alpha_coeffs, evaled_basis_points.begin(), 0.0);
    double x2 = std::inner_product(x2_i, x2_i + num_alpha_coeffs, evaled_basis_points.begin(), 0.0);
    // std::cout << "X1: " << x1 << ", X2: " << x2 << std::endl;

    double u_amin_cdf = ENDF_interp(x1, x2, otf_inelastic_alpha_cdf_grid[l_amin_ind], otf_inelastic_alpha_cdf_grid[u_amin_ind], alpha_extrema.first, 2);
    // std::cout << "u_amin_cdf: " << u_amin_cdf << std::endl;

    auto [l_amax_ind, u_amax_ind] = findSampleCoeffInterpolationIndices(alpha_start, alpha_end - num_alpha_coeffs, alpha_extrema.second, evaled_basis_points);
    x1_i = otf_inelastic_alpha_coeffs.begin() + num_alpha_coeffs * (beta_ind * otf_inelastic_alpha_cdf_grid.size() + l_amax_ind);
    x2_i = otf_inelastic_alpha_coeffs.begin() + num_alpha_coeffs * (beta_ind * otf_inelastic_alpha_cdf_grid.size() + u_amax_ind);

    x1 = std::inner_product(x1_i, x1_i + num_alpha_coeffs, evaled_basis_points.begin(), 0.0);
    x2 = std::inner_product(x2_i, x2_i + num_alpha_coeffs, evaled_basis_points.begin(), 0.0);
    // std::cout << "X1: " << x1 << ", X2: " << x2 << std::endl;

    double u_amax_cdf = ENDF_interp(x1, x2, otf_inelastic_alpha_cdf_grid[l_amax_ind], otf_inelastic_alpha_cdf_grid[u_amax_ind], alpha_extrema.second, 2);
    // std::cout << "u_amax_cdf: " << u_amax_cdf << std::endl;

    double xi_prime = scale_value(xi, 0, 1, u_amin_cdf, u_amax_cdf);
    // std::cout << "Scaled xi_prime: " << xi_prime << std::endl;

    auto [l_alpha_cdf_ind, u_alpha_cdf_ind] = findSampleInterpolationIndices(otf_inelastic_alpha_cdf_grid.begin(), otf_inelastic_alpha_cdf_grid.end(), xi_prime);
    x1_i = otf_inelastic_alpha_coeffs.begin() + num_alpha_coeffs * (beta_ind * otf_inelastic_alpha_cdf_grid.size() + l_alpha_cdf_ind);
    x2_i = otf_inelastic_alpha_coeffs.begin() + num_alpha_coeffs * (beta_ind * otf_inelastic_alpha_cdf_grid.size() + u_alpha_cdf_ind);

    x1 = std::inner_product(x1_i, x1_i + num_alpha_coeffs, evaled_basis_points.begin(), 0.0);
    x2 = std::inner_product(x2_i, x2_i + num_alpha_coeffs, evaled_basis_points.begin(), 0.0);
    // std::cout << "X1: " << x1 << ", X2: " << x2 << std::endl;

    double alpha = ENDF_interp(otf_inelastic_alpha_cdf_grid[l_alpha_cdf_ind], otf_inelastic_alpha_cdf_grid[u_alpha_cdf_ind], x1, x2, xi_prime, 2);
    // std::cout << "Interpolated alpha: " << alpha << std::endl;
    return alpha;
}

double ThermalScattering::calculate_scattering_cosine__(const double &temp, const double &inc_ener, const double &sec_ener, const double &alpha) const
{
    // std::cout << "Entering calculate_scattering_cosine__ with temp: " << temp 
            //   << ", inc_ener: " << inc_ener 
            //   << ", sec_ener: " << sec_ener 
            //   << ", alpha: " << alpha << std::endl;

    double result = ((inc_ener + sec_ener) - alpha * otf_inelastic_A0 * boltz * temp) / (2 * sqrt(inc_ener * sec_ener));
    // std::cout << "Calculated scattering cosine: " << result << std::endl;
    return result;
}

//==============================================================================
// ThermalData implementation
//==============================================================================

ThermalData::ThermalData(hid_t group)
{
  // Coherent/incoherent elastic data
  if (object_exists(group, "elastic")) {
    // Read cross section data
    hid_t elastic_group = open_group(group, "elastic");

    // Read elastic cross section
    elastic_.xs = read_function(elastic_group, "xs");

    // Read angle-energy distribution
    hid_t dgroup = open_group(elastic_group, "distribution");
    std::string temp;
    read_attribute(dgroup, "type", temp);
    if (temp == "coherent_elastic") {
      auto xs = dynamic_cast<CoherentElasticXS*>(elastic_.xs.get());
      elastic_.distribution = make_unique<CoherentElasticAE>(*xs);
    } else if (temp == "incoherent_elastic") {
      elastic_.distribution = make_unique<IncoherentElasticAE>(dgroup);
    } else if (temp == "incoherent_elastic_discrete") {
      auto xs = dynamic_cast<Tabulated1D*>(elastic_.xs.get());
      elastic_.distribution =
        make_unique<IncoherentElasticAEDiscrete>(dgroup, xs->x());
    } else if (temp == "mixed_elastic") {
      // Get coherent/incoherent cross sections
      auto mixed_xs = dynamic_cast<Sum1D*>(elastic_.xs.get());
      const auto& coh_xs =
        dynamic_cast<const CoherentElasticXS*>(mixed_xs->functions(0).get());
      const auto& incoh_xs = mixed_xs->functions(1).get();

      // Create mixed elastic distribution
      elastic_.distribution =
        make_unique<MixedElasticAE>(dgroup, *coh_xs, *incoh_xs);
    }

    close_group(elastic_group);
  }

  // Inelastic data
  if (object_exists(group, "inelastic")) {
    // Read type of inelastic data
    hid_t inelastic_group = open_group(group, "inelastic");

    // Read inelastic cross section
    inelastic_.xs = read_function(inelastic_group, "xs");

    // Read angle-energy distribution
    hid_t dgroup = open_group(inelastic_group, "distribution");
    std::string temp;
    read_attribute(dgroup, "type", temp);
    if (temp == "incoherent_inelastic") {
      inelastic_.distribution = make_unique<IncoherentInelasticAE>(dgroup);
    } else if (temp == "incoherent_inelastic_discrete") {
      auto xs = dynamic_cast<Tabulated1D*>(inelastic_.xs.get());
      inelastic_.distribution =
        make_unique<IncoherentInelasticAEDiscrete>(dgroup, xs->x());
    }

    close_group(inelastic_group);
  }
}

void ThermalData::calculate_xs(
  double E, double* elastic, double* inelastic) const
{
  // Calculate thermal elastic scattering cross section
  if (elastic_.xs) {
    *elastic = (*elastic_.xs)(E);
  } else {
    *elastic = 0.0;
  }

  // Calculate thermal inelastic scattering cross section
  *inelastic = (*inelastic_.xs)(E);
}

void ThermalData::sample(const NuclideMicroXS& micro_xs, double E,
  double* E_out, double* mu, uint64_t* seed)
{
  // Determine whether inelastic or elastic scattering will occur
  if (prn(seed) < micro_xs.thermal_elastic / micro_xs.thermal) {
    elastic_.distribution->sample(E, *E_out, *mu, seed);
  } else {
    inelastic_.distribution->sample(E, *E_out, *mu, seed);
  }

  // Because of floating-point roundoff, it may be possible for mu to be
  // outside of the range [-1,1). In these cases, we just set mu to exactly
  // -1 or 1
  if (std::abs(*mu) > 1.0)
    *mu = std::copysign(1.0, *mu);
}

void free_memory_thermal()
{
  data::thermal_scatt.clear();
  data::thermal_scatt_map.clear();
}

} // namespace openmc
