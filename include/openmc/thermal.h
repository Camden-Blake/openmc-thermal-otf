#ifndef OPENMC_THERMAL_H
#define OPENMC_THERMAL_H

#include <cstddef>
#include <string>
#include <unordered_map>

#include "xtensor/xtensor.hpp"

#include "openmc/angle_energy.h"
#include "openmc/endf.h"
#include "openmc/hdf5_interface.h"
#include "openmc/memory.h"
#include "openmc/particle.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

class ThermalScattering;

namespace data {
extern std::unordered_map<std::string, int> thermal_scatt_map;
extern vector<unique_ptr<ThermalScattering>> thermal_scatt;
} // namespace data

//==============================================================================
//! Secondary angle-energy data for thermal neutron scattering at a single
//! temperature
//==============================================================================

class ThermalData {
public:
  ThermalData(hid_t group);

  //! Calculate the cross section
  //
  //! \param[in] E Incident neutron energy in [eV]
  //! \param[out] elastic Elastic scattering cross section in [b]
  //! \param[out] inelastic Inelastic scattering cross section in [b]
  void calculate_xs(double E, double* elastic, double* inelastic) const;

  //! Sample an outgoing energy and angle
  //
  //! \param[in] micro_xs Microscopic cross sections
  //! \param[in] E_in Incident neutron energy in [eV]
  //! \param[out] E_out Outgoing neutron energy in [eV]
  //! \param[out] mu Outgoing scattering angle cosine
  //! \param[inout] seed Pseudorandom seed pointer
  void sample(const NuclideMicroXS& micro_xs, double E_in, double* E_out,
    double* mu, uint64_t* seed);

private:
  struct Reaction {
    // Default constructor
    Reaction() {}

    // Data members
    unique_ptr<Function1D> xs; //!< Cross section
    unique_ptr<AngleEnergy>
      distribution; //!< Secondary angle-energy distribution
  };

  // Inelastic scattering data
  Reaction elastic_;
  Reaction inelastic_;

  // ThermalScattering needs access to private data members
  friend class ThermalScattering;
};

//==============================================================================
//! Data for thermal neutron scattering, typically off light isotopes in
//! moderating materials such as water, graphite, BeO, etc.
//==============================================================================

class ThermalScattering {
public:
  ThermalScattering(hid_t group, const vector<double>& temperature);

  //! Determine inelastic/elastic cross section at given energy
  //!
  //! \param[in] E incoming energy in [eV]
  //! \param[in] sqrtkT square-root of temperature multipled by Boltzmann's constant 
  //! \param[out] i_temp corresponding temperature index 
  //! \param[out] elastic Thermal elastic scattering cross section 
  //! \param[out] inelastic Thermal inelastic scattering cross section 
  //! \param[inout] seed Pseudorandom seed pointer
  void calculate_xs(double E, double sqrtkT, int* i_temp, double* elastic,
    double* inelastic, uint64_t* seed) const;

  //! Determine whether table applies to a particular nuclide
  //!
  //! \param[in] name Name of the nuclide, e.g., "H1"
  //! \return Whether table applies to the nuclide
  bool has_nuclide(const char* name) const;

  std::string name_;   //!< name of table, e.g. "c_H_in_H2O"
  double awr_;         //!< weight of nucleus in neutron masses
  double energy_max_;  //!< maximum energy for thermal scattering in [eV]
  vector<double> kTs_; //!< temperatures in [eV] (k*T)
  vector<std::string> nuclides_; //!< Valid nuclides

  //! cross sections and distributions at each temperature
  vector<ThermalData> data_;

  //// OTF sampling
  // Sample an outgoing energy and angle
  void sample_otf(double sqrtkT, double E_in, double* E_out, double* mu, uint64_t* seed);

  bool is_otf_ = false;//!< is the file an otf file

  private:

  bool otf_has_elastic_coherent = false;
  std::vector<double> otf_elastic_coherent_energies;
  std::vector<int> otf_elastic_coherent_interp_laws;
  std::vector<double> otf_elastic_coherent_s_vals;
  std::vector<double> otf_elastic_coherent_temperatures;

  bool otf_has_elastic_incoherent = false;
  std::vector<double> otf_elastic_incoherent_debye_wallers;
  int otf_elastic_incoherent_interp_law;
  std::vector<double> otf_elastic_incoherent_temperatures;

  bool otf_has_inelastic = false;
  double otf_inelastic_A0;
  double otf_inelastic_bound_xs;
  double otf_inelastic_e_max;
  double otf_inelastic_free_xs;
  double otf_inelastic_m0;
  double otf_inelastic_mat;
  double otf_inelastic_max_t;
  double otf_inelastic_min_t;
  double otf_inelastic_za;

  double otf_inelastic_alpha_max_scale;
  double otf_inelastic_alpha_min_scale;
  std::string otf_inelastic_alpha_fitting_function_str;
  std::vector<double> otf_inelastic_alpha_coeffs;
  std::vector<double> otf_inelastic_alpha_cdf_grid;
  std::vector<double> otf_inelastic_alpha_beta_grid;

  double otf_inelastic_beta_max_scale;
  double otf_inelastic_beta_min_scale;
  std::string otf_inelastic_beta_fitting_function_str;
  std::vector<double> otf_inelastic_beta_coeffs;
  std::vector<double> otf_inelastic_beta_cdf_grid;
  std::vector<double> otf_inelastic_beta_energy_grid;

  double otf_inelastic_xs_max_scale;
  double otf_inelastic_xs_min_scale;
  std::string otf_inelastic_xs_fitting_function_str;
  std::vector<double> otf_inelastic_xs_coeffs;
  std::vector<double> otf_inelastic_xs_energy_grid;
};

void free_memory_thermal();

} // namespace openmc

#endif // OPENMC_THERMAL_H
