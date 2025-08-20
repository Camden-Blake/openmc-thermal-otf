#ifndef OPENMC_THERMAL_OTF_H
#define OPENMC_THERMAL_OTF_H

#include <optional>

#include "openmc/hdf5_interface.h"
#include "openmc/particle_data.h"


class ThermalScatteringOTF {
    public:

        typedef std::vector<double>(*FuncPointer)(const double &, const int &);

        struct Thermal_Cross_Sections{
            double inelastic;
            double elastic_coherent;
            double elastic_incoherent;
        };

        ThermalScatteringOTF(const hid_t& group);

        Thermal_Cross_Sections calculate_xs(const double& energy, const double& sqrtkT);

        void sample_collision(const openmc::NuclideMicroXS& xs_data, const double& inc_energy, const double& sqrtkT, double& E_out, double& mu, uint64_t* seed);

    private:

        struct Inelastic_Fit_1D {
            FuncPointer fit_function = nullptr;
            double max_scale = 0.0;
            double min_scale = 0.0;
            std::vector<double> x;
            std::vector<double> coeffs;
            int num_coeffs;
        };

        struct Inelastic_Fit_2D {
            FuncPointer fit_function = nullptr;
            double max_scale = 0.0;
            double min_scale = 0.0;
            std::vector<double> x;
            std::vector<double> y;
            std::vector<double> coeffs;
            int num_coeffs;
        };

        struct Coherent_Elastic_Data {
            std::vector<double> energies;
            std::vector<double> s_vals;
            std::vector<double> temperatures;
            std::vector<int> interp_laws;
        };

        struct Incoherent_Elastic_Data {
            std::vector<double> debye_wallers;
            std::vector<double> temperatures;
            int interp_law = 0;
        };

        struct Elastic_Data {
            std::optional<Coherent_Elastic_Data> coherent;
            std::optional<Incoherent_Elastic_Data> incoherent;
        };

        struct Inelastic_Data {
            double min_t = 0.0;
            double max_t = 0.0;
            Inelastic_Fit_1D xs;
            Inelastic_Fit_2D beta;
            Inelastic_Fit_2D alpha;
        };

        struct OTF_Data {
            double A0;
            double bound_xs;
            std::optional<Inelastic_Data> inelastic;
            std::optional<Elastic_Data> elastic;
        };

        struct Alpha_Extrema {
            double minimum;
            double maximum;
        };
        
        typedef std::vector<double>::const_iterator vec_iter;

        OTF_Data data;

        ThermalScatteringOTF::Inelastic_Fit_1D read_inelastic_fit_1D(const hid_t& group, const char* x_name);

        ThermalScatteringOTF::Inelastic_Fit_2D read_inelastic_fit_2D(const hid_t& group, const char* x_name, const char* y_name);

        ThermalScatteringOTF::Inelastic_Data read_inelastic_data(const hid_t& group);

        ThermalScatteringOTF::Coherent_Elastic_Data read_coherent_elastic_data(const hid_t& group);

        ThermalScatteringOTF::Incoherent_Elastic_Data read_incoherent_elastic_data(const hid_t& group);

        ThermalScatteringOTF::Elastic_Data read_elastic_data(const hid_t& group);

        OTF_Data read_data(const hid_t& group);

        double calculate_elastic_coherent_xs(const double& energy, const double& temperature);

        double calculate_elastic_incoherent_xs(const double& energy, const double& temperature);

        double calculate_inelastic_xs(const double& energy, const double& temperature);

        void sample_elastic_coherent_collision(const double& inc_energy, const double& temperature, double& out_energy, double& out_angle, uint64_t* seed);

        void sample_elastic_incoherent_collision(const double& inc_energy, const double& temperature, double& out_energy, double& out_angle, uint64_t* seed);

        void sample_inelastic_collision(const double& inc_energy, const double& temperature, double& out_energy, double& out_angle, uint64_t* seed);

        double sample_beta(const double& inc_energy, const double& temperature, const double& random);

        double sample_alpha(const double& inc_ener, const double& temperature, const double& beta, const double& random);

        double sample_bounding_alpha(const int& beta_ind, const Alpha_Extrema& alpha_extrema, const double& random, const std::vector<double>& evaled_basis_points);

        double rescale_alpha_random_number(const int& beta_ind, const Alpha_Extrema& alpha_extrema, const std::vector<double>& evaled_basis_points, const double& random);

        double reverse_search_alpha_cdf(const int& beta_ind, const vec_iter& alpha_start, const vec_iter& alpha_end, const double& alpha_search, const std::vector<double> evaled_basis_points);

        Alpha_Extrema calculate_alpha_extrema(const double& inc_energy, const double& temperature, const double& beta);

        double calculate_scattering_energy(const double& inc_energy, const double& temperature, const double& beta);

        double calculate_scattering_angle(const double inc_energy,const double& scat_energy, const double& temperature, const double& alpha);
};

#endif // OPENMC_THERMAL_OTF_H
