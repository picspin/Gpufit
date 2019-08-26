#ifndef GPUFIT_MODELS_CUH_INCLUDED
#define GPUFIT_MODELS_CUH_INCLUDED

#include "linear_1d.cuh"
#include "gauss_1d.cuh"
#include "gauss_2d.cuh"
#include "gauss_2d_elliptic.cuh"
#include "gauss_2d_rotated.cuh"
#include "cauchy_2d_elliptic.cuh"
#include "fletcher_powell_helix.cuh"
#include "brown_dennis.cuh"

#include "exponential.cuh"
#include "liver_fat_two.cuh"
#include "liver_fat_three.cuh"
#include "liver_fat_four.cuh"
#include "patlak.cuh"
#include "tofts.cuh"
#include "tofts_extended.cuh"
#include "tissue_uptake.cuh"
#include "two-compartment_exchange.cuh"

__device__ void calculate_model(
    ModelID const model_id,
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    int const user_info_size)
{
    switch (model_id)
    {
    case GAUSS_1D:
        calculate_gauss1d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case GAUSS_2D:
        calculate_gauss2d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case GAUSS_2D_ELLIPTIC:
        calculate_gauss2delliptic(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case GAUSS_2D_ROTATED:
        calculate_gauss2drotated(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case CAUCHY_2D_ELLIPTIC:
        calculate_cauchy2delliptic(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case LINEAR_1D:
        calculate_linear1d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case FLETCHER_POWELL_HELIX:
        calculate_fletcher_powell_helix(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case BROWN_DENNIS:
        calculate_brown_dennis(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;

	// Added models
    case EXPONENTIAL:
    	calculate_exponential(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
    	break;
    case LIVER_FAT_TWO:
    	calculate_liver_fat_2(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
    	break;
    case LIVER_FAT_THREE:
    	calculate_liver_fat_3(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
    	break;
    case LIVER_FAT_FOUR:
    	calculate_liver_fat_4(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
    	break;
    case PATLAK:
		calculate_patlak(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
		break;
    case TOFTS:
		calculate_tofts(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
		break;
    case TOFTS_EXTENDED:
		calculate_tofts_extended(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
		break;
    case TISSUE_UPTAKE:
		calculate_tissue_uptake(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
		break;
    case TWO_COMPARTMENT_EXCHANGE:
		calculate_two_compartment_exchange(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
		break;
    default:
        break;
    }
}
//n_parameters and n_model_parameters are the same in simple_example.cpp
void configure_model(ModelID const model_id, int & n_parameters, int & n_dimensions)
{
    switch (model_id)
    {
    case GAUSS_1D:                  n_parameters = 4; n_dimensions = 1; break;
    case GAUSS_2D:                  n_parameters = 5; n_dimensions = 2; break;
    case GAUSS_2D_ELLIPTIC:         n_parameters = 6; n_dimensions = 2; break;
    case GAUSS_2D_ROTATED:          n_parameters = 7; n_dimensions = 2; break;
    case CAUCHY_2D_ELLIPTIC:        n_parameters = 6; n_dimensions = 2; break;
    case LINEAR_1D:                 n_parameters = 2; n_dimensions = 1; break;
    case FLETCHER_POWELL_HELIX:     n_parameters = 3; n_dimensions = 1; break;
    case BROWN_DENNIS:              n_parameters = 4; n_dimensions = 1; break;
    case EXPONENTIAL:			    n_parameters = 2; n_dimensions = 1; break;
    
    case LIVER_FAT_TWO:			    n_parameters = 2; n_dimensions = 1; break;
    case LIVER_FAT_THREE:		    n_parameters = 3; n_dimensions = 1; break;
    case LIVER_FAT_FOUR:		    n_parameters = 4; n_dimensions = 1; break;
    case PATLAK:					n_parameters = 2; n_dimensions = 1; break;
    case TOFTS:						n_parameters = 2; n_dimensions = 1; break;
    case TOFTS_EXTENDED:			n_parameters = 3; n_dimensions = 1; break;
    case TISSUE_UPTAKE:				n_parameters = 3; n_dimensions = 1; break;
    case TWO_COMPARTMENT_EXCHANGE:	n_parameters = 4; n_dimensions = 1; break;
    default:															break;
    }
}

#endif // GPUFIT_MODELS_CUH_INCLUDED
