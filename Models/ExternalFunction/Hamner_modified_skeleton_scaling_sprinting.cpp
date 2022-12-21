#include <OpenSim/Simulation/Model/Model.h>
#include <OpenSim/Simulation/SimbodyEngine/PinJoint.h>
#include <OpenSim/Simulation/SimbodyEngine/WeldJoint.h>
#include <OpenSim/Simulation/SimbodyEngine/Joint.h>
#include <OpenSim/Simulation/SimbodyEngine/SpatialTransform.h>
#include <OpenSim/Simulation/SimbodyEngine/CustomJoint.h>
#include <OpenSim/Common/LinearFunction.h>
#include <OpenSim/Common/PolynomialFunction.h>
#include <OpenSim/Common/MultiplierFunction.h>
#include <OpenSim/Common/Constant.h>
#include <OpenSim/Simulation/Model/SmoothSphereHalfSpaceForce.h>
#include "SimTKcommon/internal/recorder.h"

#include <iostream>
#include <iterator>
#include <random>
#include <cassert>
#include <algorithm>
#include <vector>
#include <fstream>

using namespace SimTK;
using namespace OpenSim;

constexpr int n_in = 3;
constexpr int n_out = 1;
constexpr int nCoordinates = 31;
constexpr int NX = nCoordinates * 2;
constexpr int NU = nCoordinates;
constexpr int NS = 18 * 3 + 1;
constexpr int NR = 105;


template<typename T>
T value(const Recorder& e) { return e; };
template<>
double value(const Recorder& e) { return e.getValue(); };

SimTK::Array_<int> getIndicesOSInSimbody(const Model& model) {
	auto s = model.getWorkingState();
	const auto svNames = model.getStateVariableNames();
	SimTK::Array_<int> idxOSInSimbody(s.getNQ());
	s.updQ() = 0;
	for (int iy = 0; iy < s.getNQ(); ++iy) {
		s.updQ()[iy] = SimTK::NaN;
		const auto svValues = model.getStateVariableValues(s);
		for (int isv = 0; isv < svNames.size(); ++isv) {
			if (SimTK::isNaN(svValues[isv])) {
				s.updQ()[iy] = 0;
				idxOSInSimbody[iy] = isv / 2;
				break;
			}
		}
	}
	return idxOSInSimbody;
}

SimTK::Array_<int> getIndicesSimbodyInOS(const Model& model) {
	auto idxOSInSimbody = getIndicesOSInSimbody(model);
	auto s = model.getWorkingState();
	SimTK::Array_<int> idxSimbodyInOS(s.getNQ());
	for (int iy = 0; iy < s.getNQ(); ++iy) {
		for (int iyy = 0; iyy < s.getNQ(); ++iyy) {
			if (idxOSInSimbody[iyy] == iy) {
				idxSimbodyInOS[iy] = iyy;
				break;
			}
		}
	}
	return idxSimbodyInOS;
}

template<typename T>
int F_generic(const T** arg, T** res) {

	// Read inputs.
	std::vector<T> x(arg[0], arg[0] + NX);
	std::vector<T> u(arg[1], arg[1] + NU);
	std::vector<T> s(arg[2], arg[2] + NS);
	
	// Definition of model.
	OpenSim::Model* model;
	model = new OpenSim::Model();

	// Scaling factors of different bodies
	osim_double_adouble s_pelvis_x = s[0];
	osim_double_adouble s_pelvis_y = s[1];
	osim_double_adouble s_pelvis_z = s[2];

	osim_double_adouble s_femur_l_x = s[3];
	osim_double_adouble s_femur_l_y = s[4];
	osim_double_adouble s_femur_l_z = s[5];

	osim_double_adouble s_tibia_l_x = s[6];
	osim_double_adouble s_tibia_l_y = s[7];
	osim_double_adouble s_tibia_l_z = s[8];

	osim_double_adouble s_talus_l_x = s[9];
	osim_double_adouble s_talus_l_y = s[10];
	osim_double_adouble s_talus_l_z = s[11];
	
	osim_double_adouble s_calcn_l_x = s[12];
	osim_double_adouble s_calcn_l_y = s[13];
	osim_double_adouble s_calcn_l_z = s[14];
	
	osim_double_adouble s_toes_l_x = s[15];
	osim_double_adouble s_toes_l_y = s[16];
	osim_double_adouble s_toes_l_z = s[17];

	osim_double_adouble s_femur_r_x = s[18];
	osim_double_adouble s_femur_r_y = s[19];
	osim_double_adouble s_femur_r_z = s[20];

	osim_double_adouble s_tibia_r_x = s[21];
	osim_double_adouble s_tibia_r_y = s[22];
	osim_double_adouble s_tibia_r_z = s[23];

	osim_double_adouble s_talus_r_x = s[24];
	osim_double_adouble s_talus_r_y = s[25];
	osim_double_adouble s_talus_r_z = s[26];
	
	osim_double_adouble s_calcn_r_x = s[27];
	osim_double_adouble s_calcn_r_y = s[28];
	osim_double_adouble s_calcn_r_z = s[29];
	
	osim_double_adouble s_toes_r_x = s[30];
	osim_double_adouble s_toes_r_y = s[31];
	osim_double_adouble s_toes_r_z = s[32];

	osim_double_adouble s_torso_x = s[33];
	osim_double_adouble s_torso_y = s[34];
	osim_double_adouble s_torso_z = s[35];

	osim_double_adouble s_humerus_l_x = s[36];
	osim_double_adouble s_humerus_l_y = s[37];
	osim_double_adouble s_humerus_l_z = s[38];

	osim_double_adouble s_radiusulnar_l_x = s[39];
	osim_double_adouble s_radiusulnar_l_y = s[40];
	osim_double_adouble s_radiusulnar_l_z = s[41];

	osim_double_adouble s_hand_l_x = s[42];
	osim_double_adouble s_hand_l_y = s[43];
	osim_double_adouble s_hand_l_z = s[44];

	osim_double_adouble s_humerus_r_x = s[45];
	osim_double_adouble s_humerus_r_y = s[46];
	osim_double_adouble s_humerus_r_z = s[47];

	osim_double_adouble s_radiusulnar_r_x = s[48];
	osim_double_adouble s_radiusulnar_r_y = s[49];
	osim_double_adouble s_radiusulnar_r_z = s[50];

	osim_double_adouble s_hand_r_x = s[51];
	osim_double_adouble s_hand_r_y = s[52];
	osim_double_adouble s_hand_r_z = s[53];

	double pelvis_original_mass = 11.77699999999999924682;
	double femur_r_original_mass = 9.30139999999999922409;
	double tibia_r_original_mass = 3.70750000000000001776;
	double talus_r_original_mass = 0.10000000000000000555;
	double calcn_r_original_mass = 1.25000000000000000000;
	double toes_r_original_mass = 0.21659999999999998699;
	double femur_l_original_mass = 9.30139999999999922409;
	double tibia_l_original_mass = 3.70750000000000001776;
	double talus_l_original_mass = 0.10000000000000000555;
	double calcn_l_original_mass = 1.25000000000000000000;
	double toes_l_original_mass = 0.21659999999999998699;
	double torso_original_mass = 26.82659999999999911324;
	double humerus_r_original_mass = 2.03250000000000019540;
	double ulna_r_original_mass = 0.60750000000000003997;
	double radius_r_original_mass = 0.60750000000000003997;
	double hand_r_original_mass = 0.45750000000000001776;
	double humerus_l_original_mass = 2.03250000000000019540;
	double ulna_l_original_mass = 0.60750000000000003997;
	double radius_l_original_mass = 0.60750000000000003997;
	double hand_l_original_mass = 0.45750000000000001776;
	double total_original_mass = pelvis_original_mass + femur_r_original_mass + tibia_r_original_mass + talus_r_original_mass + calcn_r_original_mass + toes_r_original_mass + femur_l_original_mass + tibia_l_original_mass + talus_l_original_mass + calcn_l_original_mass + toes_l_original_mass + torso_original_mass + humerus_r_original_mass + ulna_r_original_mass + radius_r_original_mass + hand_r_original_mass + humerus_l_original_mass + ulna_l_original_mass + radius_l_original_mass + hand_l_original_mass;

	// Change mass distribution based on scaling volumes: scaled absolute segment mass (volumetric scaling ~ density fixed)
	osim_double_adouble pelvis_mass_scaling_step_1 = s_pelvis_x * s_pelvis_y * s_pelvis_z * pelvis_original_mass;
	osim_double_adouble femur_r_mass_scaling_step_1 = s_femur_r_x * s_femur_r_y * s_femur_r_z * femur_r_original_mass;
	osim_double_adouble tibia_r_mass_scaling_step_1 = s_tibia_r_x * s_tibia_r_y * s_tibia_r_z * tibia_r_original_mass;
	osim_double_adouble talus_r_mass_scaling_step_1 = s_talus_r_x * s_talus_r_y * s_talus_r_z * talus_r_original_mass;
	osim_double_adouble calcn_r_mass_scaling_step_1 = s_calcn_r_x * s_calcn_r_y * s_calcn_r_z * calcn_r_original_mass;
	osim_double_adouble toes_r_mass_scaling_step_1 = s_toes_r_x * s_toes_r_y * s_toes_r_z * toes_r_original_mass;
	osim_double_adouble femur_l_mass_scaling_step_1 = s_femur_l_x * s_femur_l_y * s_femur_l_z * femur_l_original_mass;
	osim_double_adouble tibia_l_mass_scaling_step_1 = s_tibia_l_x * s_tibia_l_y * s_tibia_l_z * tibia_l_original_mass;
	osim_double_adouble talus_l_mass_scaling_step_1 = s_talus_l_x * s_talus_l_y * s_talus_l_z * talus_l_original_mass;
	osim_double_adouble calcn_l_mass_scaling_step_1 = s_calcn_l_x * s_calcn_l_y * s_calcn_l_z * calcn_l_original_mass;
	osim_double_adouble toes_l_mass_scaling_step_1 = s_toes_l_x * s_toes_l_y * s_toes_l_z * toes_l_original_mass;
	osim_double_adouble torso_mass_scaling_step_1 = s_torso_x * s_torso_y * s_torso_z * torso_original_mass;
	osim_double_adouble humerus_r_mass_scaling_step_1 = s_humerus_r_x * s_humerus_r_y * s_humerus_r_z * humerus_r_original_mass;
	osim_double_adouble radius_r_mass_scaling_step_1 = s_radiusulnar_r_x * s_radiusulnar_r_y * s_radiusulnar_r_z * radius_r_original_mass;
	osim_double_adouble ulna_r_mass_scaling_step_1 = s_radiusulnar_r_x * s_radiusulnar_r_y * s_radiusulnar_r_z * ulna_r_original_mass;
	osim_double_adouble hand_r_mass_scaling_step_1 = s_hand_r_x * s_hand_r_y * s_hand_r_z * hand_r_original_mass;
	osim_double_adouble humerus_l_mass_scaling_step_1 = s_humerus_l_x * s_humerus_l_y * s_humerus_l_z * humerus_l_original_mass;
	osim_double_adouble radius_l_mass_scaling_step_1 = s_radiusulnar_l_x * s_radiusulnar_l_y * s_radiusulnar_l_z * radius_l_original_mass;
	osim_double_adouble ulna_l_mass_scaling_step_1 = s_radiusulnar_l_x * s_radiusulnar_l_y * s_radiusulnar_l_z * ulna_l_original_mass;
	osim_double_adouble hand_l_mass_scaling_step_1 = s_hand_l_x * s_hand_l_y * s_hand_l_z * hand_l_original_mass;

	// Scaled total mass
	osim_double_adouble total_mass_scaling_step_1 = pelvis_mass_scaling_step_1 + femur_r_mass_scaling_step_1 + tibia_r_mass_scaling_step_1 + talus_r_mass_scaling_step_1 + calcn_r_mass_scaling_step_1 + toes_r_mass_scaling_step_1 + femur_l_mass_scaling_step_1 + tibia_l_mass_scaling_step_1 + talus_l_mass_scaling_step_1 + calcn_l_mass_scaling_step_1 + toes_l_mass_scaling_step_1 + torso_mass_scaling_step_1 + humerus_r_mass_scaling_step_1 + radius_r_mass_scaling_step_1 + ulna_r_mass_scaling_step_1 + hand_r_mass_scaling_step_1 + humerus_l_mass_scaling_step_1 + radius_l_mass_scaling_step_1 + ulna_l_mass_scaling_step_1 + hand_l_mass_scaling_step_1;

	// Segment mass scaling factor
	osim_double_adouble pelvis_mass_scaling = pelvis_mass_scaling_step_1 / pelvis_original_mass;

	osim_double_adouble femur_r_mass_scaling = femur_r_mass_scaling_step_1 / femur_r_original_mass;
	osim_double_adouble tibia_r_mass_scaling = tibia_r_mass_scaling_step_1 / tibia_r_original_mass;
	osim_double_adouble talus_r_mass_scaling = talus_r_mass_scaling_step_1 / talus_r_original_mass;
	osim_double_adouble calcn_r_mass_scaling = calcn_r_mass_scaling_step_1 / calcn_r_original_mass;
	osim_double_adouble toes_r_mass_scaling = toes_r_mass_scaling_step_1 / toes_r_original_mass;

	osim_double_adouble femur_l_mass_scaling = femur_l_mass_scaling_step_1 / femur_l_original_mass;
	osim_double_adouble tibia_l_mass_scaling = tibia_l_mass_scaling_step_1 / tibia_l_original_mass;
	osim_double_adouble talus_l_mass_scaling = talus_l_mass_scaling_step_1 / talus_l_original_mass;
	osim_double_adouble calcn_l_mass_scaling = calcn_l_mass_scaling_step_1 / calcn_l_original_mass;
	osim_double_adouble toes_l_mass_scaling = toes_l_mass_scaling_step_1 / toes_l_original_mass;

	osim_double_adouble torso_mass_scaling = torso_mass_scaling_step_1 / torso_original_mass;

	osim_double_adouble humerus_r_mass_scaling = humerus_r_mass_scaling_step_1 / humerus_r_original_mass;
	osim_double_adouble radius_r_mass_scaling = radius_r_mass_scaling_step_1 / radius_r_original_mass;
	osim_double_adouble ulna_r_mass_scaling = ulna_r_mass_scaling_step_1 / ulna_r_original_mass;
	osim_double_adouble hand_r_mass_scaling = hand_r_mass_scaling_step_1 / hand_r_original_mass;

	osim_double_adouble humerus_l_mass_scaling = humerus_l_mass_scaling_step_1 / humerus_l_original_mass;
	osim_double_adouble radius_l_mass_scaling = radius_l_mass_scaling_step_1 / radius_l_original_mass;
	osim_double_adouble ulna_l_mass_scaling = ulna_l_mass_scaling_step_1 / ulna_l_original_mass;
	osim_double_adouble hand_l_mass_scaling = hand_l_mass_scaling_step_1 / hand_l_original_mass;
	

	// Maintain mass distribution - the simple way
	/*
	osim_double_adouble pelvis_mass_scaling = s_mass;

	osim_double_adouble femur_r_mass_scaling = s_mass;
	osim_double_adouble tibia_r_mass_scaling = s_mass;
	osim_double_adouble talus_r_mass_scaling = s_mass;
	osim_double_adouble calcn_r_mass_scaling = s_mass;
	osim_double_adouble toes_r_mass_scaling = s_mass;

	osim_double_adouble femur_l_mass_scaling = s_mass;
	osim_double_adouble tibia_l_mass_scaling = s_mass;
	osim_double_adouble talus_l_mass_scaling = s_mass;
	osim_double_adouble calcn_l_mass_scaling = s_mass;
	osim_double_adouble toes_l_mass_scaling = s_mass;

	osim_double_adouble torso_mass_scaling = s_mass;

	osim_double_adouble humerus_r_mass_scaling = s_mass ;
	osim_double_adouble radius_r_mass_scaling = s_mass ;
	osim_double_adouble ulna_r_mass_scaling = s_mass ;
	osim_double_adouble hand_r_mass_scaling = s_mass ;

	osim_double_adouble humerus_l_mass_scaling = s_mass;
	osim_double_adouble radius_l_mass_scaling = s_mass;
	osim_double_adouble ulna_l_mass_scaling = s_mass;
	osim_double_adouble hand_l_mass_scaling = s_mass;
	*/

	// Definition of bodies.	
	OpenSim::Body* pelvis;
	pelvis = new OpenSim::Body("pelvis", pelvis_mass_scaling * pelvis_original_mass, Vec3(-0.07069999999999999896 * s_pelvis_x, 0.00000000000000000000 * s_pelvis_y, 0.00000000000000000000 * s_pelvis_z), pelvis_mass_scaling * Inertia(0.10280000000000000249 * s_pelvis_y * s_pelvis_z, 0.08709999999999999687 * s_pelvis_x * s_pelvis_z, 0.05790000000000000008 * s_pelvis_x * s_pelvis_y, 0., 0., 0.));
	model->addBody(pelvis);
	
	OpenSim::Body* femur_r;
	femur_r = new OpenSim::Body("femur_r", femur_r_mass_scaling * femur_r_original_mass, Vec3(0.00000000000000000000 * s_femur_r_x, -0.17000000000000001221 * s_femur_r_y, 0.00000000000000000000 * s_femur_r_z), femur_r_mass_scaling *  Inertia(0.13389999999999999125 * pow(s_femur_r_y, 2), 0.03509999999999999926 * s_femur_r_x * s_femur_r_z, 0.14119999999999999218 * pow(s_femur_r_y, 2), 0., 0., 0.));
	model->addBody(femur_r);

	OpenSim::Body* tibia_r;
	tibia_r = new OpenSim::Body("tibia_r", tibia_r_mass_scaling * tibia_r_original_mass, Vec3(0.00000000000000000000 * s_tibia_r_x, -0.18670000000000000484 * s_tibia_r_y, 0.00000000000000000000 * s_tibia_r_z), tibia_r_mass_scaling *  Inertia(0.05040000000000000036 * pow(s_tibia_r_y, 2), 0.00510000000000000037 * s_tibia_r_x * s_tibia_r_z, 0.05109999999999999959 * pow(s_tibia_r_y, 2), 0., 0., 0.));
	model->addBody(tibia_r);

	OpenSim::Body* talus_r;
	talus_r = new OpenSim::Body("talus_r", talus_r_mass_scaling * talus_r_original_mass, Vec3(0.00000000000000000000 * s_talus_r_x, 0.00000000000000000000 * s_talus_r_y, 0.00000000000000000000 * s_talus_r_z), talus_r_mass_scaling *  Inertia(0.00100000000000000002 * pow(s_talus_r_y, 2), 0.00100000000000000002 * s_talus_r_x * s_talus_r_z, 0.00100000000000000002 * pow(s_talus_r_y, 2), 0., 0., 0.));
	model->addBody(talus_r);

	OpenSim::Body* calcn_r;
	calcn_r = new OpenSim::Body("calcn_r", calcn_r_mass_scaling * calcn_r_original_mass, Vec3(0.10000000000000000555 * s_calcn_r_x, 0.02999999999999999889 * s_calcn_r_y, 0.00000000000000000000 * s_calcn_r_z), calcn_r_mass_scaling *  Inertia(0.00139999999999999999 * pow(s_calcn_r_y, 2), 0.00389999999999999982 * s_calcn_r_x * s_calcn_r_z, 0.00410000000000000035 * pow(s_calcn_r_y, 2), 0., 0., 0.));
	model->addBody(calcn_r);

	OpenSim::Body* toes_r;
	toes_r = new OpenSim::Body("toes_r", toes_r_mass_scaling * toes_r_original_mass, Vec3(0.03459999999999999881 * s_toes_r_x, 0.00600000000000000012 * s_toes_r_y, -0.01750000000000000167 * s_toes_r_z), toes_r_mass_scaling *  Inertia(0.00010000000000000000 * pow(s_toes_r_y, 2), 0.00020000000000000001 * s_toes_r_x * s_toes_r_z, 0.00100000000000000002 * pow(s_toes_r_y, 2), 0., 0., 0.));
	model->addBody(toes_r);

	OpenSim::Body* femur_l;
	femur_l = new OpenSim::Body("femur_l", femur_l_mass_scaling * femur_l_original_mass, Vec3(0.00000000000000000000 * s_femur_l_x, -0.17000000000000001221 * s_femur_l_y, 0.00000000000000000000 * s_femur_l_z), femur_l_mass_scaling *  Inertia(0.13389999999999999125 * pow(s_femur_l_y, 2), 0.03509999999999999926 * s_femur_l_x * s_femur_l_z, 0.14119999999999999218 * pow(s_femur_l_y, 2), 0., 0., 0.));
	model->addBody(femur_l);

	OpenSim::Body* tibia_l;
	tibia_l = new OpenSim::Body("tibia_l", tibia_l_mass_scaling * tibia_l_original_mass, Vec3(0.00000000000000000000 * s_tibia_l_x, -0.18670000000000000484 * s_tibia_l_y, 0.00000000000000000000 * s_tibia_l_z), tibia_l_mass_scaling *  Inertia(0.05040000000000000036 * pow(s_tibia_l_y, 2), 0.00510000000000000037 * s_tibia_l_x * s_tibia_l_z, 0.05109999999999999959 * pow(s_tibia_l_y, 2), 0., 0., 0.));
	model->addBody(tibia_l);

	OpenSim::Body* talus_l;
	talus_l = new OpenSim::Body("talus_l", talus_l_mass_scaling * talus_l_original_mass, Vec3(0.00000000000000000000 * s_talus_l_x, 0.00000000000000000000 * s_talus_l_y, 0.00000000000000000000 * s_talus_l_z), talus_l_mass_scaling *  Inertia(0.00100000000000000002 * pow(s_talus_l_y, 2), 0.00100000000000000002 * s_talus_l_x * s_talus_l_z, 0.00100000000000000002 * pow(s_talus_l_y, 2), 0., 0., 0.));
	model->addBody(talus_l);

	OpenSim::Body* calcn_l;
	calcn_l = new OpenSim::Body("calcn_l", calcn_l_mass_scaling * calcn_l_original_mass, Vec3(0.10000000000000000555 * s_calcn_l_x, 0.02999999999999999889 * s_calcn_l_y, 0.00000000000000000000 * s_calcn_l_z), calcn_l_mass_scaling *  Inertia(0.00139999999999999999 * pow(s_calcn_l_y, 2), 0.00389999999999999982 * s_calcn_l_x * s_calcn_l_z, 0.00410000000000000035 * pow(s_calcn_l_y, 2), 0., 0., 0.));
	model->addBody(calcn_l);

	OpenSim::Body* toes_l;
	toes_l = new OpenSim::Body("toes_l", toes_l_mass_scaling * toes_l_original_mass, Vec3(0.03459999999999999881 * s_toes_l_x, 0.00600000000000000012 * s_toes_l_y, 0.01750000000000000167 * s_toes_l_z), toes_l_mass_scaling *  Inertia(0.00010000000000000000 * pow(s_toes_l_y, 2), 0.00020000000000000001 * s_toes_l_x * s_toes_l_z, 0.00100000000000000002 * pow(s_toes_l_y, 2), 0., 0., 0.));
	model->addBody(toes_l);

	OpenSim::Body* torso;
	torso = new OpenSim::Body("torso", torso_mass_scaling * torso_original_mass, Vec3(-0.02999999999999999889 * s_torso_x, 0.32000000000000000666 * s_torso_y, 0.00000000000000000000 * s_torso_z), torso_mass_scaling *  Inertia(1.47449999999999992184 * pow(s_torso_y, 2), 0.75549999999999994937 * s_torso_x * s_torso_z, 1.43140000000000000568 * pow(s_torso_y, 2), 0., 0., 0.));
	model->addBody(torso);

	OpenSim::Body* humerus_r;
	humerus_r = new OpenSim::Body("humerus_r", humerus_r_mass_scaling * humerus_r_original_mass, Vec3(0.00000000000000000000 * s_humerus_r_x, -0.16450200000000000933 * s_humerus_r_y, 0.00000000000000000000 * s_humerus_r_z), humerus_r_mass_scaling *  Inertia(0.01194600000000000002 * pow(s_humerus_r_y, 2), 0.00412099999999999966 * s_humerus_r_x * s_humerus_r_z, 0.01340900000000000078 * pow(s_humerus_r_y, 2), 0., 0., 0.));
	model->addBody(humerus_r);

	OpenSim::Body* ulna_r;
	ulna_r = new OpenSim::Body("ulna_r", radius_r_mass_scaling * radius_r_original_mass, Vec3(0.00000000000000000000 * s_radiusulnar_r_x, -0.12052499999999999325 * s_radiusulnar_r_y, 0.00000000000000000000 * s_radiusulnar_r_z), radius_r_mass_scaling *  Inertia(0.00296199999999999979 * pow(s_radiusulnar_r_y, 2), 0.00061799999999999995 * s_radiusulnar_r_x * s_radiusulnar_r_z, 0.00321300000000000014 * pow(s_radiusulnar_r_y, 2), 0., 0., 0.));
	model->addBody(ulna_r);

	OpenSim::Body* radius_r;
	radius_r = new OpenSim::Body("radius_r", ulna_r_mass_scaling * ulna_r_original_mass, Vec3(0.00000000000000000000 * s_radiusulnar_r_x, -0.12052499999999999325 * s_radiusulnar_r_y, 0.00000000000000000000 * s_radiusulnar_r_z), ulna_r_mass_scaling *  Inertia(0.00296199999999999979 * pow(s_radiusulnar_r_y, 2), 0.00061799999999999995 * s_radiusulnar_r_x * s_radiusulnar_r_z, 0.00321300000000000014 * pow(s_radiusulnar_r_y, 2), 0., 0., 0.));
	model->addBody(radius_r);

	OpenSim::Body* hand_r;
	hand_r = new OpenSim::Body("hand_r", hand_r_mass_scaling * hand_r_original_mass, Vec3(0.00000000000000000000 * s_hand_r_x, -0.06809500000000000275 * s_hand_r_y, 0.00000000000000000000 * s_hand_r_z), hand_r_mass_scaling *  Inertia(0.00089200000000000000 * pow(s_hand_r_y, 2), 0.00054699999999999996 * s_hand_r_x * s_hand_r_z, 0.00134000000000000005 * pow(s_hand_r_y, 2), 0., 0., 0.));
	model->addBody(hand_r);

	OpenSim::Body* humerus_l;
	humerus_l = new OpenSim::Body("humerus_l", humerus_l_mass_scaling * humerus_l_original_mass, Vec3(0.00000000000000000000 * s_humerus_l_x, -0.16450200000000000933 * s_humerus_l_y, 0.00000000000000000000 * s_humerus_l_z), humerus_l_mass_scaling * Inertia(0.01194600000000000002 * pow(s_humerus_l_y, 2), 0.00412099999999999966 * s_humerus_l_x * s_humerus_l_z, 0.01340900000000000078 * pow(s_humerus_l_y, 2), 0., 0., 0.));
	model->addBody(humerus_l);

	OpenSim::Body* ulna_l;
	ulna_l = new OpenSim::Body("ulna_l", radius_l_mass_scaling * radius_l_original_mass, Vec3(0.00000000000000000000 * s_radiusulnar_l_x, -0.12052499999999999325 * s_radiusulnar_l_y, 0.00000000000000000000 * s_radiusulnar_l_z), radius_l_mass_scaling * Inertia(0.00296199999999999979 * pow(s_radiusulnar_l_y, 2), 0.00061799999999999995 * s_radiusulnar_l_x * s_radiusulnar_l_z, 0.00321300000000000014 * pow(s_radiusulnar_l_y, 2), 0., 0., 0.));
	model->addBody(ulna_l);

	OpenSim::Body* radius_l;
	radius_l = new OpenSim::Body("radius_l", ulna_l_mass_scaling * ulna_l_original_mass, Vec3(0.00000000000000000000 * s_radiusulnar_l_x, -0.12052499999999999325 * s_radiusulnar_l_y, 0.00000000000000000000 * s_radiusulnar_l_z), ulna_l_mass_scaling * Inertia(0.00296199999999999979 * pow(s_radiusulnar_l_y, 2), 0.00061799999999999995 * s_radiusulnar_l_x * s_radiusulnar_l_z, 0.00321300000000000014 * pow(s_radiusulnar_l_y, 2), 0., 0., 0.));
	model->addBody(radius_l);

	OpenSim::Body* hand_l;
	hand_l = new OpenSim::Body("hand_l", hand_l_mass_scaling * hand_l_original_mass, Vec3(0.00000000000000000000 * s_hand_l_x, -0.06809500000000000275 * s_hand_l_y, 0.00000000000000000000 * s_hand_l_z), hand_l_mass_scaling *  Inertia(0.00089200000000000000 * pow(s_hand_l_y, 2), 0.00054699999999999996 * s_hand_l_x * s_hand_l_z, 0.00134000000000000005 * pow(s_hand_l_y, 2), 0., 0., 0.));
	model->addBody(hand_l);


    // Definition of joints.
	SpatialTransform st_ground_pelvis;
	st_ground_pelvis[0].setCoordinateNames(OpenSim::Array<std::string>("pelvis_tilt", 1, 1));
	st_ground_pelvis[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_ground_pelvis[0].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_ground_pelvis[1].setCoordinateNames(OpenSim::Array<std::string>("pelvis_list", 1, 1));
	st_ground_pelvis[1].setFunction(new LinearFunction(1.0000, 0.0000));
	st_ground_pelvis[1].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_ground_pelvis[2].setCoordinateNames(OpenSim::Array<std::string>("pelvis_rotation", 1, 1));
	st_ground_pelvis[2].setFunction(new LinearFunction(1.0000, 0.0000));
	st_ground_pelvis[2].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_ground_pelvis[3].setCoordinateNames(OpenSim::Array<std::string>("pelvis_tx", 1, 1));
	st_ground_pelvis[3].setFunction(new LinearFunction(1.0000, 0.0000));
	st_ground_pelvis[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_ground_pelvis[4].setCoordinateNames(OpenSim::Array<std::string>("pelvis_ty", 1, 1));
	st_ground_pelvis[4].setFunction(new LinearFunction(1.0000, 0.0000));
	st_ground_pelvis[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_ground_pelvis[5].setCoordinateNames(OpenSim::Array<std::string>("pelvis_tz", 1, 1));
	st_ground_pelvis[5].setFunction(new LinearFunction(1.0000, 0.0000));
	st_ground_pelvis[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* ground_pelvis;
	ground_pelvis = new OpenSim::CustomJoint("ground_pelvis", model->getGround(), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *pelvis, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_ground_pelvis);
	model->addJoint(ground_pelvis);

	SpatialTransform st_hip_r;
	st_hip_r[0].setCoordinateNames(OpenSim::Array<std::string>("hip_flexion_r", 1, 1));
	st_hip_r[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_hip_r[0].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_hip_r[1].setCoordinateNames(OpenSim::Array<std::string>("hip_adduction_r", 1, 1));
	st_hip_r[1].setFunction(new LinearFunction(1.0000, 0.0000));
	st_hip_r[1].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_hip_r[2].setCoordinateNames(OpenSim::Array<std::string>("hip_rotation_r", 1, 1));
	st_hip_r[2].setFunction(new LinearFunction(1.0000, 0.0000));
	st_hip_r[2].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_hip_r[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_pelvis_x));
	st_hip_r[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_hip_r[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_pelvis_y));
	st_hip_r[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_hip_r[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_pelvis_z));
	st_hip_r[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* hip_r;
	hip_r = new OpenSim::CustomJoint("hip_r", *pelvis, Vec3(-0.07069999999999999896*s_pelvis_x, -0.06610000000000000597*s_pelvis_y, 0.08350000000000000477*s_pelvis_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *femur_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_hip_r);
	model->addJoint(hip_r);


	SpatialTransform st_knee_r;
	st_knee_r[0].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_r", 1, 1));
	st_knee_r[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_knee_r[0].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_knee_r[1].setFunction(new Constant(0.00000000000000000000));
	st_knee_r[1].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_knee_r[2].setFunction(new Constant(0.00000000000000000000));
	st_knee_r[2].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	
	st_knee_r[3].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_r", 1, 1));
	osim_double_adouble st_knee_r_3_coeffs[5] = { 0.00136186503073299999, 0.00824635530783600010, 0.00825417128968199999, -0.00690645209565499997, -0.00428970680011700033 };
	Vector st_knee_r_3_coeffs_vec(5);
	for (int i = 0; i < 5; ++i) st_knee_r_3_coeffs_vec[i] = st_knee_r_3_coeffs[i];
	st_knee_r[3].setFunction(new MultiplierFunction(new PolynomialFunction(st_knee_r_3_coeffs_vec), s_femur_r_x));
	st_knee_r[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	
	st_knee_r[4].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_r", 1, 1));
	osim_double_adouble st_knee_r_4_coeffs[3] = { -0.00368203588865299988, 0.00524491850058900026, -0.39565045284385197411 };
	Vector st_knee_r_4_coeffs_vec(3);
	for (int i = 0; i < 3; ++i) st_knee_r_4_coeffs_vec[i] = st_knee_r_4_coeffs[i];
	st_knee_r[4].setFunction(new MultiplierFunction(new PolynomialFunction(st_knee_r_4_coeffs_vec), s_femur_r_y));
	st_knee_r[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));

	st_knee_r[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_femur_r_z));
	st_knee_r[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* knee_r;
	knee_r = new OpenSim::CustomJoint("knee_r", *femur_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *tibia_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_knee_r);
	model->addJoint(knee_r);

	SpatialTransform st_ankle_r;
	st_ankle_r[0].setCoordinateNames(OpenSim::Array<std::string>("ankle_angle_r", 1, 1));
	st_ankle_r[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_ankle_r[0].setAxis(Vec3(-0.10501354999999999718, -0.17402244999999999520, 0.97912631999999999444));
	st_ankle_r[1].setFunction(new Constant(0.00000000000000000000));
	st_ankle_r[1].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_ankle_r[2].setFunction(new Constant(0.00000000000000000000));
	st_ankle_r[2].setAxis(Vec3(0.97912631999999999444, -0.00000000000000000000, 0.10501354999999999718));

	st_ankle_r[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_tibia_r_x));
	st_ankle_r[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_ankle_r[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_tibia_r_y));
	st_ankle_r[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_ankle_r[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_tibia_r_z));
	st_ankle_r[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* ankle_r;
	ankle_r = new OpenSim::CustomJoint("ankle_r", *tibia_r, Vec3(0.00000000000000000000 * s_tibia_r_x, -0.42999999999999999334 * s_tibia_r_y, 0.00000000000000000000 * s_tibia_r_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *talus_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_ankle_r);
	model->addJoint(ankle_r);

	SpatialTransform st_subtalar_r;
	st_subtalar_r[0].setCoordinateNames(OpenSim::Array<std::string>("subtalar_angle_r", 1, 1));
	st_subtalar_r[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_subtalar_r[0].setAxis(Vec3(0.78717961000000002958, 0.60474746000000001445, -0.12094949000000000672));
	st_subtalar_r[1].setFunction(new Constant(0.00000000000000000000));
	st_subtalar_r[1].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_subtalar_r[2].setFunction(new Constant(0.00000000000000000000));
	st_subtalar_r[2].setAxis(Vec3(-0.12094949000000000672, 0.00000000000000000000, -0.78717961000000002958));
	st_subtalar_r[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_talus_r_x));
	st_subtalar_r[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_subtalar_r[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_talus_r_y));
	st_subtalar_r[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_subtalar_r[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_talus_r_z));
	st_subtalar_r[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	
	OpenSim::CustomJoint* subtalar_r;
	subtalar_r = new OpenSim::CustomJoint("subtalar_r", *talus_r, Vec3(-0.04877000000000000085 * s_talus_r_x, -0.04195000000000000118 * s_talus_r_y, 0.00791999999999999996 * s_talus_r_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *calcn_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_subtalar_r);
	model->addJoint(subtalar_r);


	SpatialTransform st_mtp_r;
	st_mtp_r[0].setCoordinateNames(OpenSim::Array<std::string>("mtp_angle_r", 1, 1));
	st_mtp_r[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_mtp_r[0].setAxis(Vec3(0, 0, 1));
	st_mtp_r[1].setFunction(new Constant(0.00000000000000000000));
	st_mtp_r[1].setAxis(Vec3(0, 1, 0));
	st_mtp_r[2].setFunction(new Constant(0.00000000000000000000));
	st_mtp_r[2].setAxis(Vec3(1, 0, 0));
	st_mtp_r[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_calcn_r_x));
	st_mtp_r[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_mtp_r[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_calcn_r_y));
	st_mtp_r[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_mtp_r[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_calcn_r_z));
	st_mtp_r[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	
	OpenSim::CustomJoint* mtp_r;
	mtp_r = new OpenSim::CustomJoint("mtp_r", *calcn_r, Vec3(0.17879999999999998672 * s_calcn_r_x, -0.00200000000000000004 * s_calcn_r_y, 0.00108000000000000001 * s_calcn_r_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *toes_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_mtp_r);
	model->addJoint(mtp_r);

	SpatialTransform st_hip_l;
	st_hip_l[0].setCoordinateNames(OpenSim::Array<std::string>("hip_flexion_l", 1, 1));
	st_hip_l[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_hip_l[0].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_hip_l[1].setCoordinateNames(OpenSim::Array<std::string>("hip_adduction_l", 1, 1));
	st_hip_l[1].setFunction(new LinearFunction(1.0000, 0.0000));
	st_hip_l[1].setAxis(Vec3(-1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_hip_l[2].setCoordinateNames(OpenSim::Array<std::string>("hip_rotation_l", 1, 1));
	st_hip_l[2].setFunction(new LinearFunction(1.0000, 0.0000));
	st_hip_l[2].setAxis(Vec3(0.00000000000000000000, -1.00000000000000000000, 0.00000000000000000000));
	st_hip_l[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_pelvis_x));
	st_hip_l[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_hip_l[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_pelvis_y));
	st_hip_l[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_hip_l[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_pelvis_z));
	st_hip_l[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* hip_l;
	hip_l = new OpenSim::CustomJoint("hip_l", *pelvis, Vec3(-0.07069999999999999896 * s_pelvis_x, -0.06610000000000000597 * s_pelvis_y, -0.08350000000000000477 * s_pelvis_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *femur_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_hip_l);
	model->addJoint(hip_l);

	SpatialTransform st_knee_l;
	st_knee_l[0].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_l", 1, 1));
	st_knee_l[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_knee_l[0].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_knee_l[1].setFunction(new Constant(0.00000000000000000000));
	st_knee_l[1].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_knee_l[2].setFunction(new Constant(0.00000000000000000000));
	st_knee_l[2].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_knee_l[3].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_l", 1, 1));
	osim_double_adouble st_knee_l_3_coeffs[5] = { 0.00136186503073299999, 0.00824635530783600010, 0.00825417128968199999, -0.00690645209565499997, -0.00428970680011700033 };
	Vector st_knee_l_3_coeffs_vec(5);
	for (int i = 0; i < 5; ++i) st_knee_l_3_coeffs_vec[i] = st_knee_l_3_coeffs[i];
	st_knee_l[3].setFunction(new MultiplierFunction(new PolynomialFunction(st_knee_l_3_coeffs_vec), s_femur_l_x));
	st_knee_l[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));

	st_knee_l[4].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_l", 1, 1));
	osim_double_adouble st_knee_l_4_coeffs[3] = { -0.00368203588865299988, 0.00524491850058900026, -0.39565045284385197411 };
	Vector st_knee_l_4_coeffs_vec(3);
	for (int i = 0; i < 3; ++i) st_knee_l_4_coeffs_vec[i] = st_knee_l_4_coeffs[i];
	st_knee_l[4].setFunction(new MultiplierFunction(new PolynomialFunction(st_knee_l_4_coeffs_vec), s_femur_l_y));
	st_knee_l[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_knee_l[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_femur_l_z));
	st_knee_l[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* knee_l;
	knee_l = new OpenSim::CustomJoint("knee_l", *femur_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *tibia_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_knee_l);
	model->addJoint(knee_l);

	SpatialTransform st_ankle_l;
	st_ankle_l[0].setCoordinateNames(OpenSim::Array<std::string>("ankle_angle_l", 1, 1));
	st_ankle_l[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_ankle_l[0].setAxis(Vec3(0.10501354999999999718, 0.17402244999999999520, 0.97912631999999999444));
	st_ankle_l[1].setFunction(new Constant(0.00000000000000000000));
	st_ankle_l[1].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_ankle_l[2].setFunction(new Constant(0.00000000000000000000));
	st_ankle_l[2].setAxis(Vec3(0.97912631999999999444, 0.00000000000000000000, -0.10501354999999999718));
	st_ankle_l[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_tibia_l_x));
	st_ankle_l[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_ankle_l[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_tibia_l_y));
	st_ankle_l[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_ankle_l[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_tibia_l_z));
	st_ankle_l[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* ankle_l;
	ankle_l = new OpenSim::CustomJoint("ankle_l", *tibia_l, Vec3(0.00000000000000000000 * s_tibia_l_x, -0.42999999999999999334 * s_tibia_l_y, 0.00000000000000000000 * s_tibia_l_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *talus_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_ankle_l);
	model->addJoint(ankle_l);

	SpatialTransform st_subtalar_l;
	st_subtalar_l[0].setCoordinateNames(OpenSim::Array<std::string>("subtalar_angle_l", 1, 1));
	st_subtalar_l[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_subtalar_l[0].setAxis(Vec3(-0.78717961000000002958, -0.60474746000000001445, -0.12094949000000000672));
	st_subtalar_l[1].setFunction(new Constant(0.00000000000000000000));
	st_subtalar_l[1].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_subtalar_l[2].setFunction(new Constant(0.00000000000000000000));
	st_subtalar_l[2].setAxis(Vec3(-0.12094949000000000672, 0.00000000000000000000, 0.78717961000000002958));
	st_subtalar_l[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_talus_l_x));
	st_subtalar_l[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_subtalar_l[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_talus_l_y));
	st_subtalar_l[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_subtalar_l[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_talus_l_z));
	st_subtalar_l[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	
	OpenSim::CustomJoint* subtalar_l;
	subtalar_l = new OpenSim::CustomJoint("subtalar_l", *talus_l, Vec3(-0.04877000000000000085 * s_talus_l_x, -0.04195000000000000118 * s_talus_l_y, -0.00791999999999999996 * s_talus_l_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *calcn_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_subtalar_l);
	model->addJoint(subtalar_l);

	SpatialTransform st_mtp_l;
	st_mtp_l[0].setCoordinateNames(OpenSim::Array<std::string>("mtp_angle_l", 1, 1));
	st_mtp_l[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_mtp_l[0].setAxis(Vec3(0, 0, 1));
	st_mtp_l[1].setFunction(new Constant(0.00000000000000000000));
	st_mtp_l[1].setAxis(Vec3(0, 1, 0));
	st_mtp_l[2].setFunction(new Constant(0.00000000000000000000));
	st_mtp_l[2].setAxis(Vec3(1, 0, 0));
	st_mtp_l[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_calcn_l_x));
	st_mtp_l[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_mtp_l[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_calcn_l_y));
	st_mtp_l[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_mtp_l[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_calcn_l_z));
	st_mtp_l[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	
	OpenSim::CustomJoint* mtp_l;
	mtp_l = new OpenSim::CustomJoint("mtp_l", *calcn_l, Vec3(0.17879999999999998672 * s_calcn_l_x, -0.00200000000000000004 * s_calcn_l_y, -0.00108000000000000001 * s_calcn_l_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *toes_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_mtp_l);
	model->addJoint(mtp_l);

	SpatialTransform st_back;
	st_back[0].setCoordinateNames(OpenSim::Array<std::string>("lumbar_extension", 1, 1));
	st_back[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_back[0].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_back[1].setCoordinateNames(OpenSim::Array<std::string>("lumbar_bending", 1, 1));
	st_back[1].setFunction(new LinearFunction(1.0000, 0.0000));
	st_back[1].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_back[2].setCoordinateNames(OpenSim::Array<std::string>("lumbar_rotation", 1, 1));
	st_back[2].setFunction(new LinearFunction(1.0000, 0.0000));
	st_back[2].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_back[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_pelvis_x));
	st_back[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_back[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_pelvis_y));
	st_back[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_back[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_pelvis_z));
	st_back[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* back;
	back = new OpenSim::CustomJoint("back", *pelvis, Vec3(-0.10069999999999999785 * s_pelvis_x, 0.08150000000000000300 * s_pelvis_y, 0.00000000000000000000 * s_pelvis_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *torso, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_back);
	model->addJoint(back);

	SpatialTransform st_acromial_r;
	st_acromial_r[0].setCoordinateNames(OpenSim::Array<std::string>("arm_flex_r", 1, 1));
	st_acromial_r[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_acromial_r[0].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_acromial_r[1].setCoordinateNames(OpenSim::Array<std::string>("arm_add_r", 1, 1));
	st_acromial_r[1].setFunction(new LinearFunction(1.0000, 0.0000));
	st_acromial_r[1].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_acromial_r[2].setCoordinateNames(OpenSim::Array<std::string>("arm_rot_r", 1, 1));
	st_acromial_r[2].setFunction(new LinearFunction(1.0000, 0.0000));
	st_acromial_r[2].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_acromial_r[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_torso_x));
	st_acromial_r[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_acromial_r[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_torso_y));
	st_acromial_r[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_acromial_r[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_torso_z));
	st_acromial_r[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* acromial_r;
	acromial_r = new OpenSim::CustomJoint("acromial_r", *torso, Vec3(0.00315499999999999982 * s_torso_x, 0.37149999999999999689 * s_torso_y, 0.17000000000000001221 * s_torso_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *humerus_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_acromial_r);
	model->addJoint(acromial_r);

	SpatialTransform st_elbow_r;
	st_elbow_r[0].setCoordinateNames(OpenSim::Array<std::string>("elbow_flex_r", 1, 1));
	st_elbow_r[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_elbow_r[0].setAxis(Vec3(0.22604695999999999123, 0.02226900000000000060, 0.97386183000000003940));
	st_elbow_r[1].setFunction(new Constant(0.00000000000000000000));
	st_elbow_r[1].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_elbow_r[2].setFunction(new Constant(0.00000000000000000000));
	st_elbow_r[2].setAxis(Vec3(0.97386183000000003940, 0.00000000000000000000, -0.22604695999999999123));
	st_elbow_r[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_humerus_r_x));
	st_elbow_r[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_elbow_r[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_humerus_r_y));
	st_elbow_r[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_elbow_r[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_humerus_r_z));
	st_elbow_r[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* elbow_r;
	elbow_r = new OpenSim::CustomJoint("elbow_r", *humerus_r, Vec3(0.01314399999999999943 * s_humerus_r_x, -0.28627299999999999969 * s_humerus_r_y, -0.00959499999999999936 * s_humerus_r_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *ulna_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_elbow_r);
	model->addJoint(elbow_r);

	OpenSim::WeldJoint* radioulnar_r;
	radioulnar_r = new OpenSim::WeldJoint("radioulnar_r", *ulna_r, Vec3(-0.00672700000000000034 * s_radiusulnar_r_x, -0.01300699999999999946 * s_radiusulnar_r_y, 0.02608299999999999855 * s_radiusulnar_r_z), Vec3(0.00000000000000000000, 1.57079632679489655800, 0.00000000000000000000), *radius_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	model->addJoint(radioulnar_r);

	OpenSim::WeldJoint* radius_hand_r;
	radius_hand_r = new OpenSim::WeldJoint("radius_hand_r", *radius_r, Vec3(-0.00879699999999999926 * s_radiusulnar_r_x, -0.23584099999999999508 * s_radiusulnar_r_y, 0.01361000000000000057 * s_radiusulnar_r_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *hand_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	model->addJoint(radius_hand_r);

	SpatialTransform st_acromial_l;
	st_acromial_l[0].setCoordinateNames(OpenSim::Array<std::string>("arm_flex_l", 1, 1));
	st_acromial_l[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_acromial_l[0].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_acromial_l[1].setCoordinateNames(OpenSim::Array<std::string>("arm_add_l", 1, 1));
	st_acromial_l[1].setFunction(new LinearFunction(1.0000, 0.0000));
	st_acromial_l[1].setAxis(Vec3(-1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_acromial_l[2].setCoordinateNames(OpenSim::Array<std::string>("arm_rot_l", 1, 1));
	st_acromial_l[2].setFunction(new LinearFunction(1.0000, 0.0000));
	st_acromial_l[2].setAxis(Vec3(0.00000000000000000000, -1.00000000000000000000, 0.00000000000000000000));
	st_acromial_l[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_torso_x));
	st_acromial_l[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_acromial_l[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_torso_y));
	st_acromial_l[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_acromial_l[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_torso_z));
	st_acromial_l[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* acromial_l;
	acromial_l = new OpenSim::CustomJoint("acromial_l", *torso, Vec3(0.00315499999999999982 * s_torso_x, 0.37149999999999999689 * s_torso_y, -0.17000000000000001221 * s_torso_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *humerus_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_acromial_l);
	model->addJoint(acromial_l);

	SpatialTransform st_elbow_l;
	st_elbow_l[0].setCoordinateNames(OpenSim::Array<std::string>("elbow_flex_l", 1, 1));
	st_elbow_l[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_elbow_l[0].setAxis(Vec3(-0.22604695999999999123, -0.02226900000000000060, 0.97386183000000003940));
	st_elbow_l[1].setFunction(new Constant(0.00000000000000000000));
	st_elbow_l[1].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_elbow_l[2].setFunction(new Constant(0.00000000000000000000));
	st_elbow_l[2].setAxis(Vec3(0.97386183000000003940, -0.00000000000000000000, 0.22604695999999999123));
	st_elbow_l[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_humerus_l_x));
	st_elbow_l[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_elbow_l[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_humerus_l_y));
	st_elbow_l[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_elbow_l[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), s_humerus_l_z));
	st_elbow_l[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* elbow_l;
	elbow_l = new OpenSim::CustomJoint("elbow_l", *humerus_l, Vec3(0.01314399999999999943 * s_humerus_l_x, -0.28627299999999999969 * s_humerus_l_y, 0.00959499999999999936 * s_humerus_l_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *ulna_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_elbow_l);
	model->addJoint(elbow_l);

	OpenSim::WeldJoint* radioulnar_l;
	radioulnar_l = new OpenSim::WeldJoint("radioulnar_l", *ulna_l, Vec3(-0.00672700000000000034 * s_radiusulnar_l_x, -0.01300699999999999946 * s_radiusulnar_l_y, -0.02608299999999999855 * s_radiusulnar_l_z), Vec3(0.00000000000000000000, -1.57079632679489655800, 0.00000000000000000000), *radius_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	model->addJoint(radioulnar_l);

	OpenSim::WeldJoint* radius_hand_l;
	radius_hand_l = new OpenSim::WeldJoint("radius_hand_l", *radius_l, Vec3(-0.00879699999999999926 * s_radiusulnar_l_x, -0.23584099999999999508 * s_radiusulnar_l_y, -0.01361000000000000057 * s_radiusulnar_l_z), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *hand_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	model->addJoint(radius_hand_l);
	
	
	
	// Definition of contacts.
	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s1_r;
	SmoothSphereHalfSpaceForce_s1_r = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s1_r", *calcn_r, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s1_r_location(0.00190115788407966006, -0.01000000000000000021, -0.00382630379623307999);
	SmoothSphereHalfSpaceForce_s1_r->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s1_r_location);
	double SmoothSphereHalfSpaceForce_s1_r_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s1_r->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s1_r_radius);
	SmoothSphereHalfSpaceForce_s1_r->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s1_r->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s1_r->set_stiffness(1600000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s1_r->set_dissipation(2.0000000000000000000);
	SmoothSphereHalfSpaceForce_s1_r->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s1_r->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s1_r->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s1_r->set_transition_velocity(0.2000000000000001110);
	SmoothSphereHalfSpaceForce_s1_r->connectSocket_sphere_frame(*calcn_r);
	SmoothSphereHalfSpaceForce_s1_r->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s1_r);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s2_r;
	SmoothSphereHalfSpaceForce_s2_r = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s2_r", *calcn_r, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s2_r_location(0.14838639994206301309, -0.01000000000000000021, -0.02871342205265400155);
	SmoothSphereHalfSpaceForce_s2_r->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s2_r_location);
	double SmoothSphereHalfSpaceForce_s2_r_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s2_r->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s2_r_radius);
	SmoothSphereHalfSpaceForce_s2_r->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s2_r->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s2_r->set_stiffness(1600000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s2_r->set_dissipation(2.0000000000000000000);
	SmoothSphereHalfSpaceForce_s2_r->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s2_r->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s2_r->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s2_r->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s2_r->connectSocket_sphere_frame(*calcn_r);
	SmoothSphereHalfSpaceForce_s2_r->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s2_r);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s3_r;
	SmoothSphereHalfSpaceForce_s3_r = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s3_r", *calcn_r, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s3_r_location(0.13300117060705099470, -0.01000000000000000021, 0.05163624734495660118);
	SmoothSphereHalfSpaceForce_s3_r->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s3_r_location);
	double SmoothSphereHalfSpaceForce_s3_r_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s3_r->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s3_r_radius);
	SmoothSphereHalfSpaceForce_s3_r->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s3_r->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s3_r->set_stiffness(1600000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s3_r->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s3_r->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s3_r->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s3_r->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s3_r->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s3_r->connectSocket_sphere_frame(*calcn_r);
	SmoothSphereHalfSpaceForce_s3_r->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s3_r);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s4_r;
	SmoothSphereHalfSpaceForce_s4_r = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s4_r", *calcn_r, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s4_r_location(0.06623466619916350273, -0.01000000000000000021, 0.02636416067416980091);
	SmoothSphereHalfSpaceForce_s4_r->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s4_r_location);
	double SmoothSphereHalfSpaceForce_s4_r_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s4_r->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s4_r_radius);
	SmoothSphereHalfSpaceForce_s4_r->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s4_r->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s4_r->set_stiffness(1600000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s4_r->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s4_r->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s4_r->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s4_r->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s4_r->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s4_r->connectSocket_sphere_frame(*calcn_r);
	SmoothSphereHalfSpaceForce_s4_r->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s4_r);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s5_r;
	SmoothSphereHalfSpaceForce_s5_r = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s5_r", *toes_r, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s5_r_location(0.05999999999999999778, -0.01000000000000000021, -0.01876030846191769838);
	SmoothSphereHalfSpaceForce_s5_r->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s5_r_location);
	double SmoothSphereHalfSpaceForce_s5_r_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s5_r->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s5_r_radius);
	SmoothSphereHalfSpaceForce_s5_r->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s5_r->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s5_r->set_stiffness(1600000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s5_r->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s5_r->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s5_r->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s5_r->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s5_r->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s5_r->connectSocket_sphere_frame(*toes_r);
	SmoothSphereHalfSpaceForce_s5_r->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s5_r);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s6_r;
	SmoothSphereHalfSpaceForce_s6_r = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s6_r", *toes_r, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s6_r_location(0.04499999999999999833, -0.01000000000000000021, 0.06185695675496519913);
	SmoothSphereHalfSpaceForce_s6_r->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s6_r_location);
	double SmoothSphereHalfSpaceForce_s6_r_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s6_r->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s6_r_radius);
	SmoothSphereHalfSpaceForce_s6_r->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s6_r->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s6_r->set_stiffness(1600000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s6_r->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s6_r->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s6_r->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s6_r->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s6_r->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s6_r->connectSocket_sphere_frame(*toes_r);
	SmoothSphereHalfSpaceForce_s6_r->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s6_r);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s1_l;
	SmoothSphereHalfSpaceForce_s1_l = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s1_l", *calcn_l, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s1_l_location(0.00190115788407966006, -0.01000000000000000021, 0.00382630379623307999);
	SmoothSphereHalfSpaceForce_s1_l->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s1_l_location);
	double SmoothSphereHalfSpaceForce_s1_l_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s1_l->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s1_l_radius);
	SmoothSphereHalfSpaceForce_s1_l->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s1_l->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s1_l->set_stiffness(1600000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s1_l->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s1_l->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s1_l->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s1_l->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s1_l->set_transition_velocity(0.2000000000000001110);
	SmoothSphereHalfSpaceForce_s1_l->connectSocket_sphere_frame(*calcn_l);
	SmoothSphereHalfSpaceForce_s1_l->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s1_l);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s2_l;
	SmoothSphereHalfSpaceForce_s2_l = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s2_l", *calcn_l, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s2_l_location(0.14838639994206301309, -0.01000000000000000021, 0.02871342205265400155);
	SmoothSphereHalfSpaceForce_s2_l->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s2_l_location);
	double SmoothSphereHalfSpaceForce_s2_l_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s2_l->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s2_l_radius);
	SmoothSphereHalfSpaceForce_s2_l->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s2_l->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s2_l->set_stiffness(1600000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s2_l->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s2_l->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s2_l->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s2_l->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s2_l->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s2_l->connectSocket_sphere_frame(*calcn_l);
	SmoothSphereHalfSpaceForce_s2_l->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s2_l);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s3_l;
	SmoothSphereHalfSpaceForce_s3_l = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s3_l", *calcn_l, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s3_l_location(0.13300117060705099470, -0.01000000000000000021, -0.05163624734495660118);
	SmoothSphereHalfSpaceForce_s3_l->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s3_l_location);
	double SmoothSphereHalfSpaceForce_s3_l_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s3_l->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s3_l_radius);
	SmoothSphereHalfSpaceForce_s3_l->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s3_l->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s3_l->set_stiffness(1600000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s3_l->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s3_l->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s3_l->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s3_l->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s3_l->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s3_l->connectSocket_sphere_frame(*calcn_l);
	SmoothSphereHalfSpaceForce_s3_l->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s3_l);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s4_l;
	SmoothSphereHalfSpaceForce_s4_l = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s4_l", *calcn_l, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s4_l_location(0.06623466619916350273, -0.01000000000000000021, -0.02636416067416980091);
	SmoothSphereHalfSpaceForce_s4_l->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s4_l_location);
	double SmoothSphereHalfSpaceForce_s4_l_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s4_l->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s4_l_radius);
	SmoothSphereHalfSpaceForce_s4_l->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s4_l->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s4_l->set_stiffness(1600000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s4_l->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s4_l->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s4_l->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s4_l->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s4_l->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s4_l->connectSocket_sphere_frame(*calcn_l);
	SmoothSphereHalfSpaceForce_s4_l->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s4_l);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s5_l;
	SmoothSphereHalfSpaceForce_s5_l = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s5_l", *toes_l, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s5_l_location(0.05999999999999999778, -0.01000000000000000021, 0.01876030846191769838);
	SmoothSphereHalfSpaceForce_s5_l->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s5_l_location);
	double SmoothSphereHalfSpaceForce_s5_l_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s5_l->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s5_l_radius);
	SmoothSphereHalfSpaceForce_s5_l->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s5_l->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s5_l->set_stiffness(1600000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s5_l->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s5_l->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s5_l->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s5_l->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s5_l->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s5_l->connectSocket_sphere_frame(*toes_l);
	SmoothSphereHalfSpaceForce_s5_l->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s5_l);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s6_l;
	SmoothSphereHalfSpaceForce_s6_l = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s6_l", *toes_l, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s6_l_location(0.04499999999999999833, -0.01000000000000000021, -0.06185695675496519913);
	SmoothSphereHalfSpaceForce_s6_l->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s6_l_location);
	double SmoothSphereHalfSpaceForce_s6_l_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s6_l->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s6_l_radius);
	SmoothSphereHalfSpaceForce_s6_l->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s6_l->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s6_l->set_stiffness(1600000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s6_l->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s6_l->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s6_l->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s6_l->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s6_l->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s6_l->connectSocket_sphere_frame(*toes_l);
	SmoothSphereHalfSpaceForce_s6_l->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s6_l);

	// Initialize system.
	SimTK::State* state;
	state = new State(model->initSystem());

	// States and controls.
	T ua[NU];
	Vector QsUs(NX);
	/// States
	for (int i = 0; i < NX; ++i) QsUs[i] = x[i];
	/// Controls
	/// OpenSim and Simbody have different state orders.
	auto indicesOSInSimbody = getIndicesOSInSimbody(*model);
	for (int i = 0; i < NU; ++i) ua[i] = u[indicesOSInSimbody[i]];

	// Set state variables and realize.
	model->setStateVariableValues(*state, QsUs);
	model->realizeVelocity(*state);

	// Compute residual forces.
	/// Set appliedMobilityForces (# mobilities).
	Vector appliedMobilityForces(nCoordinates);
	appliedMobilityForces.setToZero();
	/// Set appliedBodyForces (# bodies + ground).
	Vector_<SpatialVec> appliedBodyForces;
	int nbodies = model->getBodySet().getSize() + 1;
	appliedBodyForces.resize(nbodies);
	appliedBodyForces.setToZero();
	/// Set gravity.
	Vec3 gravity(0);
	gravity[1] = -9.80664999999999942304;
	/// Add weights to appliedBodyForces.
	for (int i = 0; i < model->getBodySet().getSize(); ++i) {
		model->getMatterSubsystem().addInStationForce(*state,
			model->getBodySet().get(i).getMobilizedBodyIndex(),
			model->getBodySet().get(i).getMassCenter(),
			model->getBodySet().get(i).getMass()*gravity, appliedBodyForces);
	}

	/// Add contact forces to appliedBodyForces.
	Array<osim_double_adouble> Force_0 = SmoothSphereHalfSpaceForce_s1_r->getRecordValues(*state);
	SpatialVec GRF_0;
	GRF_0[0] = Vec3(Force_0[3], Force_0[4], Force_0[5]);
	GRF_0[1] = Vec3(Force_0[0], Force_0[1], Force_0[2]);
	int c_idx_0 = model->getBodySet().get("calcn_r").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_0] += GRF_0;

	Array<osim_double_adouble> Force_1 = SmoothSphereHalfSpaceForce_s2_r->getRecordValues(*state);
	SpatialVec GRF_1;
	GRF_1[0] = Vec3(Force_1[3], Force_1[4], Force_1[5]);
	GRF_1[1] = Vec3(Force_1[0], Force_1[1], Force_1[2]);
	int c_idx_1 = model->getBodySet().get("calcn_r").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_1] += GRF_1;

	Array<osim_double_adouble> Force_2 = SmoothSphereHalfSpaceForce_s3_r->getRecordValues(*state);
	SpatialVec GRF_2;
	GRF_2[0] = Vec3(Force_2[3], Force_2[4], Force_2[5]);
	GRF_2[1] = Vec3(Force_2[0], Force_2[1], Force_2[2]);
	int c_idx_2 = model->getBodySet().get("calcn_r").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_2] += GRF_2;

	Array<osim_double_adouble> Force_3 = SmoothSphereHalfSpaceForce_s4_r->getRecordValues(*state);
	SpatialVec GRF_3;
	GRF_3[0] = Vec3(Force_3[3], Force_3[4], Force_3[5]);
	GRF_3[1] = Vec3(Force_3[0], Force_3[1], Force_3[2]);
	int c_idx_3 = model->getBodySet().get("calcn_r").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_3] += GRF_3;

	Array<osim_double_adouble> Force_4 = SmoothSphereHalfSpaceForce_s5_r->getRecordValues(*state);
	SpatialVec GRF_4;
	GRF_4[0] = Vec3(Force_4[3], Force_4[4], Force_4[5]);
	GRF_4[1] = Vec3(Force_4[0], Force_4[1], Force_4[2]);
	int c_idx_4 = model->getBodySet().get("toes_r").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_4] += GRF_4;

	Array<osim_double_adouble> Force_5 = SmoothSphereHalfSpaceForce_s6_r->getRecordValues(*state);
	SpatialVec GRF_5;
	GRF_5[0] = Vec3(Force_5[3], Force_5[4], Force_5[5]);
	GRF_5[1] = Vec3(Force_5[0], Force_5[1], Force_5[2]);
	int c_idx_5 = model->getBodySet().get("toes_r").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_5] += GRF_5;

	Array<osim_double_adouble> Force_6 = SmoothSphereHalfSpaceForce_s1_l->getRecordValues(*state);
	SpatialVec GRF_6;
	GRF_6[0] = Vec3(Force_6[3], Force_6[4], Force_6[5]);
	GRF_6[1] = Vec3(Force_6[0], Force_6[1], Force_6[2]);
	int c_idx_6 = model->getBodySet().get("calcn_l").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_6] += GRF_6;

	Array<osim_double_adouble> Force_7 = SmoothSphereHalfSpaceForce_s2_l->getRecordValues(*state);
	SpatialVec GRF_7;
	GRF_7[0] = Vec3(Force_7[3], Force_7[4], Force_7[5]);
	GRF_7[1] = Vec3(Force_7[0], Force_7[1], Force_7[2]);
	int c_idx_7 = model->getBodySet().get("calcn_l").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_7] += GRF_7;

	Array<osim_double_adouble> Force_8 = SmoothSphereHalfSpaceForce_s3_l->getRecordValues(*state);
	SpatialVec GRF_8;
	GRF_8[0] = Vec3(Force_8[3], Force_8[4], Force_8[5]);
	GRF_8[1] = Vec3(Force_8[0], Force_8[1], Force_8[2]);
	int c_idx_8 = model->getBodySet().get("calcn_l").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_8] += GRF_8;

	Array<osim_double_adouble> Force_9 = SmoothSphereHalfSpaceForce_s4_l->getRecordValues(*state);
	SpatialVec GRF_9;
	GRF_9[0] = Vec3(Force_9[3], Force_9[4], Force_9[5]);
	GRF_9[1] = Vec3(Force_9[0], Force_9[1], Force_9[2]);
	int c_idx_9 = model->getBodySet().get("calcn_l").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_9] += GRF_9;

	Array<osim_double_adouble> Force_10 = SmoothSphereHalfSpaceForce_s5_l->getRecordValues(*state);
	SpatialVec GRF_10;
	GRF_10[0] = Vec3(Force_10[3], Force_10[4], Force_10[5]);
	GRF_10[1] = Vec3(Force_10[0], Force_10[1], Force_10[2]);
	int c_idx_10 = model->getBodySet().get("toes_l").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_10] += GRF_10;

	Array<osim_double_adouble> Force_11 = SmoothSphereHalfSpaceForce_s6_l->getRecordValues(*state);
	SpatialVec GRF_11;
	GRF_11[0] = Vec3(Force_11[3], Force_11[4], Force_11[5]);
	GRF_11[1] = Vec3(Force_11[0], Force_11[1], Force_11[2]);
	int c_idx_11 = model->getBodySet().get("toes_l").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_11] += GRF_11;

	/// knownUdot.
	Vector knownUdot(nCoordinates);
	knownUdot.setToZero();
	for (int i = 0; i < nCoordinates; ++i) knownUdot[i] = ua[i];
	/// Calculate residual forces.
	Vector residualMobilityForces(nCoordinates);
	residualMobilityForces.setToZero();
	    model->getMatterSubsystem().calcResidualForceIgnoringConstraints(*state,
	  	appliedMobilityForces, appliedBodyForces, knownUdot, residualMobilityForces);

	/// Body origins.
	Vec3 pelvis_or = pelvis->getPositionInGround(*state);
	Vec3 femur_r_or = femur_r->getPositionInGround(*state);
	Vec3 tibia_r_or = tibia_r->getPositionInGround(*state);
	Vec3 talus_r_or = talus_r->getPositionInGround(*state);
	Vec3 calcn_r_or = calcn_r->getPositionInGround(*state);
	Vec3 toes_r_or = toes_r->getPositionInGround(*state);
	Vec3 femur_l_or = femur_l->getPositionInGround(*state);
	Vec3 tibia_l_or = tibia_l->getPositionInGround(*state);
	Vec3 talus_l_or = talus_l->getPositionInGround(*state);
	Vec3 calcn_l_or = calcn_l->getPositionInGround(*state);
	Vec3 toes_l_or = toes_l->getPositionInGround(*state);
	Vec3 torso_or = torso->getPositionInGround(*state);
	Vec3 humerus_r_or = humerus_r->getPositionInGround(*state);
	Vec3 ulna_r_or = ulna_r->getPositionInGround(*state);
	Vec3 radius_r_or = radius_r->getPositionInGround(*state);
	Vec3 hand_r_or = hand_r->getPositionInGround(*state);
	Vec3 humerus_l_or = humerus_l->getPositionInGround(*state);
	Vec3 ulna_l_or = ulna_l->getPositionInGround(*state);
	Vec3 radius_l_or = radius_l->getPositionInGround(*state);
	Vec3 hand_l_or = hand_l->getPositionInGround(*state);

	/// Ground reaction forces.
	Vec3 GRF_r(0), GRF_l(0);
	GRF_r += GRF_0[1];
	GRF_r += GRF_1[1];
	GRF_r += GRF_2[1];
	GRF_r += GRF_3[1];
	GRF_r += GRF_4[1];
	GRF_r += GRF_5[1];
	GRF_l += GRF_6[1];
	GRF_l += GRF_7[1];
	GRF_l += GRF_8[1];
	GRF_l += GRF_9[1];
	GRF_l += GRF_10[1];
	GRF_l += GRF_11[1];

	/// Ground reaction moments.
	Vec3 GRM_r(0), GRM_l(0);
	Vec3 normal(0, 1, 0);

	SimTK::Transform TR_GB_calcn_r = calcn_r->getMobilizedBody().getBodyTransform(*state);
	Vec3 SmoothSphereHalfSpaceForce_s1_r_location_G = calcn_r->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s1_r_location);
	Vec3 SmoothSphereHalfSpaceForce_s1_r_locationCP_G = SmoothSphereHalfSpaceForce_s1_r_location_G - SmoothSphereHalfSpaceForce_s1_r_radius * normal;
	Vec3 locationCP_G_adj_0 = SmoothSphereHalfSpaceForce_s1_r_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s1_r_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s1_r_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_0, *calcn_r);
	Vec3 GRM_0 = (TR_GB_calcn_r*SmoothSphereHalfSpaceForce_s1_r_locationCP_B) % GRF_0[1];
	GRM_r += GRM_0;

	Vec3 SmoothSphereHalfSpaceForce_s2_r_location_G = calcn_r->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s2_r_location);
	Vec3 SmoothSphereHalfSpaceForce_s2_r_locationCP_G = SmoothSphereHalfSpaceForce_s2_r_location_G - SmoothSphereHalfSpaceForce_s2_r_radius * normal;
	Vec3 locationCP_G_adj_1 = SmoothSphereHalfSpaceForce_s2_r_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s2_r_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s2_r_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_1, *calcn_r);
	Vec3 GRM_1 = (TR_GB_calcn_r*SmoothSphereHalfSpaceForce_s2_r_locationCP_B) % GRF_1[1];
	GRM_r += GRM_1;

	Vec3 SmoothSphereHalfSpaceForce_s3_r_location_G = calcn_r->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s3_r_location);
	Vec3 SmoothSphereHalfSpaceForce_s3_r_locationCP_G = SmoothSphereHalfSpaceForce_s3_r_location_G - SmoothSphereHalfSpaceForce_s3_r_radius * normal;
	Vec3 locationCP_G_adj_2 = SmoothSphereHalfSpaceForce_s3_r_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s3_r_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s3_r_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_2, *calcn_r);
	Vec3 GRM_2 = (TR_GB_calcn_r*SmoothSphereHalfSpaceForce_s3_r_locationCP_B) % GRF_2[1];
	GRM_r += GRM_2;

	Vec3 SmoothSphereHalfSpaceForce_s4_r_location_G = calcn_r->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s4_r_location);
	Vec3 SmoothSphereHalfSpaceForce_s4_r_locationCP_G = SmoothSphereHalfSpaceForce_s4_r_location_G - SmoothSphereHalfSpaceForce_s4_r_radius * normal;
	Vec3 locationCP_G_adj_3 = SmoothSphereHalfSpaceForce_s4_r_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s4_r_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s4_r_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_3, *calcn_r);
	Vec3 GRM_3 = (TR_GB_calcn_r*SmoothSphereHalfSpaceForce_s4_r_locationCP_B) % GRF_3[1];
	GRM_r += GRM_3;

	SimTK::Transform TR_GB_toes_r = toes_r->getMobilizedBody().getBodyTransform(*state);
	Vec3 SmoothSphereHalfSpaceForce_s5_r_location_G = toes_r->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s5_r_location);
	Vec3 SmoothSphereHalfSpaceForce_s5_r_locationCP_G = SmoothSphereHalfSpaceForce_s5_r_location_G - SmoothSphereHalfSpaceForce_s5_r_radius * normal;
	Vec3 locationCP_G_adj_4 = SmoothSphereHalfSpaceForce_s5_r_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s5_r_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s5_r_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_4, *toes_r);
	Vec3 GRM_4 = (TR_GB_toes_r*SmoothSphereHalfSpaceForce_s5_r_locationCP_B) % GRF_4[1];
	GRM_r += GRM_4;

	Vec3 SmoothSphereHalfSpaceForce_s6_r_location_G = toes_r->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s6_r_location);
	Vec3 SmoothSphereHalfSpaceForce_s6_r_locationCP_G = SmoothSphereHalfSpaceForce_s6_r_location_G - SmoothSphereHalfSpaceForce_s6_r_radius * normal;
	Vec3 locationCP_G_adj_5 = SmoothSphereHalfSpaceForce_s6_r_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s6_r_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s6_r_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_5, *toes_r);
	Vec3 GRM_5 = (TR_GB_toes_r*SmoothSphereHalfSpaceForce_s6_r_locationCP_B) % GRF_5[1];
	GRM_r += GRM_5;

	SimTK::Transform TR_GB_calcn_l = calcn_l->getMobilizedBody().getBodyTransform(*state);
	Vec3 SmoothSphereHalfSpaceForce_s1_l_location_G = calcn_l->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s1_l_location);
	Vec3 SmoothSphereHalfSpaceForce_s1_l_locationCP_G = SmoothSphereHalfSpaceForce_s1_l_location_G - SmoothSphereHalfSpaceForce_s1_l_radius * normal;
	Vec3 locationCP_G_adj_6 = SmoothSphereHalfSpaceForce_s1_l_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s1_l_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s1_l_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_6, *calcn_l);
	Vec3 GRM_6 = (TR_GB_calcn_l*SmoothSphereHalfSpaceForce_s1_l_locationCP_B) % GRF_6[1];
	GRM_l += GRM_6;

	Vec3 SmoothSphereHalfSpaceForce_s2_l_location_G = calcn_l->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s2_l_location);
	Vec3 SmoothSphereHalfSpaceForce_s2_l_locationCP_G = SmoothSphereHalfSpaceForce_s2_l_location_G - SmoothSphereHalfSpaceForce_s2_l_radius * normal;
	Vec3 locationCP_G_adj_7 = SmoothSphereHalfSpaceForce_s2_l_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s2_l_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s2_l_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_7, *calcn_l);
	Vec3 GRM_7 = (TR_GB_calcn_l*SmoothSphereHalfSpaceForce_s2_l_locationCP_B) % GRF_7[1];
	GRM_l += GRM_7;

	Vec3 SmoothSphereHalfSpaceForce_s3_l_location_G = calcn_l->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s3_l_location);
	Vec3 SmoothSphereHalfSpaceForce_s3_l_locationCP_G = SmoothSphereHalfSpaceForce_s3_l_location_G - SmoothSphereHalfSpaceForce_s3_l_radius * normal;
	Vec3 locationCP_G_adj_8 = SmoothSphereHalfSpaceForce_s3_l_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s3_l_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s3_l_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_8, *calcn_l);
	Vec3 GRM_8 = (TR_GB_calcn_l*SmoothSphereHalfSpaceForce_s3_l_locationCP_B) % GRF_8[1];
	GRM_l += GRM_8;

	Vec3 SmoothSphereHalfSpaceForce_s4_l_location_G = calcn_l->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s4_l_location);
	Vec3 SmoothSphereHalfSpaceForce_s4_l_locationCP_G = SmoothSphereHalfSpaceForce_s4_l_location_G - SmoothSphereHalfSpaceForce_s4_l_radius * normal;
	Vec3 locationCP_G_adj_9 = SmoothSphereHalfSpaceForce_s4_l_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s4_l_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s4_l_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_9, *calcn_l);
	Vec3 GRM_9 = (TR_GB_calcn_l*SmoothSphereHalfSpaceForce_s4_l_locationCP_B) % GRF_9[1];
	GRM_l += GRM_9;

	SimTK::Transform TR_GB_toes_l = toes_l->getMobilizedBody().getBodyTransform(*state);
	Vec3 SmoothSphereHalfSpaceForce_s5_l_location_G = toes_l->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s5_l_location);
	Vec3 SmoothSphereHalfSpaceForce_s5_l_locationCP_G = SmoothSphereHalfSpaceForce_s5_l_location_G - SmoothSphereHalfSpaceForce_s5_l_radius * normal;
	Vec3 locationCP_G_adj_10 = SmoothSphereHalfSpaceForce_s5_l_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s5_l_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s5_l_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_10, *toes_l);
	Vec3 GRM_10 = (TR_GB_toes_l*SmoothSphereHalfSpaceForce_s5_l_locationCP_B) % GRF_10[1];
	GRM_l += GRM_10;

	Vec3 SmoothSphereHalfSpaceForce_s6_l_location_G = toes_l->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s6_l_location);
	Vec3 SmoothSphereHalfSpaceForce_s6_l_locationCP_G = SmoothSphereHalfSpaceForce_s6_l_location_G - SmoothSphereHalfSpaceForce_s6_l_radius * normal;
	Vec3 locationCP_G_adj_11 = SmoothSphereHalfSpaceForce_s6_l_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s6_l_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s6_l_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_11, *toes_l);
	Vec3 GRM_11 = (TR_GB_toes_l*SmoothSphereHalfSpaceForce_s6_l_locationCP_B) % GRF_11[1];
	GRM_l += GRM_11;

	/// Outputs.
	/// Residual forces (OpenSim and Simbody have different state orders).
	auto indicesSimbodyInOS = getIndicesSimbodyInOS(*model);
	for (int i = 0; i < NU; ++i) res[0][i] =
		value<T>(residualMobilityForces[indicesSimbodyInOS[i]]);
	/// Ground reaction forces.
	for (int i = 0; i < 3; ++i) res[0][i + NU + 0] = value<T>(GRF_r[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 3] = value<T>(GRF_l[i]);
	/// Ground reaction moments.
	for (int i = 0; i < 3; ++i) res[0][i + NU + 6] = value<T>(GRM_r[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 9] = value<T>(GRM_l[i]);
	/// Body origins.
	for (int i = 0; i < 3; ++i) res[0][i + NU + 12] = value<T>(pelvis_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 15] = value<T>(femur_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 18] = value<T>(tibia_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 21] = value<T>(talus_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 24] = value<T>(calcn_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 27] = value<T>(toes_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 30] = value<T>(femur_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 33] = value<T>(tibia_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 36] = value<T>(talus_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 39] = value<T>(calcn_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 42] = value<T>(toes_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 45] = value<T>(torso_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 48] = value<T>(humerus_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 51] = value<T>(ulna_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 54] = value<T>(radius_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 57] = value<T>(hand_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 60] = value<T>(humerus_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 63] = value<T>(ulna_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 66] = value<T>(radius_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + NU + 69] = value<T>(hand_l_or[i]);
	res[0][103] = value<T>(total_mass_scaling_step_1/total_original_mass);
	res[0][104] = value<T>(1.30*(humerus_r_or[1] - toes_r_or[1]));
	return 0;
}

int main() {
	Recorder x[NX];
	Recorder u[NU];
	Recorder s[NS];
	Recorder tau[NR];
	for (int i = 0; i < NX; ++i) x[i] <<= 1;
	for (int i = 0; i < NU; ++i) u[i] <<= 1;
	for (int i = 0; i < NS; ++i) s[i] <<= 1;
	const Recorder* Recorder_arg[n_in] = { x,u,s };
	Recorder* Recorder_res[n_out] = { tau };
	F_generic<Recorder>(Recorder_arg, Recorder_res);
	double res[NR];
	for (int i = 0; i < NR; ++i) Recorder_res[0][i] >>= res[i];
	Recorder::stop_recording();
	return 0;
}