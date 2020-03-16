/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::default_random_engine;
using std::normal_distribution;
using std::discrete_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * DONE: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  particles.resize(num_particles); 
  weights.resize(num_particles);

  // Initialize all particles to first position with injected Gaussian noise
  default_random_engine gen;
  normal_distribution<double> x_noisy(x, std[0]);
  normal_distribution<double> y_noisy(y, std[1]);
  normal_distribution<double> theta_noisy(theta, std[2]);
  
  for (int i=0; i < num_particles; i++) {
  	particles[i].id = i;
  	particles[i].x = x_noisy(gen);
  	particles[i].y = y_noisy(gen);
  	particles[i].theta = theta_noisy(gen);
	particles[i].weight = 1.0;

	weights[i] = 1.0; // Set all weights initially to 1.
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * DONE: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine gen;
  normal_distribution<double> x_noise(0., std_pos[0]);
  normal_distribution<double> y_noise(0., std_pos[1]);
  normal_distribution<double> theta_noise(0., std_pos[2]);

  for (int i=0; i<num_particles; i++) {
	double theta_o = particles[i].theta;
	double x_o = particles[i].x;
	double y_o = particles[i].y;

	double xt, yt, theta_t;
	if (yaw_rate != 0.0) {
		xt = x_o + (velocity / yaw_rate) * \
					(sin(theta_o + yaw_rate*delta_t) - sin(theta_o));
		yt = y_o + (velocity / yaw_rate) * \
					(cos(theta_o) - cos(theta_o + yaw_rate*delta_t));
	    theta_t = theta_o + yaw_rate * delta_t;	
	} else {
		xt = x_o + velocity * sin(theta_o);
		yt = y_o + velocity * cos(theta_o);
		theta_t = theta_o;
	}

	// Save new particle position with injected noise according to std_pos[]
	particles[i].x = xt + x_noise(gen);
	particles[i].y = yt + y_noise(gen);
	particles[i].theta = theta_t + theta_noise(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * DONE: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  // Loop over observations. Loop over all predicted landmarks to find closest
  // and assign id of closest to current observation
  for (int i=0; i < observations.size(); i++) {
	LandmarkObs obs_i = observations[i];
	double obs_x = obs_i.x;
	double obs_y = obs_i.y;
	
	double min_dist = 0;
	int min_id = 0;
	for (int j=0; j < predicted.size(); j++) {
		LandmarkObs pred_j = predicted[j];
		// Negative id fields mean this prediction is outside the sensor range
		// and this landmark should be ignored
		if (pred_j.id < 0) {
			continue;
		}
		double pred_dist = dist(obs_x, obs_y, pred_j.x, pred_j.y);
		if (j==0) {
			min_dist = pred_dist;
			min_id = pred_j.id;
		} else {
			if (pred_dist < min_dist) {
				min_dist = pred_dist;		
				min_id = pred_j.id;	
			}
		}
	}
	// Save obsevation's landmark id as min_id, the id of closest predicted landmark
	obs_i.id = min_id;
  }
}

double _gaussian_probability(double sigma_x, double sigma_y,
							double obs_x, double obs_y,
							double mu_x, double mu_y) {
  // Calculate the norm term
  double gauss_norm;
  gauss_norm = 1. / (2. * M_PI * sigma_x * sigma_y);

  // Calculate the exponent
  double exponent = (pow(obs_x - mu_x, 2) / (2 * pow(sigma_x, 2)))
	  + (pow(obs_y - mu_y, 2) / (2 * pow(sigma_y, 2)));

  double weight = gauss_norm * exp(-exponent);

  return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * DONE: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // For each particle:
  // 1) Transform the observations to map coordinates
  // 2) Associate the observations with nearest landmark
  // 3) Calculate the probabilities of all observations according to assoc.
  // 4) Update particle's weight according to product-reduction of probabilities
  for (int i=0; i<num_particles; i++) {
	Particle part_i = particles[i];
  	double xp = part_i.x;
	double yp = part_i.y;
	double theta_p = part_i.theta;
	
	double weight_p = 1.0; 

	// Convert all observations to map-frame according to particle pose
	vector<LandmarkObs> obs_map;
  	for (int j=0; j<observations.size(); j++){
		LandmarkObs li_obs = observations[j];
	    int obs_id = li_obs.id;

		// Create copy of observation (stack-alloc'd)
		LandmarkObs obs_m;
		obs_m.id = obs_id;
		obs_m.x = li_obs.x * cos(theta_p) - li_obs.y * sin(theta_p) + xp;
	    obs_m.y = li_obs.x * sin(theta_p) + li_obs.y * cos(theta_p) + yp;

		// Push map-frame observation
		obs_map.push_back(obs_m);
	}	

	// Create vector of predicted measurements (ordered) to all landmarks
	// from particle pose
	vector<LandmarkObs> pred_i; // Will be ordered by id
	// NOTE:This for-loop assumes landmark_list is ordered by id
	for (Map::single_landmark_s map_landmark : map_landmarks.landmark_list) {
		// Calculate the distance to determine if within sensor_range
		LandmarkObs pred_obs_i;
		double distance = dist(xp, yp, map_landmark.x_f, map_landmark.y_f);
		if (distance > sensor_range) {
			// Mark this landmark as out of range
			pred_obs_i.id = -map_landmark.id_i;
		} else {
			pred_obs_i.id = map_landmark.id_i;
		}
		pred_obs_i.x = map_landmark.x_f - xp;
		pred_obs_i.y = map_landmark.y_f - yp;
		
		// Push predicted landmark measurement (in order by id) to predicted measurements
		pred_i.push_back(pred_obs_i);
	}

	// Get dataAssociations
	dataAssociation(pred_i, obs_map); // obs_map now has final landmark id

	// SetAssociations according to obs_map
	vector<int> obs_landmarks;
	vector<double> obs_x;
	vector<double> obs_y;
	for (LandmarkObs lo : obs_map) {
		obs_landmarks.push_back(lo.id);
		obs_x.push_back(lo.x);
		obs_y.push_back(lo.y);
	}
	SetAssociations(part_i, obs_landmarks, obs_x, obs_y);

	// Determine probability of each observed landmark's observation
	// and total likelihood of this particle's observations
	for (LandmarkObs lo : obs_map) {
		// Get associated predicted measurement
		LandmarkObs assoc_pred = pred_i[lo.id];

		// Calculate probability
		double obs_prob;
		obs_prob = _gaussian_probability(std_landmark[0], std_landmark[1],
				lo.x, lo.y, assoc_pred.x, assoc_pred.y);
		weight_p *= obs_prob;
	}

	// Save new weight
	part_i.weight = weight_p;
	weights[i] = part_i.weight;
  }

}

void ParticleFilter::resample() {
  /**
   * DONE: Resample particles with replacement with probability proportional
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  // Initialize a discrete distribution using current weights
  default_random_engine gen;
  discrete_distribution<int> bag_of_particles(weights.begin(), weights.end());
    
  // Re-sample with bag_of_particles (with replacement) num_particles-many
  // particles to keep
  vector<Particle> sampled_particles;
  for (int i=0; i<num_particles; i++) {
	int part_idx = bag_of_particles(gen);
	sampled_particles.push_back(particles[part_idx]);
  }

  // Lastly, replace 'particles' with 'sampled_particles'
  particles = sampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
