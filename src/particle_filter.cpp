/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cassert>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  if(!is_initialized)
  {
    default_random_engine gen;

    num_particles = 100;

    //create normal (Gaussian) distributions
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (unsigned i = 0; i < num_particles; ++i) {
      //Create a particle and sample from these normal distributions
      Particle sample_particle;

      sample_particle.id = i;

      sample_particle.x = dist_x(gen);
      sample_particle.y = dist_y(gen);
      sample_particle.theta = dist_theta(gen);

      sample_particle.weight = 1.0;

      particles.push_back(sample_particle);
      weights.push_back(sample_particle.weight);
    }

    is_initialized = true;
    return;

  }
  else
  {
    return;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  //avoid division by zero
  default_random_engine gen;

  for (unsigned i = 0; i < num_particles; ++i)
  {
    Particle current_particle = particles[i];
    if (fabs(yaw_rate) > 0.0001) {
      current_particle.x = current_particle.x +
          velocity/yaw_rate * ( sin (current_particle.theta + yaw_rate*delta_t) - sin(current_particle.theta));

      current_particle.y = current_particle.y +
          velocity/yaw_rate * ( cos(current_particle.theta) - cos(current_particle.theta+yaw_rate*delta_t) );
    }
    else {
      current_particle.x = current_particle.x + velocity*delta_t*cos(current_particle.theta);
      current_particle.y = current_particle.y + velocity*delta_t*sin(current_particle.theta);
    }
    current_particle.theta = current_particle.theta + yaw_rate*delta_t;

    //add noise
    //create normal (Gaussian) distributions
    normal_distribution<double> dist_x(current_particle.x, std_pos[0]);
    normal_distribution<double> dist_y(current_particle.y, std_pos[1]);
    normal_distribution<double> dist_theta(current_particle.theta, std_pos[2]);

    current_particle.x = dist_x(gen);
    current_particle.y = dist_y(gen);
    current_particle.theta = dist_theta(gen);

    particles[i] = current_particle;
  }

}

LandmarkObs transformObservation(Particle particle, LandmarkObs LocalObs)
{
  LandmarkObs transformed_obs;
  transformed_obs.x = LocalObs.x*cos(particle.theta) - LocalObs.y*sin(particle.theta) + particle.x;
  transformed_obs.y = LocalObs.x*sin(particle.theta) + LocalObs.y*cos(particle.theta) + particle.y;

  return transformed_obs;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{

}

int findAssociation(LandmarkObs observation, Map map_landmarks) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  double min_distance = numeric_limits<double>::max();
  int closest_id = -1;

  for (unsigned int i = 0; i < map_landmarks.landmark_list.size(); ++i)
  {
    // Declare single_landmark:
    Map::single_landmark_s single_landmark_temp = map_landmarks.landmark_list[i];

    double distance = dist(observation.x, observation.y, single_landmark_temp.x_f, single_landmark_temp.y_f);
    if(distance < min_distance)
    {
      min_distance = distance;
      closest_id = i;
    }
  }
  return closest_id;
}

double MVGaussian(Map::single_landmark_s predicted_landmark, double std_landmark[], LandmarkObs observation)
{
  double exp_arg = -0.5 * ( pow((observation.x-predicted_landmark.x_f),2)/pow(std_landmark[0],2)+
      pow((observation.y-predicted_landmark.y_f),2)/pow(std_landmark[1],2) );

  double prob = exp(exp_arg)/(2.0*M_PI*std_landmark[0]*std_landmark[1]);

  return prob;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  for (unsigned i = 0; i < num_particles; ++i)
  {
    Particle current_particle = particles[i];
    double current_weight = 1.0;
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
    //transform observations to map coordinates
    //dataAssociation
    //multiply likelihood of each landmark

    for (unsigned int j = 0; j < observations.size(); ++j) {
      LandmarkObs temp_obs = observations[j];
      LandmarkObs transformed_obs = transformObservation(current_particle,temp_obs);
      transformed_obs.id = findAssociation(transformed_obs, map_landmarks);
      Map::single_landmark_s predicted = map_landmarks.landmark_list[transformed_obs.id];

      //associations.push_back(transformed_obs.id);
      //sense_x.push_back(transformed_obs.x);
      //sense_y.push_back(transformed_obs.y);

      current_weight *= MVGaussian(predicted, std_landmark, transformed_obs);
    }

    current_particle = SetAssociations(current_particle, associations, sense_x, sense_y);
    //update weight
    current_particle.weight = current_weight;

    particles[i] = current_particle;
    weights[i] = current_weight;

  }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;

  std::discrete_distribution<> dist_weights(weights.begin(),weights.end());
  std::vector<Particle> temp_particles;

  for (unsigned i = 0; i < num_particles; ++i)
  {
    Particle sample_particle = particles[dist_weights(gen)];
    sample_particle.id = i;
    temp_particles.push_back(sample_particle);
  }

  particles.clear();
  particles = temp_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
