/*
 * smallpt.cpp
 *
 *  Created on: Mar 27, 2019
 *      Author: mauro
 */

#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
#include <iostream>
#include <array>
#include <chrono>
#include <map>
#include <algorithm>
#include "utilities.h"
#include <fstream>
#include <vector>
#include <numeric>
#include <sstream>
using namespace std;
using namespace std::chrono;

const int NUMBER_OBJ = 17;
const int dim_action_space = 24;
using Key = std::array<float, 6>;
using QValue = std::array<float, dim_action_space + 1>;
using ColorValue = std::array<float, 3>;
using StateAction = std::array<float, 7>;		// Key for map, State + Action for learning rate and counts
using StateActionCount = float;					// Count of state-action pair visits for the learning rate
float lr = 1;

struct Vec {
	float x, y, z;                  // position, also color (r,g,b)
	Vec(float x_ = 0, float y_ = 0, float z_ = 0) {
		x = x_;
		y = y_;
		z = z_;
	}
	Vec operator+(const Vec &b) const {
		return Vec(x + b.x, y + b.y, z + b.z);
	}
	Vec operator-(const Vec &b) const {
		return Vec(x - b.x, y - b.y, z - b.z);
	}
	Vec operator*(float b) const {
		return Vec(x * b, y * b, z * b);
	}
	Vec operator*(double b) const {
		return Vec(x * b, y * b, z * b);
	}
	Vec operator*(int b) const {
		return Vec(x * b, y * b, z * b);
	}

	Vec mult(const Vec &b) const {
		return Vec(x * b.x, y * b.y, z * b.z);
	}
	Vec& norm() {
		return *this = *this * (1 / sqrt(x * x + y * y + z * z));
	}
	float dot(const Vec &b) const {
		return x * b.x + y * b.y + z * b.z;
	}
	Vec operator%(Vec &b) {
		return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
	} // cross
	Vec operator%(const Vec &b) {
		return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
	} // cross
	float magnitude() {
		return sqrt(x * x + y * y + z * z);
	}
};


// LOOKFROM for the Camera
const Vec LOOKFROM = Vec(50, 40, 168);

// Action and Direction for Q-learning
using Action = int;
using Direction = Vec;

struct Ray {
	Vec o, d;
	Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};

enum Refl_t {				// material types, used in radiance()
	DIFF, SPEC, REFR
};

struct Hit_records {		// Store object element
	Refl_t refl;
	Vec c;
	Vec e;
};

struct Struct_states {				// Store info on old and new state
	std::array<float, 6> old_state;
	int old_action;
	float prob;
	int old_id;
};


class Hitable {			// a pure virtual function makes sure we always override the function hit
public:
	virtual float intersect(const Ray &r) const = 0;
	virtual bool intersect(const Vec &vec) const = 0;
	virtual Vec normal(const Ray &r, Hit_records &hit, Vec &x) const = 0;
	virtual std::array<float, 6> add_key(Vec &pos, Vec& nl) const {
		Vec x_reduced = Vec(ceil((float) pos.x / 10), ceil((float) pos.y/ 10), ceil((float) pos.z / 10));
		return { x_reduced.x, x_reduced.y, x_reduced.z, nl.x, nl.y, nl.z};
	};
	virtual std::array<float, 3> add_value_color(std::array<float, 3>& x_reduced) const = 0;
	virtual std::array<float, dim_action_space + 1> add_value() const {
		std::array<float, dim_action_space+1> q_values;
		for(int i=0; i <dim_action_space; i++){
			q_values[i]=1;
		}
		q_values[dim_action_space]=1*dim_action_space;
		return q_values;
	};

	virtual float get_fixed_coord() const = 0;
	template<typename T>
	bool isA() {
		return (dynamic_cast<T*>(this) != NULL);
	}

};

class Rectangle_xz: public Hitable {
public:
	float x1, x2, z1, z2, y;
	Vec e, c;         // emission, color
	Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
	Rectangle_xz(float x1_, float x2_, float z1_, float z2_, float y_,
			Vec e_, Vec c_, Refl_t refl_) :
			x1(x1_), x2(x2_), z1(z1_), z2(z2_), y(y_), e(e_), c(c_), refl(refl_) {
	}

	float intersect(const Ray &r) const { // returns distance, 0 if no hit
		float t = (y - r.o.y) / r.d.y;		// ray.y = t* dir.y
		const float& x = r.o.x + r.d.x * t;
		const float& z = r.o.z + r.d.z * t;
		if (x < x1 || x > x2 || z < z1 || z > z2 || t < 0) {
			t = 0;
			return 0;
		} else {
			return t;
		}
	}

	bool intersect(const Vec &v) const {		 // check if point is on sphere
		return v.x > x1 && v.x < x2 && v.z > z1 && v.z < z2 && (v.y > (y - 1)) && (v.y < y + 1) ?	true : false;
	}

	Vec normal(const Ray &r, Hit_records &hit, Vec &x) const {
		Vec n = Vec(0, 1, 0);
		hit.refl = refl;
		hit.c = c;
		hit.e = e;
		return n.dot(r.d) < 0 ? n : n * -1;
	}

//	std::array<float, 6> add_key(Vec& pos, Vec& nl) const {
//		Vec x_reduced = Vec(ceil((float) pos.x / 10), ceil((float) pos.y/ 10), ceil((float) pos.z / 10));
//		return { x_reduced.x, x_reduced.y, x_reduced.z, nl.x, nl.y, nl.z};
//	}

	std::array<float, 3> add_value_color(std::array<float, 3>& x_reduced) const {
		return { x_reduced[0] / 10 * (rand() / float(RAND_MAX)), x_reduced[1] * (rand() / float(RAND_MAX)),
				x_reduced[2]  / 10 * (rand() / float(RAND_MAX)) };
	}

	 float get_fixed_coord() const {
		 return y;
	 }
};

class Rectangle_xy: public Hitable {
public:
	float x1, x2, y1, y2, z;
	Vec e, c;         // emission, color
	Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
	Rectangle_xy(float x1_, float x2_, float y1_, float y2_, float z_,	Vec e_, Vec c_, Refl_t refl_) :
			x1(x1_), x2(x2_), y1(y1_), y2(y2_), z(z_), e(e_), c(c_), refl(refl_) {}

	float intersect(const Ray &r) const { // returns distance, 0 if no hit
		float t = (z - r.o.z) / r.d.z;
		const float& x = r.o.x + r.d.x * t;
		const float& y = r.o.y + r.d.y * t;
		if (x < x1 || x > x2 || y < y1 || y > y2 || t < 0) {
			t = 0;
			return 0;
		} else {
			return t;
		}
	}

	bool intersect(const Vec &v) const {		 // check if point is on sphere
		return v.x > x1 && v.x < x2 && v.y > y1 && v.y < y2 && (v.z > (z - 1)) && (v.z < (z + 1))  ? true : false;
	}

	Vec normal(const Ray &r, Hit_records &hit, Vec &x) const {
		Vec n = Vec(0, 0, 1);
		hit.refl = refl;
		hit.c = c;
		hit.e = e;
		return n.dot(r.d) < 0 ? n : n * -1;
	}

//	std::array<float, 5> add_key(Vec& pos, Vec& nl) const {
//		Vec x_reduced = Vec(ceil((float) pos.x / 10), ceil((float) pos.y / 10),	pos.z / 10);
//		return { x_reduced.x, x_reduced.y, nl.x, nl.y, nl.z};
//	}

	std::array<float, 3> add_value_color(std::array<float, 3>& x_reduced) const {
		return { x_reduced[0] / 10 * (rand() / float(RAND_MAX)), x_reduced[1] 	/ 10 * (rand() / float(RAND_MAX)),
				x_reduced[2]  * (rand() / float(RAND_MAX)) };
	}

	 float get_fixed_coord() const {
		 return z;
	 }
};

class Rectangle_yz: public Hitable {
public:
	float y1, y2, z1, z2, x;
	Vec e, c;         // emission, color
	Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
	Rectangle_yz(float y1_, float y2_, float z1_, float z2_, float x_,	Vec e_, Vec c_, Refl_t refl_) :
			y1(y1_), y2(y2_), z1(z1_), z2(z2_), x(x_), e(e_), c(c_), refl(refl_) {}

	float intersect(const Ray &r) const { // returns distance, 0 if no hit
		float t = (x - r.o.x) / r.d.x;
		const float& y = r.o.y + r.d.y * t;
		const float& z = r.o.z + r.d.z * t;
		if (y < y1 || y > y2 || z < z1 || z > z2 || t < 0) {
			t = 0;
			return 0;
		} else {
			return t;
		}
	}

	bool intersect(const Vec &v) const {		 // check if point is on sphere
		return v.y > y1 && v.y < y2 && v.z > z1 && v.z < z2 && (v.x > (x - 1)) && (v.x < (x + 1)) ? true : false;
	}

	Vec normal(const Ray &r, Hit_records &hit, Vec &x) const {
		Vec n = Vec(1, 0, 0);
		hit.refl = refl;
		hit.c = c;
		hit.e = e;
		return n.dot(r.d) < 0 ? n : n * -1;
	}

//	std::array<float, 5> add_key(Vec& pos, Vec& nl) const {
//		Vec x_reduced = Vec( pos.x / 10, ceil((float) pos.y / 10),	ceil((float) pos.z / 10));
//		return { x_reduced.y, x_reduced.z, nl.x, nl.y, nl.z};
//	}

	std::array<float, 3> add_value_color(std::array<float, 3>& x_reduced) const {
		return { x_reduced[0] * (rand() / float(RAND_MAX)), x_reduced[1] 	/ 10 * (rand() / float(RAND_MAX)),
				x_reduced[2]  / 10 * (rand() / float(RAND_MAX)) };
	}

	 float get_fixed_coord() const {
		 return x;
	 }
};

class Sphere: public Hitable {
public:
	float rad;       // radius
	Vec p, e, c;      // position, emission, color
	Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
	Sphere(float rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) :	rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
	float intersect(const Ray &r) const { // returns distance, 0 if no hit
		Vec op = p - r.o; 	// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		float t, eps = 1e-4;
		float b = op.dot(r.d);
		float det = b * b - op.dot(op) + rad * rad;
		if (det < 0)
			return 0;
		else
			det = sqrt(det);
		return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
	}

	bool intersect(const Vec &v) const { // check if point is on sphere
		return ((v.x - p.x) * (v.x - p.x) + (v.y - p.y) * (v.y - p.y) + (v.z - p.z) * (v.z - p.z) > ((rad * rad) - 5)
				&& (v.x - p.x) * (v.x - p.x) + (v.y - p.y) * (v.y - p.y) + (v.z - p.z) * (v.z - p.z) < ((rad * rad) + 5)) ? true : false;
	}

	Vec normal(const Ray &r, Hit_records &hit, Vec &x) const {
		Vec n = (x - p).norm();						// sphere normal
		hit.refl = refl;
		hit.c = c;
		hit.e = e;
		return n.dot(r.d) < 0 ? n : n * -1;	// properly orient the normal. If I am inside the sphere, the normal needs to point towards the inside
											// indeed, the angle would be < 90, so dot() < 0. Also, if in a glass it enters or exits
	}
};

class Camera {
public:
	// lookfrom is the origin
	// lookat is the point to look at
	// vup, the view up vector to project on the new plane when we incline it. We can also tilt
	// the plane
	Camera(Vec lookfrom, Vec lookat, Vec vup, float vfov, float aspect) {// vfov is top to bottom in degrees, field of view on the vertical axis
		Vec w, u, v;
		float theta = vfov * M_PI / 180;	// convert to radiants
		float half_height = tan(theta / 2);
		float half_width = aspect * half_height;
		origin = lookfrom;
		w = (lookat - lookfrom).norm();
		u = (w % vup).norm();
		v = (u % w);

		lower_left_corner = origin - u * half_width - v * half_height + w;
		horizontal = u * (half_width * 2);
		vertical = v * (half_height * 2);
	}
	Ray get_ray(const float& s, const float& t) {
		return Ray(origin,
				lower_left_corner + horizontal * s + vertical * t - origin);
	}

	Vec origin;
	Vec lower_left_corner;
	Vec horizontal;
	Vec vertical;
};

Hitable *rect[NUMBER_OBJ] = {
	new Rectangle_xy(1, 99, 0, 81.6, 0, Vec(),Vec(.75, .75, .75), DIFF), 		// Front
	new Rectangle_xy(1, 99, 0, 81.6, 170, Vec(), Vec(.75, .75, .75), DIFF),		// Back
	new Rectangle_yz(0, 81.6, 0, 170, 1, Vec(), Vec(.25, .75, .25), DIFF),		// Left, green
	new Rectangle_yz(0, 81.6, 0, 170, 99, Vec(), Vec(.75, .25, .25), DIFF),		// Right, red
	new Rectangle_xz(1, 99, 0, 170, 0, Vec(), Vec(.75, .75, .75), DIFF),		// Bottom
	new Rectangle_xz(1, 99, 0, 170, 81.6, Vec(), Vec(.75, .75, .75), DIFF),		// Top
	new Rectangle_xz(32, 68, 63, 96, 81.595, Vec(12, 12, 12), Vec(), DIFF),		// Light


	/*new Sphere(16.5,Vec(27,16.5,47), Vec(),Vec(1,1,1)*.999, DIFF),			//Mirr
	new Sphere(16.5,Vec(73,16.5,78), Vec(),Vec(.75,.75,.75), DIFF) */  			//Glas

	new Rectangle_xy(12, 42, 0, 50, 32, Vec(), Vec(1,1,1), DIFF),				// Tall box
	new Rectangle_xy(12, 42, 0, 50, 62, Vec(), Vec(1,1,1), DIFF),
	new Rectangle_yz(0, 50, 32, 62, 12, Vec(), Vec(1,1,1), DIFF),
	new Rectangle_yz(0, 50, 32, 62, 42 , Vec(), Vec(1,1,1), DIFF),
	new Rectangle_xz(12, 42, 32, 62, 50, Vec(), Vec(1,1,1), DIFF),
																				// Short box
	new Rectangle_xy(63, 88, 0, 25, 63, Vec(), Vec(1,1,1), DIFF),
	new Rectangle_xy(63, 88, 0, 25, 88, Vec(), Vec(1,1,1), DIFF),
	new Rectangle_yz(0, 25, 63, 88, 63, Vec(), Vec(1,1,1), DIFF),
	new Rectangle_yz(0, 25, 63, 88, 88, Vec(), Vec(1,1,1), DIFF),
	new Rectangle_xz(63, 88, 63, 88, 25, Vec(), Vec(1,1,1), DIFF)
};


// clamp makes sure that the set is bounded (used for radiance() )
inline float clamp(float x) {
	return x < 0 ? 0 : x > 1 ? 1 : x;
}

// toInt() applies a gamma correction of 2.2, because our screen doesn't show colors linearly
inline int toInt(float x) {
	return int(pow(clamp(x), 1 / 2.2) * 255 + .5);
}

// Convert spherical coordinates into cartesian
inline Vec spherToCart(Vec& spher){
	return Vec(sin(spher.y)*sin(spher.z), sin(spher.y)*cos(spher.z), cos(spher.y));
}

// convert Cartesian coordinates into spherical
inline Vec cartToSpher(Vec& cart){
	return Vec(1, atan((sqrt(cart.x*cart.x + cart.y*cart.y))/cart.z), atan2(cart.x, cart.y));
}

inline bool intersect(const Ray &r, float &t, int &id, int& old_id) {
	const float& n = NUMBER_OBJ; //Divide allocation of byte of the whole scene, by allocation in byte of one single element
	float d;
	float inf = t = 1e20;
	for (int i = 0; i < n; i++) {
		if ((d = rect[i]->intersect(r)) && d < t && i != old_id) {	// Distance of hit point
			t = d;
			id = i;
		}
	}

	// Return the closest intersection, as a bool
	return t < inf;
}

inline Vec importanceSampling_scattering(const Vec& nl, unsigned short *Xi) {
	/*
	// COSINE-WEIGHTED SAMPLING
	float r1 = 2 * M_PI * erand48(Xi);		// get random angle
	float r2 = erand48(Xi);			// get random distance from center
	float r2s = sqrt(r2);
	// Create orthonormal coordinate frame for scattered ray
	Vec w = nl;			// w = normal
	Vec u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm();
	Vec v = w % u;
	return (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();

	// reflection ray with cosine sampling (check calculus)
	 */

	 // UNIFORM SAMPLING
	 float r1 = 2*M_PI*erand48(Xi);		// get random angle Gamma
	 float r2 = erand48(Xi);			// get random distance from center
	 // Create orthonormal coordinate frame for scattered ray
	 Vec w = nl;			// w = normal
	 Vec u = ((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm();
	 Vec v = w%u;
	 return (u*cos(r1)*sqrt(r2*(2-r2)) + v*sin(r1)*sqrt(r2*(2-r2)) + w*(1-r2)).norm();   // random reflection ray

}

inline Vec light_sampling(const Vec& nl, const Vec& hit, unsigned short *Xi) {
	// SAMPLING LIGHT (HARD-CODED)
	float x_light = 32 + rand() * 36 / float(RAND_MAX);
	float z_light = 63 + rand() * 36 / float(RAND_MAX);
	Vec light_vec = Vec(x_light, 81.6, z_light) - hit;
	return light_vec;
}

inline Vec hittingPoint(const Ray &r, int &id, int& old_id) {
	float t;                             // distance to intersection
	if (!intersect(r, t, id, old_id))
		return Vec();

	Vec x = (r.o + r.d * (t - 0.01));// ray intersection point (t calculated in intersect())

	/*if( rect[id]->isA<Rectangle_xz>() ){
		x.y = rect[id]->get_fixed_coord();
	 }else if(rect[id]->isA<Rectangle_xy>()){
		x.z = rect[id]->get_fixed_coord();
	 }else if(rect[id]->isA<Rectangle_yz>()){
		x.x = rect[id]->get_fixed_coord();
	 }*/
	return x ;
}

inline Vec normal(Vec &point, Vec &center) {
	return (point - center).norm();
}

inline Vec getTangent(Vec& normal) {
	Vec new_vector = Vec(normal.x+1, normal.y+1, normal.z+1);		// This cannot be zero, or parallel to the normal.
															// Usually I need to check, but since all my tangents are (0,0,1),
															// (0,1,0) and (1,0,0), I don't need to check in this SPECIFIC CASE.
	return (normal % (new_vector.norm()));
}

inline int create_state_space(std::map<Key, QValue> *dict, Vec& nl) {
	std::map<Key, QValue> &addrDict = *dict;
	int count = 0;
	for (int x = 0; x < 100; x++) {
		for (int y = -1; y < 85; y++) {
			for (int z = -1; z < 171; z++) {
				Vec vec = Vec(x, y, z);
				Ray r = Ray(LOOKFROM, (vec - LOOKFROM).norm());
				int id = 0;
				int old_id = 0;				// NOT TRUE, NEED TO PASS STATES_REC
				Vec pos = hittingPoint(r, id, old_id);
				Key key = rect[id]->add_key(pos, nl);
				//QValue value = rect[id]->add_value_color(key);				//To return different colors for each state. If you don't want colors,
																			//comment this
				if (addrDict.count(key) < 1) {
					QValue value = rect[id]->add_value();			// To initialize Q-values. To return colors, comment this line.
					addrDict[key] = value;

					/*if(key[0]==0 || key[1] == 0 || key[2] == 0){
						std::cout << key[0] << " " << key[1] << " " << key[2] << std::endl;}*/
					count += 1;
				}
			}
		}
	}
	return count;
}

inline void initialize_dictAction(std::map<Action, Direction> *dictAction){
	std::map<Action, Direction> &addrDict = *dictAction;
	std::ifstream myFile;
	myFile.open("sphere_point.csv");
	std::string x, y, z;
	int count = 0;
	while(myFile.good()){
		if( getline(myFile, x, ',') && getline(myFile, y, ',') &&	getline(myFile, z, '\n')) {
		addrDict[count] = Vec(std::stod(x),std::stod(y),std::stod(z));
		count += 1;
		}
	}
	myFile.close();
}


// DRAW SCATTER RAYS IN PLOTLY
std::array<int,6> target_state = {8,2,9,0,0,1};
inline void plotly_scatter_rays(Struct_states &states_rec, Vec& x){
	if(states_rec.old_state[0]==target_state[0] && states_rec.old_state[1]==target_state[1] && states_rec.old_state[2]==target_state[2]
		&& states_rec.old_state[3]==target_state[3] && states_rec.old_state[4]==target_state[4] && states_rec.old_state[5]==target_state[5]){
		std::ofstream outfile;
		outfile.open("scattered_rays.txt", std::ios_base::app);
		outfile << x.x << "," << x.y << "," << x.z << std::endl;
	}
};

// DRAW SCATTERED STATE IN PLOTLY
inline void plotly_scatter_state(Key& key , Vec& x){
	if(key[0]==target_state[0] && key[1]==target_state[1] && key[2]==target_state[2]
		&& key[3]==target_state[3] && key[4]==target_state[4] && key[5]==target_state[5]){
		std::ofstream outfile;
		outfile.open("scattered_state.txt", std::ios_base::app);
		outfile << x.x << "," << x.y << "," << x.z << std::endl;
	}
};

// DRAW SCATTERED SPHERE IN PLOTLY
inline void plotly_scatter_sphere(Key& key, Vec& d){
	if(key[0]==target_state[0] && key[1]==target_state[1] && key[2]==target_state[2]
		&& key[3]==target_state[3] && key[4]==target_state[4] && key[5]==target_state[5]){
		Vec sphere = Vec(key[0]*10 - 5 + 5*d.x, key[1]*10 - 5 + 5*d.y,key[2]*10 - 5 + 5*d.z);
		std::ofstream outfile;
		outfile.open("scattered_sphere.txt", std::ios_base::app);
		outfile << sphere.x << "," << sphere.y << "," << sphere.z << std::endl;
	}
};

inline void plot_Q(Key& state,std::map<Key, QValue> *dict){
	if(state[0]==target_state[0] && state[1]==target_state[1] && state[2]==target_state[2]
			&& state[3]==target_state[3] && state[4]==target_state[4] && state[5]==target_state[5]){
		std::map<Key, QValue> &addrDict = *dict;
		std::ofstream outfile;
		outfile.open("q-plot.txt", std::ios_base::app);
		for(int i=0; i < dim_action_space; i++){
			outfile << addrDict[state][i] << ",";
		}
		outfile << std::endl;
	}
}

inline void updateQtable(Key& state, Key& next_state, Hit_records& hit,std::map<Key, QValue> *dict, std::map<Action, Direction> *dictAction, int &old_action, float& BRDF, Vec& nl,
		Vec& x, float prob, std::map<StateAction, StateActionCount> *dictStateActionCount){

	std::map<Key, QValue> &addrDict = *dict;
	std::map<Action, Direction> &addrDictAction = *dictAction;
	std::map<StateAction, StateActionCount> &addrDictStateActionCount = *dictStateActionCount;

	float update = 0;
	float& dict_state = addrDict[state][old_action];

	std::array<float, dim_action_space + 1>& dict_next_state = addrDict[next_state];
	lr = 100 / (100 + addrDictStateActionCount[{state[0], state[1],state[2], state[3],state[4], state[5], (float) old_action}]);

	/*if(lr>0.1){
		lr = 1 / (1 + addrDictStateActionCount[{state[0], state[1],state[2], state[3],state[4], state[5], (float) old_action}]);
	}else{
		lr = 0.1;
	}*/
	if (hit.e.x > 5){		// if light
		update = dict_state * (1 - lr) + lr* std::max({hit.e.x, hit.e.y, hit.e.z});
	}else{
		float cumulative_q = 0;
		for(int i=0; i< dim_action_space; i++){
			// calculate cos_theta_i
			cumulative_q += dict_next_state[i] * addrDictAction[i].y;
			//std::cout << cumulative_q << std::endl;
		}
		update = dict_state * (1 - lr) + lr * (1/dim_action_space) * cumulative_q *  BRDF;		//maybe BRDF divided by pi
		//std::cout << "------------------"<< std::endl;
		//std::cout << "lr: " << lr << std::endl;
		//std::cout << "update: " << update << std::endl;

	}
	addrDict[state][old_action] = update;

	//update total
	float total = 0;
	for(int s=0; s<dim_action_space; s++){
		total += addrDict[state][s];
	}
	addrDict[state][dim_action_space] = total;
	plot_Q(state,dict);
}

inline Vec sampleScatteringMaxQ(std::map<Key, QValue> *dict, std::map<Action, Direction> *dictAction, int &id, Vec &x, Vec &nl, const Ray &r, Struct_states& states_rec,
		bool learning_phase, std::map<StateAction, StateActionCount> *dictStateActionCount, float& epsilon) {

	std::map<Key, QValue> &addrDict = *dict;
	std::map<Action, Direction> &addrDictAction = *dictAction;
	std::map<StateAction, StateActionCount> &addrDictStateActionCount = *dictStateActionCount;

	const Key& state = rect[id]->add_key(x, nl.norm());		// coordinates

	if (addrDict.count(state) < 1) {
		const QValue& value = rect[id]->add_value();			// To initialize Q-values. To return colors, comment this line.
		addrDict[state] = value;
	}

	// Create temporary coordinates system
	Vec& w = nl.norm();
	const Vec& u = getTangent(w).norm();
	const Vec& v = (w % u).norm();

	std::array<float, dim_action_space + 1>& qvalue = addrDict[state];

	Vec point_old_coord;
	int action;
	const float& random = rand() / float(RAND_MAX);
	// Choose action based on probability of actions, during TRAINING PHASE
	if(learning_phase){
		const float& total = qvalue[dim_action_space];
		float maximum = 0;
		for(int s=0; s<dim_action_space; s++){
			if(qvalue[s] > maximum){
				maximum = qvalue[s];
			}
		}
		//float p = *std::max_element(std::begin(qvalue), std::end(qvalue)); // max Q_value for Russian Roulette
		if( random > epsilon && total!=0){			// When Q_value is low, choose random. Else, choose max action
			const float& prob = random * 0.9999;
			float cumulativeProbability = 0.0;

			for (int i=0; i < dim_action_space; i++) {
				cumulativeProbability += qvalue[i]/total;
				if (prob <= cumulativeProbability) {
					action = i;
					break;
				}
			}
			states_rec.prob = total/(qvalue[action]);

			//action = std::distance(qvalue.begin(), std::max_element(qvalue.begin(), qvalue.end()));		// get position max action
			point_old_coord= addrDictAction[action];
		}else{ 											// Choose random action
			action = (int) (random *23);
			point_old_coord= addrDictAction[action];
			states_rec.prob = 24;
		}
	}else{	// Choose action base don maximum Q value, during ACTIVE PHASE
		// CHOOSING Q-PROPORTIONAL SCATTERING
		const float& total = qvalue[dim_action_space];
//		for(int s=0; s<dim_action_space; s++){
//			total += qvalue[s];
//		}
		const float& prob = random*0.99;
		float cumulativeProbability = 0.0;
		action = 0;
		for (int i=0; i < dim_action_space; i++) {
			cumulativeProbability += qvalue[i]/total;
			if (prob <= cumulativeProbability) {
				action = i;
				break;
			}
		}
		states_rec.prob = (total *  M_PI)/(qvalue[action]*12);	// 1/ ( (q/tot) * (1/ (2 * pi) / 24))
		point_old_coord= addrDictAction[action];

		// CHOOSING MAX Q
		/*action = std::distance(qvalue.begin(), std::max_element(qvalue.begin(), qvalue.end()));		// get position max action
		point_old_coord= addrDictAction[action];
		states_rec.prob = 0.2618; 	*/	// 1 / (1 / (2 *pi) / n));
	}

	states_rec.old_state = state;
	states_rec.old_action = action;


	// Add StateAction count
	const std::array<float, 7>& stateactionpair = {state[0],state[1],state[2],state[3],state[4],state[5], (float) action};
	if(addrDictStateActionCount.count(stateactionpair) < 1){
		addrDictStateActionCount[stateactionpair] = 0;
	}
	addrDictStateActionCount[stateactionpair] = addrDictStateActionCount[stateactionpair] + 1;

	// Scatter random inside the selected patch, convert to spherical coordinates for semplicity and then back to cartesian

	Vec spher_coord = cartToSpher(point_old_coord);

	// RESULTS ARE NOT THE EXACT ONES. tHIS ALLOWS TO HAVE A SMALL FRAME, OTHERWISE THE LACK OF ACCURACY SENDS RAY IN THE WRONG DIRECTON
	spher_coord.z = (0.78539*(rand() / float(RAND_MAX)) - 0.39269) + spher_coord.z;		// add or subtract randomly range {-22.5, 22.5} degrees to phi, in radian
	if(point_old_coord.z < 0.33){
		spher_coord.y = 0.33*(rand() / float(RAND_MAX)) + 1.23;		// math done on the notes: theta - 0.168 < theta < theta - 0.168
	}
	else if(point_old_coord.z >= 0.33 && point_old_coord.z < 0.66){
		spher_coord.y = 0.389*(rand() / float(RAND_MAX)) + 0.841;		// theta - 0.192 < theta < theta - 0.192
	}
	else{
		spher_coord.y = 0.841*(rand() / float(RAND_MAX));			//theta - 0.42 < theta < theta - 0.42
	}

	point_old_coord = spherToCart(spher_coord);
	return (u*point_old_coord.x  + v*point_old_coord.y  + w*point_old_coord.z); // new_point.x * u + new_point.y * v + new_point.z * w + hitting_point
}

// Create the centers in the state patches to use for the learning phase
inline bool create_center_states(std::map<Key, QValue> *dict, int id, Vec& x, int& counter_red, Vec& nl){
	Key key = rect[id]->add_key(x, nl);
	std::map<Key, QValue> &addrDict = *dict;

	// CENTRE OF STATES
	if(((x.x > (key[0]*10- 5.2)) && (x.x < (key[0]*10- 4.8)) && (x.y > (key[1]*10- 5.2)) && (x.y < (key[1]*10- 4.8))) ||
			((x.x > (key[0]*10- 5.2)) && (x.x < (key[0]*10- 4.8)) && (x.z > (key[2]*10- 5.2)) && (x.z < (key[2]*10- 4.8)))||
			((x.y > (key[1]*10- 5.2)) && (x.y < (key[1]*10- 4.8)) && (x.z > (key[2]*10- 5.2)) && (x.z < (key[2]*10- 4.8)))){
				counter_red = counter_red + 1;
				return true;
		}
	return false;
}

inline Vec visualize_states(std::map<Key, QValue> *dict, int id, Vec& x, int& counter_red, Vec& nl){
	// COLOR STATES
	std::map<Key, QValue> &addrDict = *dict;
	Key key = rect[id]->add_key(x, nl.norm());

	// COLOR CENTRE OF STATES
	if( ((x.x > (key[0]*10- 6)) && (x.x < (key[0]*10- 4)) && (x.y > (key[1]*10- 6)) && (x.y < (key[1]*10- 4))) ||
			((x.x > (key[0]*10- 6)) && (x.x < (key[0]*10- 4)) && (x.z > (key[2]*10- 6)) && (x.z < (key[2]*10- 4)))||
			((x.y > (key[1]*10- 6)) && (x.y < (key[1]*10- 4)) && (x.z > (key[2]*10- 6)) && (x.z < (key[2]*10- 4)))){
			counter_red = counter_red + 1;
			return Vec(1,0,0);
	}
	return Vec(addrDict[key][0], addrDict[key][1], addrDict[key][2]);
};



inline Vec radiance(const Ray &r, int depth, unsigned short *Xi, float *path_length, std::map<Key, QValue> *dict, int& counter_red, std::map<Action, Direction> *dictAction,
		Struct_states &states_rec, bool& learning_phase, bool q_learning_mode, float& reward, std::map<StateAction, StateActionCount> *dictStateAction, int& counter, float& epsilon) {
	Hit_records hit;
	int id = 0;                           // initialize id of intersected object
	int old_id = states_rec.old_id;
	Vec x = hittingPoint(r, id, old_id);            // id calculated inside the function

	// DRAW SCATTER RAYS IN PLOTLY
	//plotly_scatter_rays(states_rec, x);

	if(x.x==0){
		counter +=1;
	}
	// To visualize states
	//return visualize_states(dict, id, x, counter_red);

	Hitable* &obj = rect[id];				// the hit object
	Vec nl = obj->normal(r, hit, x);

	if(x.x >= 32  && x.x <= 68 && x.y >= 81 && x.z >= 63 && x.z <= 96){
		hit.c = Vec(0,0,0);
		hit.e = Vec(12,12,12);
	}
	Key key;
	if(q_learning_mode){
		states_rec.old_id = id;
		Key key = rect[id]->add_key(x, nl.norm());
		std::map<Key, QValue> &addrDict = *dict;

		if (addrDict.count(key) < 1) {
			QValue value = rect[id]->add_value();			// To initialize Q-values. To return colors, comment this line.
			addrDict[key] = value;
		}
	}
	// DRAW LINES IN PLOTLY
//	std::ofstream outfile;
//	outfile.open("lines.txt", std::ios_base::app);
//	outfile << key[0] << ","<< key[1] << "," << key[2] << "," << key[3] << "," << key[4] << "," << key[5] << "," << x.x << "," << x.y << "," << x.z << std::endl;

	// DRAW SCATTERED STATE IN PLOTLY
	//plotly_scatter_state(key, x);

	Vec f = hit.c;							// object color
	float p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max reflectivity (maximum component of r,g,b)
	const float& q = rand() / float(RAND_MAX);

	if (++depth > 5 || !p)  {// Russian Roulette. After 5 bounces, it determines if the ray continues or stops.
		if (q < p) {
			f = f * (1 / p);
		} else {
			if(depth>1 && learning_phase && q_learning_mode){		// If gets light or nothing, update Q-value
				float BRDF = std::max({hit.c.x, hit.c.y, hit.c.z})/M_PI;
				updateQtable(states_rec.old_state, key, hit, dict, dictAction, states_rec.old_action, BRDF , nl, x, states_rec.prob,dictStateAction);
			}

			if(hit.e.x > 5){
				reward += hit.e.x;
			}
			return hit.e;
		}
	}

	Vec d;
	Vec d1;
	Vec d2;
	float PDF_inverse = 1;
	float BRDF = 1;
	float PDF_inverse2 = 1;
	float BRDF2 = 1;
	float t = 0; 	// distance to intersection
	bool explicit_light = true;
	// This is based on the reflectivity, and the BRDF scaled to compensate for it.
	if (hit.refl == DIFF && !q_learning_mode) {
		// ------------- EXPLICIT LIGHT SAMPLING -------------------------------------
		// Samples need to be doubled.
		if(explicit_light){
			d1 = importanceSampling_scattering(nl, Xi);
			intersect(Ray(x, d1.norm()), t, id, old_id);
			if(t < 1e20){
				*path_length = *path_length + t;
			}
		//		if(depth==1 && id==6){
		//			return Vec(0,0,0);
		//		}
		//		return hit.e + f.mult(radiance(Ray(x, d1.norm()), depth, Xi, path_length, dict, counter_red, dictAction, states_rec, learning_phase, q_learning_mode, reward, dictStateAction, counter, epsilon)) * PDF_inverse * BRDF;// get color in recursive functi
			if(id != 6){
				d2 = light_sampling(nl, x, Xi);
				intersect(Ray(x, d2.norm()), t, id, old_id);
				if(id==6){
					if(t < 1e20){
						*path_length = *path_length + t;
					}
					PDF_inverse2 = fabs((1296 * d2.norm().dot(Vec(0, 1, 0))) / (t * t));	//PDF = r^2 / (A * cos(theta_light))
					BRDF2 = fabs(d2.norm().dot(nl) / M_PI);
		//				return hit.e + f.mult(radiance(Ray(x, d2.norm()), depth, Xi, path_length, dict, counter_red, dictAction, states_rec, learning_phase, q_learning_mode, reward, dictStateAction, counter, epsilon)) * PDF_inverse2 * BRDF2 ;  // get color in recursi;// get color in recursive function
					return hit.e + f.mult(radiance(Ray(x, d1.norm()), depth, Xi, path_length, dict, counter_red, dictAction, states_rec, learning_phase, q_learning_mode, reward, dictStateAction, counter, epsilon)) * PDF_inverse * BRDF +
							 f.mult(radiance(Ray(x, d2.norm()), depth, Xi, path_length, dict, counter_red, dictAction, states_rec, learning_phase, q_learning_mode, reward, dictStateAction, counter, epsilon)) * PDF_inverse2 * BRDF2 ;  // get color in recursi;// get color in recursive function
					}
			}
			return hit.e + f.mult(radiance(Ray(x, d1.norm()), depth, Xi, path_length, dict, counter_red, dictAction, states_rec, learning_phase, q_learning_mode, reward, dictStateAction, counter, epsilon)) * PDF_inverse * BRDF;// get color in recursive function
		}
		else{
			// ------------- RANDOM SCATTERING ------------------------------------
			d = importanceSampling_scattering(nl, Xi);
			intersect(Ray(x, d.norm()), t, id, old_id);
			if(t < 1e20){
				*path_length = *path_length + t;
			}
		}
		//plotly_scatter_sphere(key, d);
		return hit.e + f.mult(radiance(Ray(x, d.norm()), depth, Xi, path_length, dict, counter_red, dictAction, states_rec, learning_phase, q_learning_mode, reward, dictStateAction, counter, epsilon)) * PDF_inverse * BRDF;// get color in recursive function
	}
	else if (hit.refl == DIFF && depth == 1 && learning_phase ) {
	// ------------ Q-LEARNING, TRAINING PHASE, FIRST BOUNCE --------------------------------------------
	d = sampleScatteringMaxQ(dict, dictAction, id, x, nl, r, states_rec, learning_phase, dictStateAction, epsilon);
	PDF_inverse = states_rec.prob;		// PDF = 1/24 since the ray can be scattered in one of the 24 areas
	//intersect(Ray(x, d.norm()), t, id, old_id);
	//*path_length = *path_length + t;
	return radiance(Ray(x, d.norm()), depth, Xi, path_length, dict, counter_red, dictAction, states_rec, learning_phase, q_learning_mode, reward, dictStateAction, counter, epsilon);// get color in recursive function
	}
	else if(hit.refl == DIFF && depth > 1 && learning_phase){
		BRDF =  std::max({hit.c.x, hit.c.y, hit.c.z})/M_PI;
		updateQtable(states_rec.old_state, key, hit, dict, dictAction, states_rec.old_action, BRDF , nl, x, states_rec.prob,dictStateAction);

		d = sampleScatteringMaxQ(dict, dictAction, id, x, nl, r, states_rec, learning_phase, dictStateAction, epsilon);
		PDF_inverse = states_rec.prob;		// PDF = 1/24 since the ray can be scattered in one of the 24 areas
		//intersect(Ray(x, d.norm()), t, id, old_id);
		//*path_length = *path_length + t;
		return radiance(Ray(x, d.norm()), depth, Xi, path_length, dict, counter_red, dictAction, states_rec, learning_phase, q_learning_mode, reward, dictStateAction, counter, epsilon);// get color in recursive function
	}
	else if (hit.refl == DIFF && !learning_phase) {
		// --------------Q-LEARNING, ACTIVE PHASE --------------------------------------------
		d = sampleScatteringMaxQ(dict, dictAction, id, x, nl, r, states_rec, learning_phase, dictStateAction, epsilon);
		const float& cos_theta = nl.dot(d.norm());

//		plotly_scatter_sphere(key, d);
		// Calculate PDF as probability to scatter in that direction
		PDF_inverse = states_rec.prob;
		BRDF = 1/M_PI;
		intersect(Ray(x, d.norm()), t, id, old_id);
		return hit.e + f.mult(radiance(Ray(x, d.norm()), depth, Xi, path_length, dict, counter_red, dictAction, states_rec, learning_phase, q_learning_mode, reward, dictStateAction, counter, epsilon)) * PDF_inverse * BRDF * cos_theta;// get color in recursive function
	}


	/*
	 else if (obj.refl == SPEC)            // Ideal SPECULAR reflection
	 return obj.e + f.mult(radiance(Ray(x,r.d-n*2*n.dot(r.d)),depth,Xi));

	 Ray reflRay(x, r.d-n*2*n.dot(r.d));     // Ideal dielectric REFRACTION
	 bool into = n.dot(nl)>0;                // Ray from outside going in?
	 float nc=1, nt=1.5, nnt=into?nc/nt:nt/nc, ddn=r.d.dot(nl), cos2t;
	 if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0)    // Total internal reflection
	 return obj.e + f.mult(radiance(reflRay,depth,Xi));
	 Vec tdir = (r.d*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm();
	 float a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:tdir.dot(n));
	 float Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP=Tr/(1-P);
	 return obj.e + f.mult(depth>2 ? (erand48(Xi)<P ?   // Russian roulette
	 radiance(reflRay,depth,Xi)*RP:radiance(Ray(x,tdir),depth,Xi)*TP) :
	 radiance(reflRay,depth,Xi)*Re+radiance(Ray(x,tdir),depth,Xi)*Tr);	*/
};

// LOOPS OVER IMAGE PIXELS, SAVES TO PPM FILE
int main(int argc, char *argv[]) {
	srand(time(NULL));
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// Set up image
	int w = 512, h = 512; 		// Resolution
	int samps = 16; 			// Number samples
	Vec r;					// Used for colors of samples
	Vec *c = new Vec[w * h]; 	// The image
	int num_training_samples = 100; 			// Samples for Q learning

	std::map<Key, QValue>* dict = new std::map<Key, QValue>;    			// dict position-Qvalue

	// TEMPORARY COUNTER
	int counter_red = 0;

	// Create states
	//int number_states = create_state_space(dict);
	//std::cout << "NUMBER STATES: " << number_states << std::endl;

	// Dictionary for actions
	std::map<Action, Direction>* dictAction = new std::map<Action, Direction>; 	// dict action-direction
	initialize_dictAction(dictAction);
    /*for ( const auto &myPair : *dictAction ) {
        std::cout << myPair.first << std::endl;
        std::cout << myPair.second.x << std::endl;
    }*/

	// Count visit in each state-action pait to adjust the learning rate
	std::map<StateAction, StateActionCount>* dictStateActionCount = new std::map<StateAction, StateActionCount>; 	// dict action-direction


	// Set up camera
	Camera cam(LOOKFROM, Vec(50, 40, 5), Vec(0, 1, 0), 65, float(w) / float(h));

	// Average path length
	float path_length = 0;
	float* ptrPath_length = &path_length;
	float reward = 0;
	float epsilon = 0;
	//float epsilon_decay = 0.9965;
	float epsilon_decay = 0;
	// DEBUG
	int counter = 0;

	// Settings
	bool q_learning_mode = true;		// Q-Learning VS. Explicit light sampling
	bool learning_phase = true;			// Scatter proportional to Q VS. Greedy Q
	bool q_table_available = false ;		// create Q-Table or retrieve a pregenerated one

	if(!q_table_available && q_learning_mode){
		float avg_reward;

		// TRAINING PHASE, NO Q-Table available--------------------------------------------------------------
		for (int s = 0; s < 5; s++) {		//	scatter rays 100 times more than the size of the screen
			std::cout << "-----------------------" << std::endl << "EPISODE " << (s+1) << std::endl;
			for (int y = 0; y < h; y++) {                 // Loop over image rows
				//fprintf(stderr, "\rTRAINING PHASE: %d Sample, Rendering (%d spp) %5.2f%%", s, num_training_samples, 100. * y / (h - 1));   // Print progress // @suppress("Invalid arguments")
				reward = 0;
				for (unsigned short x = 0, Xi[3] = { 0, 0, y * y * y }; x < w; x++) { // Loop cols. Xi = random seed
					const float& u = float(512*rand() / float(RAND_MAX)) / float(w);
					const float& v = float(512*rand() / float(RAND_MAX)) / float(h);
					Ray d = cam.get_ray(u, v);
					Struct_states state_rec;
					state_rec.old_state = {0,0,0,0,0};
					state_rec.old_action = -1;
					state_rec.old_id = -1;
					state_rec.prob = -1;
					radiance(Ray(cam.origin, d.d.norm()), 0, Xi, ptrPath_length, dict, counter_red, dictAction, state_rec, learning_phase, q_learning_mode, reward, dictStateActionCount, counter, epsilon); // The average is the same as averaging at the end
				}

				// Write reward to file
				avg_reward = reward/512;
				std::ofstream outfile;
				outfile.open("reward_epsilon_decay.txt", std::ios_base::app);
				outfile << avg_reward << " " << epsilon << std::endl;
				epsilon *= epsilon_decay;
			}
			if(s==0){
				std::ofstream outfile;
				outfile.open("average_visits.txt", std::ios_base::app);
				std::map<StateAction, StateActionCount> &addrDictStateActionCount = *dictStateActionCount;
				for(auto const &x : *dictStateActionCount) {
					outfile << x.second << std::endl;
				}
			}

		}

		std::cout << "NUMBER OF CENTER STATES: " << counter_red << std::endl;

		// Write Q value to a file
		ofstream myfile ("q-value-TEST.txt");
		if (myfile.is_open())
		{
			for(auto const &x : *dict) {
				for(int i=0; i<6; i++){
					myfile << x.first[i] << ",";
				}
				for(int i=0; i< dim_action_space+1; i++){
				  myfile << x.second[i] << ",";
				}
				myfile << std::endl;
			}
			myfile.close();
		}

		std::cout << "FILE CLOSED" << std::endl;
	}else if(q_table_available && learning_phase){
	// TRAINING PHASE, Q-Table available--------------------------------------------------------------

		std::ifstream infile( "q-value-TEST.txt" );
		std::string line;
		vector<string> results;
		std::array<float, 6> temp_state;
		std::array<float, dim_action_space+1> temp_qvalue;
		while (std::getline(infile, line))
		{
			std::istringstream iss(line);

			while (std::getline(iss, line, ',')) {
				results.push_back(line);
			}

			for(int i=0; i<6; i++){
				temp_state[i]=(float)std::stod(results[i]);
			}
			for(int i=0; i<dim_action_space+1; i++){
				temp_qvalue[i]=(float)std::stod(results[i+6]);
			}
			std::map<Key, QValue> &addrDict = *dict;
			addrDict[temp_state] = temp_qvalue;
			results.clear();
		}
			std::cout << "FILE CLOSED" << std::endl;
	}


	// ACTIVE PHASE -----------------------------------------------------------------------------------
	counter = 0;
	learning_phase=false;
	epsilon = 0;
	for (int y = 0, i = 0; y < h; y++) {                 // Loop over image rows
		fprintf(stderr, "\rACTIVE PHASE: Rendering (%d spp) %5.2f%%", samps, 100. * y / (h - 1));
		reward = 0;
		for (unsigned short x = 0, Xi[3] = { 0, 0, y * y * y }; x < w; x++) { // Loop cols. Xi = random seed
			for (int s = 0; s < samps; s++) {
				// u and v represents the percentage of the horizontal and vertical values
				const float& u = float(x - 0.5 + rand() / float(RAND_MAX)) / float(w);
				const float& v = float((h - y - 1) - 0.5 + rand() / float(RAND_MAX)) / float(h);
				Ray d = cam.get_ray(u, v);
				Struct_states state_rec;
				state_rec.old_state = {0,0,0,0,0};
				state_rec.old_action = -1;
				state_rec.old_id = -1;
				state_rec.prob = -1;
//				std::ofstream outfile;
//				outfile.open("lines.txt", std::ios_base::app);
//				outfile <<"---------------------------" << std::endl;
				r = r + radiance(Ray(cam.origin, d.d.norm()), 0, Xi, ptrPath_length, dict, counter_red, dictAction, state_rec, learning_phase, q_learning_mode, reward, dictStateActionCount, counter, epsilon) * (float) (1. / (2 * samps)); // The average is the same as averaging at the end
			} // Camera rays are pushed ^^^^^ forward to start in interior
			c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z));
			i++;
			r = Vec();
		}
	}
	std::cout << "PATH LENGTH: " << path_length / (samps * w * h) << std::endl;
	std::cout << "COUNTER RED : " << counter_red << std::endl;
	std::cout << "COUNTER : " << counter << std::endl;


	FILE *f = fopen("try.ppm", "w"); // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	for (int i = 0; i < w * h; i++)
		fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));

	// Calculate duration
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t2 - t1).count();
	std::cout << " DURATION : " << duration;
	delete dict;
	delete dictAction;
}

