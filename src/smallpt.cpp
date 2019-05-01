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
#include "utilities.h"
using namespace std;
using namespace std::chrono;

const int NUMBER_OBJ = 17;
using Key = std::array<float, 3>;
using QValue = std::array<float, 3>;
using ColorValue = std::array<float, 3>;

struct Vec {
	double x, y, z;                  // position, also color (r,g,b)
	Vec(double x_ = 0, double y_ = 0, double z_ = 0) {
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
	Vec operator*(double b) const {
		return Vec(x * b, y * b, z * b);
	}
	Vec operator*(float b) const {
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
	double dot(const Vec &b) const {
		return x * b.x + y * b.y + z * b.z;
	}
	Vec operator%(Vec &b) {
		return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
	} // cross
	double magnitude() {
		return sqrt(x * x + y * y + z * z);
	}
};

// LOOKFROM for the Camera
const Vec LOOKFROM = Vec(50, 40, 168);

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

class Hitable {			// a pure virtual function makes sure we always override the function hit
public:
	virtual double intersect(const Ray &r) const = 0;
	virtual bool intersect(const Vec &vec) const = 0;
	virtual Vec normal(const Ray &r, Hit_records &hit, Vec &x) const = 0;
	virtual std::array<float, 3> add_key(Vec &pos) const = 0;
	virtual std::array<float, 3> add_value(std::array<float, 3>& x_reduced) const = 0;

};

class Rectangle_xz: public Hitable {
public:
	double x1, x2, z1, z2, y;
	Vec e, c;         // emission, color
	Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
	Rectangle_xz(double x1_, double x2_, double z1_, double z2_, double y_,
			Vec e_, Vec c_, Refl_t refl_) :
			x1(x1_), x2(x2_), z1(z1_), z2(z2_), y(y_), e(e_), c(c_), refl(refl_) {
	}

	double intersect(const Ray &r) const { // returns distance, 0 if no hit
		double t = (y - r.o.y) / r.d.y;		// ray.y = t* dir.y
		float x = r.o.x + r.d.x * t;
		float z = r.o.z + r.d.z * t;
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

	std::array<float, 3> add_key(Vec& pos) const {
		Vec x_reduced = Vec(ceil((float) pos.x / 10), pos.y / 10,ceil((float) pos.z / 10));
		return { x_reduced.x, x_reduced.y, x_reduced.z};
	}

	std::array<float, 3> add_value(std::array<float, 3>& x_reduced) const {
		return { x_reduced[0] / 10 * (rand() / float(RAND_MAX)), x_reduced[1] * (rand() / float(RAND_MAX)),
				x_reduced[2]  / 10 * (rand() / float(RAND_MAX)) };
	}
};

class Rectangle_xy: public Hitable {
public:
	double x1, x2, y1, y2, z;
	Vec e, c;         // emission, color
	Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
	Rectangle_xy(double x1_, double x2_, double y1_, double y2_, double z_,	Vec e_, Vec c_, Refl_t refl_) :
			x1(x1_), x2(x2_), y1(y1_), y2(y2_), z(z_), e(e_), c(c_), refl(refl_) {}

	double intersect(const Ray &r) const { // returns distance, 0 if no hit
		double t = (z - r.o.z) / r.d.z;
		float x = r.o.x + r.d.x * t;
		float y = r.o.y + r.d.y * t;
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

	std::array<float, 3> add_key(Vec& pos) const {
		Vec x_reduced = Vec(ceil((float) pos.x / 10), ceil((float) pos.y / 10),	pos.z / 10);
		return { x_reduced.x, x_reduced.y, x_reduced.z};
	}

	std::array<float, 3> add_value(std::array<float, 3>& x_reduced) const {
		return { x_reduced[0] / 10 * (rand() / float(RAND_MAX)), x_reduced[1] 	/ 10 * (rand() / float(RAND_MAX)),
				x_reduced[2]  * (rand() / float(RAND_MAX)) };
	}
};

class Rectangle_yz: public Hitable {
public:
	double y1, y2, z1, z2, x;
	Vec e, c;         // emission, color
	Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
	Rectangle_yz(double y1_, double y2_, double z1_, double z2_, double x_,	Vec e_, Vec c_, Refl_t refl_) :
			y1(y1_), y2(y2_), z1(z1_), z2(z2_), x(x_), e(e_), c(c_), refl(refl_) {}

	double intersect(const Ray &r) const { // returns distance, 0 if no hit
		double t = (x - r.o.x) / r.d.x;
		float y = r.o.y + r.d.y * t;
		float z = r.o.z + r.d.z * t;
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

	std::array<float, 3> add_key(Vec& pos) const {
		Vec x_reduced = Vec( pos.x / 10, ceil((float) pos.y / 10),	ceil((float) pos.z / 10));
		return { x_reduced.x, x_reduced.y, x_reduced.z};
	}

	std::array<float, 3> add_value(std::array<float, 3>& x_reduced) const {
		return { x_reduced[0] * (rand() / float(RAND_MAX)), x_reduced[1] 	/ 10 * (rand() / float(RAND_MAX)),
				x_reduced[2]  / 10 * (rand() / float(RAND_MAX)) };
	}
};

class Sphere: public Hitable {
public:
	double rad;       // radius
	Vec p, e, c;      // position, emission, color
	Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
	Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) :	rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
	double intersect(const Ray &r) const { // returns distance, 0 if no hit
		Vec op = p - r.o; 	// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		double t, eps = 1e-4;
		double b = op.dot(r.d);
		double det = b * b - op.dot(op) + rad * rad;
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
	Ray get_ray(float s, float t) {
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
	new Rectangle_yz(0, 81.6, 0, 170, 1, Vec(), Vec(.25, .75, .25), DIFF),		// Left
	new Rectangle_yz(0, 81.6, 0, 170, 99, Vec(), Vec(.75, .25, .25), DIFF),		// Right
	new Rectangle_xz(1, 99, 0, 170, 0, Vec(), Vec(.75, .75, .75), DIFF),		// Bottom
	new Rectangle_xz(1, 99, 0, 170, 81.6, Vec(), Vec(.75, .75, .75), DIFF),		// Top
	new Rectangle_xz(32, 68, 63, 96, 81.5, Vec(12, 12, 12), Vec(), DIFF),		// Light


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
inline double clamp(double x) {
	return x < 0 ? 0 : x > 1 ? 1 : x;
}

// toInt() applies a gamma correction of 2.2, because our screen doesn't show colors linearly
inline int toInt(double x) {
	return int(pow(clamp(x), 1 / 2.2) * 255 + .5);
}

inline bool intersect(const Ray &r, double &t, int &id) {
	double n = NUMBER_OBJ; //Divide allocation of byte of the whole scene, by allocation in byte of one single element
	double d;
	double inf = t = 1e20;
	for (int i = 0; i < n; i++) {
		if ((d = rect[i]->intersect(r)) && d < t) {	// Distance of hit point
			t = d;
			id = i;
		}
	}
	// Return the closest intersection, as a bool
	return t < inf;
}

inline Vec random_scattering(const Vec& nl, unsigned short *Xi) {

	// COSINE-WEIGHTED SAMPLING
	double r1 = 2 * M_PI * erand48(Xi);		// get random angle
	double r2 = erand48(Xi);			// get random distance from center
	double r2s = sqrt(r2);
	// Create orthonormal coordinate frame for scattered ray
	Vec w = nl;			// w = normal
	Vec u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm();
	Vec v = w % u;
	return (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();

	// reflection ray with cosine sampling (check calculus)

	/*
	 // UNIFORM SAMPLING
	 double r1 = 2*M_PI*erand48(Xi);		// get random angle Gamma
	 double r2 = erand48(Xi);			// get random distance from center
	 // Create orthonormal coordinate frame for scattered ray
	 Vec w = nl;			// w = normal
	 Vec u = ((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm();
	 Vec v = w%u;
	 return (u*cos(r1)*sqrt(r2*(2-r2)) + v*sin(r1)*sqrt(r2*(2-r2)) + w*(1-r2)).norm();   // random reflection ray
	 */
}

inline Vec light_sampling(const Vec& nl, const Vec& hit, unsigned short *Xi) {
	// SAMPLING LIGHT (HARD-CODED)
	double x_light = 32 + rand() * 36 / double(RAND_MAX);
	double z_light = 63 + rand() * 36 / double(RAND_MAX);
	Vec light_vec = Vec(x_light, 81.6, z_light) - hit;
	return light_vec;
}

inline Vec hittingPoint(const Ray &r, int &id) {
	double t;                             // distance to intersection
	if (!intersect(r, t, id))
		return Vec();
	Vec x = r.o + r.d * t;// ray intersection point (t calculated in intersect())
	return x;
}

inline Vec normal(Vec &point, Vec &center) {
	return (point - center).norm();
}

inline int create_state_space(std::map<Key, QValue> *dict) {
	std::map<Key, QValue> &addrDict = *dict;
	int count = 0;
	for (int x = 0; x < 100; x++) {
		for (int y = -1; y < 85; y++) {
			for (int z = -1; z < 171; z++) {
				Vec vec = Vec(x, y, z);
				Ray r = Ray(LOOKFROM, (vec - LOOKFROM).norm());
				int id = 0;
				Vec pos = hittingPoint(r, id);
				Key key = rect[id]->add_key(pos);
				QValue value = rect[id]->add_value(key);

				if (addrDict.count(key) < 1) {
					addrDict[key] = value;
					if(key[0]==0 || key[1] == 0 || key[2] == 0){
						std::cout << key[0] << " " << key[1] << " " << key[2] << std::endl;}
					count += 1;
				}
			}
		}
	}
	return count;
}

inline Vec center_state(std::map<Key, QValue> *dict){
	for (auto const& x : *dict)
	{
	    Vec state_red = ((x.first[0]*10)+5, (x.first[1]*10)+5, (x.first[2]*10)+5);
	    Ray r = Ray(LOOKFROM, (state_red - LOOKFROM).norm());
	    for(int id = 0; id < NUMBER_OBJ; id ++){
	    	Vec x = hittingPoint(r, id);
	    }
	}
}

inline Vec radiance(const Ray &r, int depth, unsigned short *Xi, double *path_length, std::map<Key, QValue> *dict, int& counter_red) {
	Hit_records hit;
	int id = 0;                           // initialize id of intersected object
	Vec x = hittingPoint(r, id);            // id calculated inside the function

	std::map<Key, QValue> &addrDict = *dict;

	Key key = rect[id]->add_key(x);
	/*
	Vec x_reduced = Vec(ceil((float) x.x / 10), ceil((float) x.y / 10), ceil((float) x.z / 10));
	Key key = { x_reduced.x, x_reduced.y, x_reduced.z };
	*/
	//std::cout << "KEY: " << key[0] << "    X.X: " << x.x << "    KEY[O]*10 - 5: "  << key[0]*10- 5 << std::endl;

	// COLOR RED STATE
	if( ((x.x > (key[0]*10- 6)) && (x.x < (key[0]*10- 4)) && (x.y > (key[1]*10- 6)) && (x.y < (key[1]*10- 4))) ||
			((x.x > (key[0]*10- 6)) && (x.x < (key[0]*10- 4)) && (x.z > (key[2]*10- 6)) && (x.z < (key[2]*10- 4)))||
			((x.y > (key[1]*10- 6)) && (x.y < (key[1]*10- 4)) && (x.z > (key[2]*10- 6)) && (x.z < (key[2]*10- 4)))){
			counter_red = counter_red + 1;
			return Vec(1,0,0);
	}
	//////////////////////

	return Vec(addrDict[key][0], addrDict[key][1], addrDict[key][2]);

	Hitable* &obj = rect[id];				// the hit object
	Vec nl = obj->normal(r, hit, x);
	Vec f = hit.c;							// object color
	double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max reflectivity (maximum component of r,g,b)
	if (++depth > 5 || !p) {// Russian Roulette. After 5 bounces, it determines if the ray continues or stops.
		if (erand48(Xi) < p) {
			f = f * (1 / p);
		} else {
			return hit.e;
		}
	}

	// This is based on the reflectivity, and the BRDF scaled to compensate for it.
	if (hit.refl == DIFF) {                   // Ideal DIFFUSE reflection.
		// TOTAL RANDOM SCATTERING
		Vec d;
		double q = rand() / double(RAND_MAX);
		double PDF_inverse = 1;
		double BRDF = 1;
		double t; 	// distance to intersection
		if (q < 1) {
			d = light_sampling(nl, x, Xi);
			intersect(Ray(x, d.norm()), t, id);
			if (id != 6) {
				d = random_scattering(nl, Xi);
				intersect(Ray(x, d.norm()), t, id);
			} else {
				PDF_inverse = fabs((1296 * d.norm().dot(Vec(0, 1, 0))) / (t * t));	//PDF = r^2 / (A * cos(theta_light))
				BRDF = fabs(d.norm().dot(nl) / M_PI);
			}
		} else {
			d = random_scattering(nl, Xi);
			intersect(Ray(x, d.norm()), t, id);
		}
		*path_length = *path_length + t;
		return hit.e + f.mult(radiance(Ray(x, d.norm()), depth, Xi, path_length, dict, counter_red)) * PDF_inverse * BRDF;// get color in recursive function
	}
	/*
	 else if (obj.refl == SPEC)            // Ideal SPECULAR reflection
	 return obj.e + f.mult(radiance(Ray(x,r.d-n*2*n.dot(r.d)),depth,Xi));

	 Ray reflRay(x, r.d-n*2*n.dot(r.d));     // Ideal dielectric REFRACTION
	 bool into = n.dot(nl)>0;                // Ray from outside going in?
	 double nc=1, nt=1.5, nnt=into?nc/nt:nt/nc, ddn=r.d.dot(nl), cos2t;
	 if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0)    // Total internal reflection
	 return obj.e + f.mult(radiance(reflRay,depth,Xi));
	 Vec tdir = (r.d*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm();
	 double a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:tdir.dot(n));
	 double Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP=Tr/(1-P);
	 return obj.e + f.mult(depth>2 ? (erand48(Xi)<P ?   // Russian roulette
	 radiance(reflRay,depth,Xi)*RP:radiance(Ray(x,tdir),depth,Xi)*TP) :
	 radiance(reflRay,depth,Xi)*Re+radiance(Ray(x,tdir),depth,Xi)*Tr);	*/
}

// WHEN RAY BOUNCES, IF IT HITS THE NOT VISIBLE SIDE OF THE SPHERE THEY KEY DOESNT EXIST
// IN THAT CASE, SCATTER RANDOMLY

// LOOPS OVER IMAGE PIXELS, SAVES TO PPM FILE
int main(int argc, char *argv[]) {
	srand(time(NULL));
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// Set up image
	int w = 512, h = 512; 		// Resolution
	int samps = 16; 			// Number samples
	Vec r;					// Used for colors of samples
	Vec *c = new Vec[w * h]; 	// The image
	std::map<Key, QValue>* dict = new std::map<Key, QValue>;

	// TEMPORARY COUNTER
	int counter_red = 0;

	// Create states
	int number_states = create_state_space(dict);
	std::cout << "NUMBER STATES: " << number_states << std::endl;

	// Set up camera
	Camera cam(LOOKFROM, Vec(50, 40, 5), Vec(0, 1, 0), 65, float(w) / float(h));

	// Average path length
	double path_length = 0;
	double* ptrPath_length = &path_length;
	// #pragma omp parallel for schedule(dynamic, 1) private(r)       // OpenMP. Each loop should be run in its own thread
	// LOOP OVER ALL IMAGE PIXELS
	for (int y = 0, i = 0; y < h; y++) {                 // Loop over image rows
		fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps, 100. * y / (h - 1));   // Print progress
		for (unsigned short x = 0, Xi[3] = { 0, 0, y * y * y }; x < w; x++) { // Loop cols. Xi = random seed
			for (int s = 0; s < samps; s++) {
				// u and v represents the percentage of the horizontal and vertical values
				float u = float(x - 0.5 + rand() / double(RAND_MAX)) / float(w);
				float v = float((h - y - 1) - 0.5 + rand() / double(RAND_MAX)) / float(h);
				Ray d = cam.get_ray(u, v);
				r = r + radiance(Ray(cam.origin, d.d.norm()), 0, Xi, ptrPath_length, dict, counter_red) * (1. / samps); // The average is the same as averaging at the end
			} // Camera rays are pushed ^^^^^ forward to start in interior
			c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z));
			i++;
			r = Vec();
		}
	}
	std::cout << "PATH LENGTH: " << path_length / (samps * w * h) << std::endl;

	std::cout << "COUNTER RED : " << counter_red << std::endl;


	FILE *f = fopen("show_allrect_differentplane_red_state.ppm", "w"); // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	for (int i = 0; i < w * h; i++)
		fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));

	// Calculate duration
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t2 - t1).count();
	std::cout << " DURATION : " << duration;
}

