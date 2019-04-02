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
using namespace std;
using namespace std::chrono;

#define RAND48_MULT_0   (0xe66d)
#define RAND48_MULT_1   (0xdeec)
#define RAND48_MULT_2   (0x0005)
#define RAND48_ADD      (0x000b)

const int NUMBER_OBJ = 10;
unsigned short _rand48_add = RAND48_ADD;
unsigned short _rand48_mult[3] = {
    RAND48_MULT_0,
    RAND48_MULT_1,
    RAND48_MULT_2
};

void _dorand48(unsigned short xseed[3])
{
    unsigned long accu;
    unsigned short temp[2];

    accu = (unsigned long)_rand48_mult[0] * (unsigned long)xseed[0] +
        (unsigned long)_rand48_add;
    temp[0] = (unsigned short)accu;        /* lower 16 bits */
    accu >>= sizeof(unsigned short) * 8;
    accu += (unsigned long)_rand48_mult[0] * (unsigned long)xseed[1] +
        (unsigned long)_rand48_mult[1] * (unsigned long)xseed[0];
    temp[1] = (unsigned short)accu;        /* middle 16 bits */
    accu >>= sizeof(unsigned short) * 8;
    accu += _rand48_mult[0] * xseed[2] + _rand48_mult[1] * xseed[1] + _rand48_mult[2] * xseed[0];
    xseed[0] = temp[0];
    xseed[1] = temp[1];
    xseed[2] = (unsigned short)accu;
}

double erand48(unsigned short xseed[3])
{
    _dorand48(xseed);
    return ldexp((double)xseed[0], -48) +
        ldexp((double)xseed[1], -32) +
        ldexp((double)xseed[2], -16);
}

struct Vec {        // Usage: time ./smallpt 5000 && xv image.ppm
  double x, y, z;                  // position, also color (r,g,b)
  Vec(double x_=0, double y_=0, double z_=0){ x=x_; y=y_; z=z_; }
  Vec operator+(const Vec &b) const { return Vec(x+b.x,y+b.y,z+b.z); }
  Vec operator-(const Vec &b) const { return Vec(x-b.x,y-b.y,z-b.z); }
  Vec operator*(double b) const { return Vec(x*b,y*b,z*b); }
  Vec operator*(float b) const { return Vec(x*b,y*b,z*b); }
  Vec operator*(int b) const { return Vec(x*b,y*b,z*b); }

  Vec mult(const Vec &b) const { return Vec(x*b.x,y*b.y,z*b.z); }
  Vec& norm(){ return *this = *this * (1/sqrt(x*x+y*y+z*z)); }
  double dot(const Vec &b) const { return x*b.x+y*b.y+z*b.z; }
  Vec operator%(Vec &b){return Vec(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);} // cross
};

struct Ray { Vec o, d; Ray(Vec o_, Vec d_) : o(o_), d(d_) {} };
enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()


class Sphere{
	public:
		double rad;       // radius
		Vec p, e, c;      // position, emission, color
		Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
		Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_):
		rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}

		double intersect(const Ray &r) const { // returns distance, 0 if no hit
			Vec op = p-r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
			double t, eps=1e-4;
			double b = op.dot(r.d);
			double det = b*b-op.dot(op)+rad*rad;
			if (det<0) return 0; else det=sqrt(det);

			return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : 0);
		}
};

/*
class Rectangle_xz {
	public:
		double x1, x2, z1, z2, y;
		Vec e, c;         // emission, color
		Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)

		Rectangle_xz(double x1_,  double x2_, double z1_, double z2_, double y_, Vec e_, Vec c_, Refl_t refl_):
		x1(x1_), x2(x2_), z1(z1_), z2(z2_), y(y_), e(e_), c(c_), refl(refl_) {}
		double intersect(const Ray &r) const { // returns distance, 0 if no hit
			double t = (y-r.o.y)/r.d.y;		// ray.y = t* dir.y
			float x = r.o.x + r.d.x*t;
			float z = r.o.z + r.d.z*t;
			if(x < x1 || x > x2 || z < z1 || z > z2){
				return 0;
			}else{
				return t;
			}
		}
};*/

class Camera{
	public:
		// lookfrom is the origin
		// lookat is the point to look at
		// vup, the view up vector to project on the new plane when we incline it. We can also tilt
		// the plane
		Camera(Vec lookfrom, Vec lookat, Vec vup, float vfov, float aspect){// vfov is top to bottom in degrees, field of view on the vertical axis
			Vec w, u, v;
			float theta = vfov * M_PI/180;	// convert to radiants
			float half_height = tan(theta/2);
			float half_width = aspect * half_height;
			origin = lookfrom;
			w = (lookat-lookfrom).norm();
			u = (w%vup).norm();
			v = (u%w);

			lower_left_corner = origin - u*half_width - v*half_height + w;
			horizontal = u*(half_width*2);
			vertical = v*(half_height*2);
		}
		Ray get_ray(float s, float t){ return Ray(origin, lower_left_corner + horizontal*s + vertical*t - origin); }

		Vec origin;
		Vec lower_left_corner;
		Vec horizontal;
		Vec vertical;
};


Sphere spheres[] = {
	Sphere(1e5, Vec( 1e5+1,40.8,81.6), Vec(),Vec(.25,.75,.25),DIFF), //Scene: radius, position, emission, color, material
	Sphere(1e5, Vec( 1e5+1,40.8,81.6), Vec(),Vec(.25,.75,.25),DIFF),//Left
	Sphere(1e5, Vec(-1e5+99,40.8,81.6),Vec(),Vec(.75,.25,.25),DIFF),//Rght
	Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),Vec(.75,.75,.75),DIFF),//Back
	Sphere(1e5, Vec(50,40.8,-1e5+170), Vec(),Vec(),           DIFF),//Frnt
	Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),Vec(.75,.75,.75),DIFF),//Botm
	Sphere(1e5, Vec(50,-1e5+81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF),//Top
	Sphere(16.5,Vec(27,16.5,47),       Vec(),Vec(1,1,1)*.999, DIFF),//Mirr
	//Sphere(16.5,Vec(73,16.5,78),       Vec(),Vec(1,1,1)*.999, REFR),//Glas
	Sphere(16.5,Vec(73,16.5,78),		Vec(),Vec(.75,.75,.75), DIFF),//Glas
	Sphere(600, Vec(50,681.6-.27,81.6),Vec(12,12,12),  Vec(), DIFF) //Light
};

// clamp makes sure that the set is bounded (used for radiance() )
inline double clamp(double x){ return x<0 ? 0 : x>1 ? 1 : x; }

// toInt() applies a gamma correction of 2.2, because our screen doesn't show colors linearly
inline int toInt(double x){ return int(pow(clamp(x),1/2.2)*255+.5); }

inline bool intersect(const Ray &r, double &t, int &id){
  double n= sizeof(spheres)/sizeof(Sphere); //Divide allocation of byte of the whole scene, by allocation in byte of one single element
  double d;
  double inf=t=1e20;
  for(int i=int(n);i--;) {
	  if((d=spheres[i].intersect(r))&&d<t){	// Distance of hit point
		  t=d;
		  id=i;
	  }
  }
  // Return the closest intersection, as a bool
  return t<inf;
}

inline Vec random_scattering(const Vec& nl, unsigned short *Xi){

	// COSINE-WEIGHTED SAMPLING
	double r1 = 2*M_PI*erand48(Xi);		// get random angle
	double r2 = erand48(Xi);			// get random distance from center
	double r2s = sqrt(r2);
	// Create orthonormal coordinate frame for scattered ray
	Vec w = nl;			// w = normal
	Vec u = ((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm();
	Vec v = w%u;
	return (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm();   // reflection ray with cosine sampling (check calculus)

	/*
	// UNIFORM SAMPLING
	double r1 = 2*M_PI*erand48(Xi);		// get random angle Gamma
	double r2 = erand48(Xi);			// get random distance from center
	// Create orthonormal coordinate frame for scattered ray
	Vec w = nl;			// w = normal
	Vec u = ((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm();
	Vec v = w%u;
	return (u*cos(r1)*sqrt(r2*(2-r2)) + v*sin(r1)*sqrt(r2*(2-r2)) + w*(1-r2)).norm();   // random reflection ray */
}

Vec light_sampling(const Vec& nl, const Vec& x, unsigned short *Xi, int* ptrDepth, bool* ptrHitlight){
	// SAMPLING LIGHT (HARD-CODED)
	Vec light_vec = Vec(50,81.6, 81.6) - x;

	if(light_vec.dot(nl)<0){
		return random_scattering(nl, Xi);
	}
	*ptrDepth = *ptrDepth + 10;
	*ptrHitlight = true;
	//std::cout << *ptrDepth << std::endl;	//*ptrDepth = 20;
	return light_vec;
}


Vec radiance(const Ray &r, int depth, unsigned short *Xi){
  double t;                             // distance to intersection
  int id=0;                             // id of intersected object

  if (!intersect(r, t, id)) return Vec(); // if miss, return black
  	  	  	  	  	  	  	  	  	  	  // in the function, it also sets the t to the closest hitting distance
  Sphere &obj = spheres[id];              // the hit object
  Vec x = r.o+r.d*t;					// ray intersection point (t calculated in intersect())
  Vec n = (x - obj.p).norm();			// sphere normal
  Vec nl = n.dot(r.d)<0?n:n*-1;			// properly orient the normal. If I am inside the sphere, the normal needs to point towards the inside
  										// indeed, the angle would be < 90, so dot() < 0. Also, if in a glass it enters or exits
  Vec f =  obj.c;     					// sphere color
  double p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; // max reflectivity (maximum component of r,g,b)
  if (++depth>5 || !p){
	  if (erand48(Xi)<p){
		  f=f*(1/p);
	  }
	  else {
		  return obj.e; // Russian Roulette. After 5 bounces, it determines if the ray continues or stops.
	  }
  }
  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  //This is based on the reflectivity, and the BRDF scaled to compensate for it.
  if (obj.refl == DIFF){                // Ideal DIFFUSE reflection.
	// TOTAL RANDOM SCATTERING
	Vec d;
	d = random_scattering(nl, Xi);
    return obj.e + f.mult(radiance(Ray(x,d),depth,Xi));				// get color in recursive function
  }

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
    radiance(reflRay,depth,Xi)*Re+radiance(Ray(x,tdir),depth,Xi)*Tr);
}

// LOOPS OVER IMAGE PIXELS, SAVES TO PPM FILE
int main(int argc, char *argv[]){
	 high_resolution_clock::time_point t1 = high_resolution_clock::now();

  // Set up image
  int w=512, h=512; // Resolution
  int samps = 16; 	// Number samples
  Vec r;	// Used for colors of samples
  Vec *c = new Vec[w*h]; // The image

  // Set up camera
  Camera cam(Vec(50,40,168), Vec(50,40, 5), Vec(0,1,0), 65, float(w)/float(h));

 // #pragma omp parallel for schedule(dynamic, 1) private(r)       // OpenMP. Each loop should be run in its own thread
  // LOOP OVER ALL IMAGE PIXELS
 for (int y=0, i=0; y<h; y++){                       // Loop over image rows
    fprintf(stderr,"\rRendering (%d spp) %5.2f%%",samps,100.*y/(h-1));   // Print progress
    for (unsigned short x=0, Xi[3]={0,0,y*y*y}; x<w; x++) {  // Loop cols. Xi = random seed
    	for (int s=0; s<samps; s++){
    		 // u and v represents the percentage of the horizontal and vertical values
        	 float u = float(x - 0.5 + rand() / double(RAND_MAX)) / float(w);
        	 float v = float((h-y-1) - 0.5 + rand() / double(RAND_MAX)) / float(h);
        	 Ray d = cam.get_ray(u,v);
             r = r + radiance(Ray(cam.origin,d.d.norm()),0,Xi)*(1./samps);  // The average is the same as averaging at the end
          } // Camera rays are pushed ^^^^^ forward to start in interior
    	c[i] = c[i] + Vec(clamp(r.x),clamp(r.y),clamp(r.z));
    	i++;
    	r = Vec();
   }
  }
  FILE *f = fopen("image14.ppm", "w");         // Write image to PPM file.
  fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
  for (int i=0; i<w*h; i++)
    fprintf(f,"%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
  std::cout << " DURATION : " << duration;
}


