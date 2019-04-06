# small-pathtracer

This path tracer is based on the Kevin Beason's smallpt.
http://www.kevinbeason.com/smallpt/

Some elements were modified to be more intuitive:
- Camera: I implemented the class Camera, and made it more intuitive (based on Peter Shirley Raytracer in One Weekend series).
- Multiple rays are shot for each pixel. Their distribution is obtained through uniform sampling, and not through a Tent Filter as in the original smallpt version.
- When a ray hits the light source, it immediately returns the emittance. In the original version, rays bounce at least 5 times and they stop later according to a Russian Roulette technique.
- Gamma correction was modified.
- Everything is commented to be easily understandable.

Features implemented:

- Russian Roulette to stop rays after a specific number of bounces. Stopping rays after a specific number of bounces would introduce bias.
- Cosine-Weighted Importance Sampling.
- Uniform Sampling 
- Monte Carlo Path Tracing.
- Ray-Sphere Intersection.
- Explicit Light Sampling

![path_tracer](https://user-images.githubusercontent.com/30290271/55671264-a3e7f980-588e-11e9-9b87-931fddcfe9b1.png)

<i>From left to right:
- Random sampling. Sample per pixel: 32. Rendering time: 83s
- Cosine-weighted Monte Carlo importance sampling. Sample per pixel: 32 spp. Rendering time: 84s. 
- Explicit light sampling. Sample per pixel: 32 spp. Rendering time:31s. 
</i>
