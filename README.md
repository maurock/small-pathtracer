# small-pathtracer

This path tracer is based on the Kevin Beason's smallpt.
http://www.kevinbeason.com/smallpt/

Some elements were modified to be more intuitive:
- Camera: I implemented the class Camera, and made it more intuitive (based on Peter Shirley Raytracer in One Weekend series).
- Multiple rays are shot for each pixel. Their distribution is obtained through uniform sampling, and not through a Tent Filter as in the original smallpt version.
- Gamma correction was modified.
- Everything is commented to be easily understandable.

Features implemented:

- Russian Roulette to stop rays after a specific number of bounces. Stopping rays after a specific number of bounces would introduce bias.
- Cosine-Weighted Importance Sampling.
- Uniform Sampling 
- Ray-Sphere Intersection.
- Ray-Plane Intersection.
- Tilted planes
- Explicit Light Sampling
![path_tracer](https://github.com/maurock/small-pathtracer/blob/master/comparison_uni_imp.png)


