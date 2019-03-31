# small-pathtracer

This path tracer is based on the Kevin Beason's smallpt.
http://www.kevinbeason.com/smallpt/

Some elements were modified to be more intuitive:
- Camera: I implemented the class Camera, and made it more intuitive (based on Peter SHirley Raytracer in One Weekend series).
- Multiple rays are shot for each pixel. Their distribution is obtained through uniform sampling, and not through a Tent Filter as in the original smallpt version.
- Everything is commented to be easily understandable.
