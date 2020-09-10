#ifndef COLOR_HPP
#define COLOR_HPP

#include <nextweek/vec3.cuh>
#include <nextweek/external.hpp>

void write_color(std::ostream &out, Color pixel_color) {
  // Write the translated [0,255] value of each color component.
  out << static_cast<int>(255.999 * pixel_color.x()) << " "
      << static_cast<int>(255.999 * pixel_color.y()) << " "
      << static_cast<int>(255.999 * pixel_color.z()) << std::endl;
}

#endif
