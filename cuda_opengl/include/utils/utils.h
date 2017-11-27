#pragma once

#include <string>

namespace utils
{
  unsigned int nextPow2(unsigned int v);

  /// <summary>
  /// Checks whether a string is an hexadecimal number of not.
  /// </summary>
  /// <param name="s">String containing an hexadecimal number or not.</param>
  /// <returns>
  ///   <c>true</c> if the specified string is hexadecimal; otherwise, <c>false</c>.
  /// </returns>
  bool isHexa(const std::string& s);
}
