#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main () {
  auto s8 = make_layout(Int<8>{});
  auto d8 = make_layout(8);

  print(s8); print("\n");
  print(d8); print("\n");

  return 0;
}
