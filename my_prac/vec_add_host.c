#include <stdio.h>
#include <stdlib.h>

#define N 1024

void vec_add(float* h_a, float* h_b, float* h_c, int n){
  for(int i=0; i<n; i++) {
    h_c[i] = h_a[i] + h_b[i];
  }
}

void init_vector(float *vec, int n) {
  for(int i=0; i<n; i++) {
    vec[i] = (float)rand() / RAND_MAX;
  }
}

int main() {
  float *h_a, *h_b, *h_c;
  int size = N * sizeof(float);

  // alloting memory in cpu
  h_a = (float*)malloc(size);
  h_b = (float*)malloc(size);
  h_c = (float*)malloc(size);

  // generating random values in vector
  init_vector(h_a, N);
  init_vector(h_b, N);

  // adding vectors
  vec_add(h_a, h_b, h_c, N);

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
