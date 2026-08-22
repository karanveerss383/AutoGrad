#include <stdio.h>
#include <string.h>
#include <math.h>
#include "backward.h"
#include "autograd.h"
#include "utils.h"
#include "activations.h"
#include "weight_init.h"


typedef struct{
  int dims;
  int* shape;
  int size;
}TensorMetadata;

typedef struct{
  Tensor* left;
  Tensor* right;
  void (*backward)(Tensor*);
  char op;
}TensorGraph

typedef struct{
  TensorMetadata meta;
  float* data;
  float* grad;
  TensorGraph graph;
}Tensor;

Tensor* create_Tensor(int dims, int* shape){
  
  Tensor* cur_tensor = malloc(sizeof(Tensor)*1);
  
  cur_tensor->meta.dims = dims;
  cur_tensor->meta.shape = malloc(sizeof(int)*cur_tensor->meta.dims);
  memcpy(cur_tensor->meta.shape, shape, sizeof(int)*dims);
  cur_tensor->meta.size = 1;
  
  for (int i = 0; i < dims; i++) cur_tensor->meta.size *= cur_tensor->meta.shape[i];

  cur_tensor->data = calloc(cur_tensor->meta.size, sizeof(float));
  cur_tensor->grad = calloc(cur_tensor->meta.size, sizeof(float));

  return cur_tensor;
}

void add_tensor_backward(Tensor* self){
    Tensor* a = self->graph.left;
    Tensor* b = self->graph.right;
    for(int i = 0; i < self->meta.size; i++){
        a->grad[i] += self->grad[i];
        b->grad[i] += self->grad[i];
    }
}

Tensor* add_tensor(Tensor* a, Tensor*  b){
  
  if (a->meta.dims != b->meta.dims) {
    fprintf(stderr, "Dimensions count doesnt match\n");
    return NULL;
  }
  if (a->meta.size != b->meta.size) {
    fprintf(stderr, "Size of Tensors doesnt match\n");
    return NULL;
  }
  for (int i = 0; i < a->meta.dims; i++){
    if (a->meta.shape[i] != b->meta.shape[i]) {
      fprintf(stderr, "Shape of Tensors doesnt match\n");
      return NULL;
    }
  }

  Tensor* new_t = create_Tensor(a->meta.dims, a->meta.shape);

  for (int i = 0; i < new_t->meta.size; i++) {
    new_t->data[i] = a->data[i] + b->data[i];
  }

  new_t->graph.left = a;
  new_t->graph.right = b;
  new_t->graph.op = '+';
  new_t->graph.backward = add_tensor_backward;

  return new_t;
}

Tensor* mul_tensor(Tensor* a, Tensor* b){

}


int main(){
  
  int a[2] =  {6,6};
  Tensor* t = create_Tensor(2, a);

  for (int i =0 ;i < t->meta.size; i++){
    printf("%f\n", t->data[i]);
  }
  printf("%d", t->meta.size);

  return 0;
}
