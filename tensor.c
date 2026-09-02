#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

typedef struct Tensor Tensor;

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
  size_t visited;
}TensorGraph;

struct Tensor{
  TensorMetadata meta;
  float* data;
  float* grad;
  TensorGraph graph;
};

void empty_backward(Tensor* self) { }

Tensor* create_Tensor(int dims, int* shape, float* data){
  
  Tensor* cur_tensor = malloc(sizeof(Tensor)*1);
  
  cur_tensor->meta.dims = dims;
  cur_tensor->meta.shape = malloc(sizeof(int)*cur_tensor->meta.dims);
  memcpy(cur_tensor->meta.shape, shape, sizeof(int)*dims);
  cur_tensor->meta.size = 1;
  
  for (int i = 0; i < dims; i++) cur_tensor->meta.size *= cur_tensor->meta.shape[i];

  cur_tensor->data = calloc(cur_tensor->meta.size, sizeof(float));
  cur_tensor->grad = calloc(cur_tensor->meta.size, sizeof(float));

  if (data != NULL) memcpy(cur_tensor->data, data, sizeof(float) * cur_tensor->meta.size);
  
  cur_tensor->graph.backward = empty_backward;
 
  return cur_tensor;
}

Tensor* ones_tensor(int dims, int* shape){
  
  Tensor* cur_tensor = create_Tensor(dims, shape, NULL);

  for(int i = 0; i < cur_tensor->meta.size; i++){
    cur_tensor->grad[i] = 1.0f;
  }

  return cur_tensor;
}

Tensor* transpose(Tensor* t){
    Tensor* new_t = create_Tensor(2, (int[]){t->meta.shape[1], t->meta.shape[0]}, NULL);
    for(int i = 0; i < t->meta.shape[1]; i++)
        for(int j = 0; j < t->meta.shape[0]; j++)
            new_t->data[i * t->meta.shape[0] + j] = t->data[j * t->meta.shape[1] + i];
    return new_t;
}

void add_tensor_backward(Tensor* self){
    Tensor* a = self->graph.left;
    Tensor* b = self->graph.right;
    for(int i = 0; i < self->meta.size; i++){
        a->grad[i] += self->grad[i];
        b->grad[i] += self->grad[i];
    }
}

Tensor* a_tensor_float_matmul(float* f, Tensor* t, int dims, int* shape){
  
  Tensor* result = create_Tensor(dims, shape, NULL);

  for (int i = 0; i < shape[0]; i++){
    for(int j = 0; j < t->meta.shape[1]; j++){
      for (int k = 0; k < t->meta.shape[0]; k++){
        result->data[i * shape[1] + j] += f[i * t->meta.shape[0] + k] * t->data[k * t->meta.shape[1] + j];
      }
    }
  }
  return result;
}

Tensor* b_tensor_float_matmul(Tensor* t, float* f, int dims, int* shape){
  
  Tensor* result = create_Tensor(dims, shape, NULL);

  for (int i = 0; i < t->meta.shape[0]; i++){
    for(int j = 0; j < shape[1]; j++){
      for (int k = 0; k < t->meta.shape[1]; k++){
        result->data[i * shape[1] + j] += t->data[i * t->meta.shape[1] + k] * f[k * shape[1] + j];
      }
    }
  }
  return result;
}

void matmul_tensor_backward(Tensor* self){
  Tensor* a = self->graph.left;
  Tensor* b = self->graph.right;
  
  Tensor* a_t = transpose(a);
  Tensor* b_t = transpose(b);
  
  Tensor* d_a = a_tensor_float_matmul(self->grad, b_t, b->meta.dims, a->meta.shape);
  Tensor* d_b = b_tensor_float_matmul(a_t, self->grad, a->meta.dims, b->meta.shape);
  for (int i = 0; i < a->meta.size; i++){
    a->grad[i] += d_a->data[i];
  }
  
  for (int i = 0; i < b->meta.size; i++){
    b->grad[i] += d_b->data[i];
  }
}

void build_topo(Tensor* root, Tensor** topo, size_t* idx){
  
  if (!root->graph.op || root->graph.visited) return;
  
  root->graph.visited = 1;

  if (root->graph.left) build_topo(root->graph.left, topo, idx);
  if (root->graph.right) build_topo(root->graph.right, topo, idx);

  topo[(*idx)++] = root;
}

void tensor_backward(Tensor* root){
  
  Tensor** topo = malloc(sizeof(Tensor*) * 1000);
  size_t idx = 0;

  build_topo(root, topo, &idx);

  for (int i = 0; i < root->meta.size; i++) root->grad[i] = 1.0f;

  for(int i = idx-1; i >= 0; i--) topo[i]->graph.backward(topo[i]);
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

  Tensor* new_t = create_Tensor(a->meta.dims, a->meta.shape, NULL);

  for (int i = 0; i < new_t->meta.size; i++) {
    new_t->data[i] = a->data[i] + b->data[i];
  }

  new_t->graph.left = a;
  new_t->graph.right = b;
  new_t->graph.op = '+';
  new_t->graph.backward = add_tensor_backward;

  return new_t;
}

Tensor* matmul_tensor(Tensor* a, Tensor* b){

  if (a->meta.dims != b->meta.dims) {
    fprintf(stderr, "Dimensions count doesnt match\n");
    return NULL;
  }

  if (a->meta.dims != 3 && a->meta.dims != 2){
    fprintf(stderr, "Not Supported Dimension Set yet\n");
    return NULL;
  }

  if (a->meta.shape[1] != b->meta.shape[0]) {
    fprintf(stderr, "Cannot Multiple the Matrics A(Columns) != B(Rows)\n");
    return NULL;
  }

  int* shape = malloc(sizeof(int) * a->meta.dims); 
  memcpy(shape, a->meta.shape, sizeof(int)*a->meta.dims);
  shape[1] = b->meta.shape[1];
  
  Tensor* new_t = create_Tensor(a->meta.dims, shape, NULL);

  for (int i = 0; i < a->meta.shape[0]; i++){
    for (int j = 0; j < b->meta.shape[1]; j++){
      for (int k = 0; k < b->meta.shape[0]; k++){
        new_t->data[(i * b->meta.shape[1]) + j] += a->data[(i * a->meta.shape[1]) + k] * b->data[ (k * b->meta.shape[1]) + j ];
       }
    }
  }
  new_t->graph.left = a;
  new_t->graph.right = b;
  new_t->graph.op = '@';
  new_t->graph.backward = matmul_tensor_backward;

  return new_t;
}

int main(){
  
  int a[2] =  {2,3};
  int b[2] = {3,2};
  float data1[6] = {1,2,3,4,5,6};
  float data2[6] = {2,4,6,8,10,12};
  
  Tensor* t1 = create_Tensor(2, a, data1);
  Tensor* t2 = create_Tensor(2, b, data2);

  Tensor* result = matmul_tensor(t1, t2);

  for (int i = 0; i < result->meta.size; i++){
    printf("%f,  ", result->data[i]);
    if((i+1) % result->meta.shape[1] == 0) printf("\nNext Row\n");
  }

  Tensor* result2 = matmul_tensor(result, t1);

  tensor_backward(result2);
  printf("\n\n");
  for (int i = 0; i < t1->meta.size; i++ ) printf("%f,   ", t1->grad[i]);
}
