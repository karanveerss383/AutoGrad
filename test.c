#include "stdio.h"
#include "weight_init.h"
#include "utils.h"
#include "autograd.h"

int main(){
  
  Layer L1;
  createLayer(&L1, 1, 1, 6, 9);
  for(int i = 0; i < L1.fan_out; i++){
    printf("Neuron %d\n : ", (i+1));
    for (int j = 0; j < L1.fan_in; j++){
      
      printf("w%d - %f\n", (j+1), L1.neurons[i].weights[j].data);
    
    }
    printf("b - %f\n", L1.neurons[i].bias.data);
  }

  float a[] = {0.5, 0.5, 0.5, 0.5 , 0.5, 0.5, 1, 1, 1, 1, 1, 1};
  Value* input = init_Value(a, 12);

  Value*** out_matmul = batch_matmul(&L1, input, 2);
  
  printf("MATMUL OUTPUT STARTS: \n");
  
  for(int batch = 0; batch < 2; batch++){
    
    printf("Batch %d\n",batch);

    for (int i = 0; i < L1.fan_out; i++){
  
      printf("Neuron-%d\n output -: %f\n", (i+1), out_matmul[batch][i]->data);
    }
  
  }
  return 0;
}
