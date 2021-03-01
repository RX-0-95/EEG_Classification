## Shallow ConvNet 

* input data : (B,1,22,1000)

### Layers 

1. Conv1d (40,(1,25)): Process time interval 
    * input: (B,1,22,1000)
    * output: (B,40,22,976)

2. Dense: Time inverval 
    * input: (B,40,22,976)
    * reshape (B,976,22,40)
    * flatten (-1,976,880)

3. full connect (880,40): contribute of one single time interval 
    * input (-1,976,800)
    * output(-1,976,40)


4. Square: square and focus on process timer inverval 
    * input (-1,976,40)
    * square 
    * reform (-1,40,976)
    * output(-1,40,976)


5. Average pool: (75, stride = 15)
    * input: (-1,40,976)
    * average pool
    * output: (-1, 40, (976-75)/15 +1= 61)

6. log 
    * input: (-1,40,61)
    * log
    * reshape (-1,40*61) 
    * out: (-1,40*61)
    
7. full connect: (2240,4)
    * in: (-1,40,61)
    * out: (-1,4)

8. softmax:
    * in: (-1,4)
    * out: (-1,1)

## ResConvNet 
