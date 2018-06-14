# countNet
tensorflow model for counting objects

the model can count upto 10 objects

1. gets two input images of similar items, one with a single item the other with multiple

2. passes each through mobilenet to extract the features then multiples the resulting activations and uses the result as an input for the countnet model.

3. countnet should output an array eg

	[1,0,0,0,0,0,0,0,0,0] - for one object,
	[0,1,0,0,0,0,0,0,0,0] - for two objects etc,
	
training is ok but loss does not change
