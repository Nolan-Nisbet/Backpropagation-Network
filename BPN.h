/*
Backpropagtion Neural Network
Nolan Nisbet 
*/


#pragma once

//Neuron data structure
struct neuron
{
	neuron();
	void build(int inputcount);//Allocates memory and initializates values

	float *weights; // neuron input weights or synaptic connections
	float *deltavalues; //neuron delta values
	float output; //output value
	float gain;//Gain value
	float wgain;//Weight gain value
};

//Layer data structure
struct layer
{
	layer();//Object constructor. Initializates all values as 0
	void build(int inputsize, int neuroncount);//Creates the layer and allocates memory
	void calculate();//Calculates all neurons performing the network formula
	neuron **neurons;//array of all neurons in layer
	int neuroncount;//# of neurons in layer
	float *layerinput;//the input to the layer
	int inputcount;//numbe rof inputs to each neuron in the layer
};

//Network class 
class BPN
{
private:
	layer hiddenlayer;//networks hidden layer
	layer outputlayer;//networks output layer



public:
	//function tu create in memory the network structure
	BPN();//Construction..initialzates all values to 0
		  //Creates the network structure on memory
	void build(int inputcount, int inputneurons, int outputcount);

	void propagate(const float *input);//Calculates the network values given an input pattern
									   //Updates the weight values of the network given a desired output and applying the backpropagation
									   //Algorithm

	float train(float lrate, float momentum, const float *desiredoutput, const float *input);

	//returns output layer structure in order to read results of pattern propagation through the network
	inline layer &getOutput()
	{
		return outputlayer;
	}

};



