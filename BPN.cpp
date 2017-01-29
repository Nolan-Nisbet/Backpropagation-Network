/*
Backpropagtion Neural Network
Nolan Nisbet 
*/


#include "BPN.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
using namespace std;


//Initializates neuron weights
void neuron::build(int inputcount)
{
	float random;//random number
	weights = new float[inputcount];
	deltavalues = new float[inputcount];
	//important initializate all weights as random unsigned values
	//and delta values as 0
	for (int i = 0; i<inputcount; i++)
	{
		//random number -1.0 to 1.0
		random = (float(rand()) / float(RAND_MAX)) * 2.f - 1;
		weights[i] = random;
		deltavalues[i] = 0;
	}
	float sign = -1;//change sign
					//random number -1.0 to 1.0
	random = (float(rand()) / float(RAND_MAX)) * 2.f - 1;
	random *= sign;
	sign *= -1;
	wgain = random;
	gain = 1; // standard gain value
}

//Constructors
neuron::neuron() :weights(0), deltavalues(0), output(0), gain(0), wgain(0)
{

}

layer::layer() : neurons(0), neuroncount(0), layerinput(0), inputcount(0)
{

}

//Initializates layer
void layer::build(int inputsize, int _neuroncount)
{
	neurons = new neuron*[_neuroncount];
	for (int i = 0; i<_neuroncount; i++)
	{
		neurons[i] = new neuron;
		neurons[i]->build(inputsize);
	}

	layerinput = new float[inputsize];
	neuroncount = _neuroncount;
	inputcount = inputsize;
}
//Calculates the neural network result of the layer using the sigmoid function
void layer::calculate()
{
	int i, j;
	float sum;
	//Apply the formula for each neuron
	for (i = 0; i<neuroncount; i++)
	{
		sum = 0;//store the sum of all values here
		for (j = 0; j<inputcount; j++)
		{
			//Performing function
			sum += neurons[i]->weights[j] * layerinput[j]; //apply input * weight
		}
		sum += neurons[i]->wgain * neurons[i]->gain; //apply the gain or theta multiplied by the gain weight.
													 //sigmoidal activation function
		neurons[i]->output = 1.f / (1.f + exp(-sum));//calculate the sigmoid function
	}
}




BPN::BPN()
{}

//allocates memory and intializes layers in the network
void BPN::build(int inputcount, int inputneurons, int outputcount)
{
	hiddenlayer.build(inputcount, inputneurons);
	outputlayer.build(inputneurons, outputcount);

}

//Propgate a pattern through the network
void BPN::propagate(const float *input)
{
	//copy inmput pattern to hidden layers input
	memcpy(hiddenlayer.layerinput, input, hiddenlayer.inputcount * sizeof(float));
	//calulcate outputs of hidden layer
	hiddenlayer.calculate();

	//propagting the output from the hidden layer to the inputs of the output layer
	for (int i = 0; i < hiddenlayer.neuroncount; i++)
	{
		outputlayer.layerinput[i] = hiddenlayer.neurons[i]->output;
	}

	//calculating the outputs of the outptu layer
	outputlayer.calculate();
}



//Backprogation Training Function
float BPN::train(float lrate, float momentum, const float *desiredoutput, const float *input)
{
	float errorg = 0; //general quadratic error (divide by 2 to get MSE)
	float errorc; //local error;
	float sum = 0, csum = 0;
	float delta, newdelta;
	float output;

	propagate(input); //propgate input pattern through network

	int i, j, k;

	//BACKPROPAGATION ALGORITHM

	//start from the output layer
	for (i = 0; i<outputlayer.neuroncount; i++)
	{

		//calculate the error value for the output layer
		output = outputlayer.neurons[i]->output;
		errorc = (desiredoutput[i] - output) * output * (1 - output);
		//and the general error as the sum of delta values. Where delta is the squared difference
		//of the desired value with the output value
		errorg += (desiredoutput[i] - output) * (desiredoutput[i] - output); //quadratic error
																			 //update the weights of the neuron
		for (j = 0; j<outputlayer.inputcount; j++) //iterate through all inputs to each neuron in the oputput layer
		{
			delta = outputlayer.neurons[i]->deltavalues[j];//current delta value
			newdelta = lrate * errorc * outputlayer.layerinput[j] + delta * momentum;//update delta value
			outputlayer.neurons[i]->weights[j] += newdelta; //update the weight values
			outputlayer.neurons[i]->deltavalues[j] = newdelta;
			//propogate to hidden layer
			sum += outputlayer.neurons[i]->weights[j] * errorc;
		}
		outputlayer.neurons[i]->wgain += lrate * errorc * outputlayer.neurons[i]->gain;//calculate weight gain

	}


	//now process the hidden layer
	for (i = 0; i<hiddenlayer.neuroncount; i++)
	{
		output = hiddenlayer.neurons[i]->output;
		errorc = output * (1 - output) * sum;
		for (j = 0; j<hiddenlayer.inputcount; j++)
		{
			delta = hiddenlayer.neurons[i]->deltavalues[j];
			newdelta = lrate * errorc * hiddenlayer.layerinput[j] + delta * momentum;
			hiddenlayer.neurons[i]->weights[j] += newdelta;//update weights
			hiddenlayer.neurons[i]->deltavalues[j] = newdelta;
		}
		hiddenlayer.neurons[i]->wgain += lrate * errorc * hiddenlayer.neurons[i]->gain;//update the gain weight
	}

	//return MSE
	return errorg / 2;
}



