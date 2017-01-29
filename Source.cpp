/*

Backpropagtion Neural Network
CMPE 452 Assignment 2
Nolan Nisbet 10089874

*/

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include "BPN.h"
using namespace std;

#define PATTERN_COUNT_TRAIN 3823 //# of patterns in training set
#define PATTERN_COUNT_TEST 1797 //# of patterns in test set
#define PATTERN_SIZE 64
#define HIDDEN_LAYER_NEURONS 64 
#define NETWORK_OUTPUT 10
#define THRESHHOLD 114 //satisfactory MSE during training
#define EPOCHS 2000 //max iterations during training

int main()
{

	//Importing training values 64 [0..16] values followed by [0..9] value indicating letter represented
	vector<vector<float> > values;
	vector<float> valueline;
	ifstream trainin("training.txt");
	string item;
	for (string line; getline(trainin, line); )
	{
		istringstream in(line);

		while (getline(in, item, ','))
		{
			valueline.push_back(atof(item.c_str()));
		}

		values.push_back(valueline);
		valueline.clear();
	}
	trainin.close();


	vector<vector<float> > pattern; //vector used to hold training patterns
	vector<vector<float> > desiredout; //vector used to hold the desired output pattern of each input pattern

	for (int i = 0; i<PATTERN_COUNT_TRAIN; i++)
	{
		for (int j = 0; j<PATTERN_SIZE; j++)
		{
			valueline.push_back(values[i][j]);
		}
		pattern.push_back(valueline);
		valueline.clear();
	}

	/*To avoid cross talk each possible pattern value is represnted by a unique output from a particular neuron in the output layer*/
	for (int i = 0; i<PATTERN_COUNT_TRAIN; i++)
	{
		if (0 == values[i][PATTERN_SIZE]) {
			vector<float> v{ 0,0,0,0,0,0,0,0,0,1 };
			desiredout.push_back(v);
		}
		else if (1 == values[i][PATTERN_SIZE]) {
			vector<float> v{ 0,0,0,0,0,0,0,0,1,0 };
			desiredout.push_back(v);
		}
		else if (2 == values[i][PATTERN_SIZE]) {
			vector<float> v{ 0,0,0,0,0,0,0,1,0,0 };
			desiredout.push_back(v);
		}
		else if (3 == values[i][PATTERN_SIZE]) {
			vector<float> v{ 0,0,0,0,0,0,1,0,0,0 };
			desiredout.push_back(v);
		}
		else if (4 == values[i][PATTERN_SIZE]) {
			vector<float> v{ 0,0,0,0,0,1,0,0,0,0 };
			desiredout.push_back(v);
		}
		else if (5 == values[i][PATTERN_SIZE]) {
			vector<float> v{ 0,0,0,0,1,0,0,0,0,0 };
			desiredout.push_back(v);
		}
		else if (6 == values[i][PATTERN_SIZE]) {
			vector<float> v{ 0,0,0,1,0,0,0,0,0,0 };
			desiredout.push_back(v);
		}
		else if (7 == values[i][PATTERN_SIZE]) {
			vector<float> v{ 0,0,1,0,0,0,0,0,0,0 };
			desiredout.push_back(v);
		}
		else if (8 == values[i][PATTERN_SIZE]) {
			vector<float> v{ 0,1,0,0,0,0,0,0,0,0 };
			desiredout.push_back(v);
		}
		else {
			vector<float> v{ 1,0,0,0,0,0,0,0,0,0 };
			desiredout.push_back(v);
		}

	}

	BPN network; //create our neural network object


				 //Defining parameters of network object
	network.build(PATTERN_SIZE, HIDDEN_LAYER_NEURONS, NETWORK_OUTPUT);

	//arrays used to pass single patterns and its desired output
	float desiredoutS[10];
	float patternS[64];

	float error; //MSE
	int iterations = 0;
	for (int i = 0; i<EPOCHS; i++) //Training loop
	{
		iterations++;
		error = 0;
		for (int j = 0; j<PATTERN_COUNT_TRAIN; j++) //Iterates through each training pattern
		{
			for (int r = 0; r < PATTERN_SIZE; r++) {
				patternS[r] = pattern[j][r] / 16; //Normalizing input values from 0-16 to 0-1
			}
			for (int r = 0; r < NETWORK_OUTPUT; r++) {
				desiredoutS[r] = desiredout[j][r];
			}
			/* train(desuredoutS[10], pattern[64], learning rate, momentum) */
			error += network.train(0.03f, 0.1f, desiredoutS, patternS); //Pass single training pattern to network
		}

		if (error < THRESHHOLD) { //if MSE value is below an acceptabel threshold the training ends
			cout << endl << "ACCEPTABLE MSE VALUE ACHIEVED" << endl << endl;

			break;
		}

		//display error
		cout << "MSE:" << error << endl;

	}

	//display iterations
	cout << endl << "ITERATIONS:" << iterations << endl;

	/*********          TESTING PHASE          ************/
	cout << endl << "TESTING..." << endl << endl;

	//Importing testing values 64 [0..16] values followed by [0..9] value indicating letter represented
	values.clear();
	valueline.clear();
	ifstream testin("testing.txt");
	for (string line; getline(testin, line); )
	{
		istringstream in(line);

		while (getline(in, item, ','))
		{
			valueline.push_back(atof(item.c_str()));
		}

		values.push_back(valueline);
		valueline.clear();
	}
	testin.close();

	pattern.clear();

	for (int i = 0; i<PATTERN_COUNT_TEST; i++)
	{
		for (int j = 0; j<PATTERN_SIZE; j++)
		{
			valueline.push_back(values[i][j]);
		}
		pattern.push_back(valueline);
		valueline.clear();
	}

	vector<int> desiredoutNum; //holds the numerical value of the desired output

	for (int i = 0; i<PATTERN_COUNT_TEST; i++)    //This loops on the rows.
	{
		if (0 == values[i][PATTERN_SIZE]) {
			desiredoutNum.push_back(0);
		}
		else if (1 == values[i][PATTERN_SIZE]) {
			desiredoutNum.push_back(1);
		}
		else if (2 == values[i][PATTERN_SIZE]) {
			desiredoutNum.push_back(2);
		}
		else if (3 == values[i][PATTERN_SIZE]) {
			desiredoutNum.push_back(3);
		}
		else if (4 == values[i][PATTERN_SIZE]) {
			desiredoutNum.push_back(4);
		}
		else if (5 == values[i][PATTERN_SIZE]) {
			desiredoutNum.push_back(5);
		}
		else if (6 == values[i][PATTERN_SIZE]) {
			desiredoutNum.push_back(6);
		}
		else if (7 == values[i][PATTERN_SIZE]) {
			desiredoutNum.push_back(7);
		}
		else if (8 == values[i][PATTERN_SIZE]) {
			desiredoutNum.push_back(8);
		}
		else {
			desiredoutNum.push_back(9);
		}

	}

	int c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0, c8 = 0, c9 = 0; //holds how many of each number is correctly identified
	int t0 = 178, t1 = 182, t2 = 177, t3 = 183, t4 = 181, t5 = 182, t6 = 181, t7 = 179, t8 = 174, t9 = 180; //total of each number

	ofstream myfile;
	myfile.open("full_results.txt");

	for (int i = 0; i < PATTERN_COUNT_TEST; i++) { //iterate through entire test set

		for (int r = 0; r < PATTERN_SIZE; r++) { //ggrab individual patterns
			patternS[r] = pattern[i][r] / 16;
		}

		network.propagate(patternS); //propagte pattern through the trained network

									 /*Find the largest output value from the output layer*/
		int highest = 0;
		for (int j = 1; j < NETWORK_OUTPUT; j++) {
			float s = network.getOutput().neurons[j]->output;
			if (network.getOutput().neurons[j]->output > network.getOutput().neurons[highest]->output) {
				highest = j;
			}
		}

		
		myfile << "Test Pattern " << i << "  Desired Output: " << desiredoutNum[i] << " Actual Output: " << 9-highest << endl;
		myfile << "Output Values: ";
		for (int j = 0; j < NETWORK_OUTPUT; j++) {
			myfile << 9-j << ": " << network.getOutput().neurons[j]->output << "  ";
		}
		myfile << endl;
		
		

		highest = 9 - highest;

		if (desiredoutNum[i] == highest) { //check if netork output matches desired output numerically
			if (desiredoutNum[i] == 0) //add count to correct classification is network correctly classifies number
				c0++;
			else if (desiredoutNum[i] == 1)
				c1++;
			else if (desiredoutNum[i] == 2)
				c2++;
			else if (desiredoutNum[i] == 3)
				c3++;
			else if (desiredoutNum[i] == 4)
				c4++;
			else if (desiredoutNum[i] == 5)
				c5++;
			else if (desiredoutNum[i] == 6)
				c6++;
			else if (desiredoutNum[i] == 7)
				c7++;
			else if (desiredoutNum[i] == 8)
				c8++;
			else if (desiredoutNum[i] == 9)
				c9++;
		}



	}

	//output results of network on test data
	cout << "Correct Classifications for each Test Number" << endl;

	cout << "0 -> " << c0 << " / " << t0 << " = " << ((float)c0 / (float)t0) * 100 << " %" << endl;
	cout << "1 -> " << c1 << " / " << t1 << " = " << ((float)c1 / (float)t1) * 100 << " %" << endl;
	cout << "2 -> " << c2 << " / " << t2 << " = " << ((float)c2 / (float)t2) * 100 << " %" << endl;
	cout << "3 -> " << c3 << " / " << t3 << " = " << ((float)c3 / (float)t3) * 100 << " %" << endl;
	cout << "4 -> " << c4 << " / " << t4 << " = " << ((float)c4 / (float)t4) * 100 << " %" << endl;
	cout << "5 -> " << c5 << " / " << t5 << " = " << ((float)c5 / (float)t5) * 100 << " %" << endl;
	cout << "6 -> " << c6 << " / " << t6 << " = " << ((float)c6 / (float)t6) * 100 << " %" << endl;
	cout << "7 -> " << c7 << " / " << t7 << " = " << ((float)c7 / (float)t7) * 100 << " %" << endl;
	cout << "8 -> " << c8 << " / " << t8 << " = " << ((float)c8 / (float)t8) * 100 << " %" << endl;
	cout << "9 -> " << c9 << " / " << t9 << " = " << ((float)c9 / (float)t9) * 100 << " %" << endl;


	while (1);
	return 0;
}

