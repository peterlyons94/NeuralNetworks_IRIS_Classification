// Peter Lyons 2014 CMPE 452
// Student Number: 10042024 
// Net ID: 11pl18 
package com.company;
import java.util.Arrays;
import java.util.Scanner;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import static java.lang.Double.parseDouble;


public class Main {
    public static void main(String[] args) throws IOException {
        double learningRate = 0.5;
        double[][] Array = new double[3][332];
        double[] weights = {.5,.25,.85,-.23}; // initialize weights
        double[] weights2 = {.1,.8};
        System.out.println("Working Directory = " +
                System.getProperty("user.dir"));
        //Read the data from the text file 
        readData(Array);
        //call the test function which implements the computeNet, testANN & computeOutput 
        test(learningRate, weights, weights2, Array);
    }
    private static void test(double learningRate, double[] weights, double weights2[], double[][] Array) {
        computeNet(learningRate, weights, weights2, Array);
        //prints the final weight values
        System.out.println("\nThe Final Weights:" + Arrays.toString(weights));
    }
    /**
     * How this method works:
     * @param learningRate : a constant that is used when adjusting weights
     * @param weights : the weights for the hidden layer 2
     * @param weights2: the weights for the output layer
     * */
    public static void computeNet(double learningRate, double[] weights, double weights2[], double Array[][]) {
        // giving the square error an initial value
        double mse = 1;
        //while (mse > 0.1) {
            // cycles through the dataset
            for(int i =0; i < 150 ; i++){
                //setting up the hidden layer and calculating the output values of that hidden layer from the sigmoid function
                double hidden1output = computeOutput(Array[0][i] * weights[0] + Array[1][i] * weights[2]);
                double hidden2output = computeOutput(Array[0][i] * weights[1] + Array[1][i] * weights[3]);
                System.out.println(Array[0][i] * weights[0] + Array[1][i] * weights[2]);
                //compute the sigmoid function on the output of the hidden layer multiplied by the weights
                double sum = computeOutput(hidden1output * weights2[0] + hidden2output * weights2[1]);
                //updating the mse
                mse = .5 * (Array[2][i]-sum) * (Array[2][i]-sum);
                //finding delta for the output layer
                double delta = Array[2][i]-sum * sum * (1-sum);
                // adjust the weights on the output layers
                weights2 = hidden2(learningRate, weights2, delta, hidden1output, hidden2output);
                //adjust the weights for the hidden layer
                weights = hidden1(learningRate, hidden1output, hidden2output, weights, sum, Array[0][i], Array[1][i], Array[2][i]-sum);
            }
        //}
        System.out.println("");
        System.out.println("For weights: "+ weights[0] + "," + weights[1] + "," + weights[2] + "," + weights[3]);
        System.out.println("For weights2: "+ weights2[0] + "," + weights2[1]);
        // Now that the training is done we need to test the output to see if it works properly on both sets
        System.out.println("");
        System.out.println("");
        System.out.println("Now that the learning is complete, here are the test values:");
        System.out.println("");
        int correct = 0;
        int wrong = 0;
        for( int j = 0; j < 60; j++) {
            //computing the output from the hidden layer
            double hidden1output = Array[0][j] * weights[0] + Array[1][j] * weights[2];
            double hidden2output = Array[0][j] * weights[1] + Array[1][j] * weights[3];
            //compute the sigmoid function on the output of the hidden layer multiplied by the weights
            double sum = hidden1output * weights2[0] + hidden2output * weights2[1];
            // the counter to check if the classification was right
            mse = .5 * (Array[2][j]-sum) * (Array[2][j]-sum);
            double e = Math.abs(Array[2][j] - sum);
            if(mse < 0.4){
                correct++;
            }else wrong ++;
        }
        System.out.println("Right: " + correct + " Wrong: " + wrong);

    }
    // training function for the first hidden layer
    public static double[] hidden1(double c, double hidden1output, double hidden2output, double[] weights, double sum, double input1, double input2, double e) {
        // calculates the sum of delta * weights for each node
        double deltahj1 = (e * sum * (1-sum) * weights[0] + e * sum * (1-sum) * weights[1]) * hidden1output * (1 - hidden1output);
        double deltahj2 = (e * sum * (1-sum) * weights[2] + e * sum * (1-sum) * weights[3]) * hidden2output * (1 - hidden2output);
        // performs the weight updates for layer w10ij
        weights[0] += c * input1 * deltahj1 * hidden1output * (1 - hidden1output);
        weights[1] += c * input1 * deltahj1 * hidden1output * (1 - hidden1output);
        weights[2] += c * input2 * deltahj2 * hidden2output * (1 - hidden2output);
        weights[3] += c * input2 * deltahj2 * hidden2output * (1 - hidden2output);
        //System.out.println(hidden1output + " " + hidden2output + " " + deltaOj1 + " " + deltaOj2);
        return weights;
    }
    // training function for the output layer
    public static double[] hidden2(double c, double[] weights2, double delta, double hidden1output, double hidden2output) {
        //updates the weights, calculates the node input * learning rate * deltaoj
        weights2[0] += c  * hidden1output * delta;
        weights2[1] += c  * hidden2output * delta;
        //System.out.println(Arrays.toString(weights2));
        return weights2;

    }
    /**
     * This method computes the output of the sigmoid function
     * @param sig_sum: the sum of the values before  applying the sigmoid function
     */
    private static double computeOutput(double sig_sum) {
        //System.out.print(" The Sum: "+ sum + "\t\t");
        return  1 / (1 + Math.exp(-1 * sig_sum));
    }
    /**
     * This method reads the text file, which is comma separated and sorts it into a 2d array list by appending
     * adding each row value into a type double[] array and then appending that to List<double[]> array.
     *
     * @throws IOException  : If the user inputs the wrong filename it will throw an exception
     */
    public static void readData( double[][] Array) throws IOException {
        //This accepts the Filename that the user wants to use. 
        Scanner user_input = new Scanner(System.in);
        System.out.println("Enter the Name of the File you wish to use:");
        String filename;
        filename = user_input.next();
        // this function reads the text file and organizes the values into three arrays
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            //three counters for each of the three arrays
            int i = 0;
            //loops through every line in the text file
            while ((line = br.readLine()) != null) {
                String[] ar = line.split(" ");
                // simple if else to determine which array the row of data should be added to
                Array[0][i] = parseDouble(ar[0]);
                Array[1][i] = parseDouble(ar[1]);
                Array[2][i] = parseDouble(ar[2]);
                i++;
            }
            br.close();
        }
    }
}