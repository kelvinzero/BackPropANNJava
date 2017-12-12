import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NeuralNet {

    private int     _numInputs;
    private int     _numHiddens;
    private int     _numOutputs;
    private Network _network;

    public NeuralNet(int numInputs, int numHiddens, int numOutputs){
        _numInputs = numInputs;
        _numHiddens = numHiddens;
        _numOutputs = numOutputs;
        _network = new Network();
        initializeNetwork();
     }

    /**
     * Instantiate/create layers.
     */
    private void initializeNetwork(){

        Neuron[] hiddenLayer = new Neuron[_numHiddens];
        Neuron[] outputLayer = new Neuron[_numOutputs];

        for(int hidden = 0; hidden < _numHiddens; hidden++)
            hiddenLayer[hidden] = new Neuron(_numInputs + 1);
        for(int outputs = 0; outputs < _numOutputs; outputs++)
            outputLayer[outputs] = new Neuron(_numHiddens + 1);

        _network.append(hiddenLayer);
        _network.append(outputLayer);
    }

    /**
     * Propagates the last neuron activation to the new neuron as input.
     * @param inputs - Training inputs from the data set
     * @return - The outputs from neurons in the output layer
     */
    public double[] forwardPropagate(double [] inputs){

        double[] layerInputs = new double[inputs.length];
        double[] layerOutputs;
        System.arraycopy(inputs, 0, layerInputs, 0, inputs.length);

        for(Neuron[] layer : _network.getLayers()){

            layerOutputs = new double[layer.length];
            int n = 0;

            for(Neuron neuron : layer) // for each neuron in the layer
                layerOutputs[n++] = neuron.activate(layerInputs);

            layerInputs = new double[layerOutputs.length];
            System.arraycopy(layerOutputs, 0, layerInputs, 0, layerOutputs.length);
        }
        return layerInputs;
    }

    /**
     * Calculate error delta in each neuron using expected outputs and previous layer error delta values.
     * @param expected - Array[double] of expected outputs
     */
    public void backPropagate(double[] expected){

        Neuron[] thisLayer;
        Neuron thisNeuron;
        double error;
        ArrayList<Neuron[]> networkLayers = _network.getLayers();

        for(int layerIdx = networkLayers.size()-1; layerIdx >= 0; layerIdx--){  // each layer in reverse

            thisLayer = networkLayers.get(layerIdx);
            for(int neuronIdx = 0; neuronIdx < thisLayer.length; neuronIdx++){ // each neuron in the layer

                thisNeuron = networkLayers.get(layerIdx)[neuronIdx]; // this neuron in the layer
                error = 0.0d;

                if(layerIdx == networkLayers.size()-1) // if this is output layer
                    thisNeuron.setDeltaError(
                            (expected[neuronIdx] - thisNeuron.getOutput()) * thisNeuron.getTransferDerivative()
                    );

                else{ // else if this is a hidden layer or entry layer
                    for(Neuron plusLevelNeuron : networkLayers.get(layerIdx+1))
                        error += (plusLevelNeuron.getWeights()[neuronIdx] * plusLevelNeuron.getDeltaError());
                    thisNeuron.setDeltaError(error * thisNeuron.getTransferDerivative());
                }
            }
        }
    }


    /**
     * Update weights in the layers using previously calculated delta errors during back propagation.
     * @param record - The training record
     * @param learnRate - The learning rate
     */
    public void updateWeights(double[] record, double learnRate) {

        double[] layerInputs = new double[_numInputs];
        int numLayers = _network.getLayers().size();

        ArrayList<Neuron[]> networkLayers = _network.getLayers();
        Neuron thisNeuron;
        Neuron[] thisLayer;
        double[] thisNeuronWeights;

        System.arraycopy(record, 0, layerInputs, 0, _numInputs);

        for (int layerIdx = 0; layerIdx < numLayers; layerIdx++) { // for each layer

            thisLayer = networkLayers.get(layerIdx);

            for (int neuronIdx = 0; neuronIdx < thisLayer.length; neuronIdx++) { // for each neuron in the layer

                thisNeuron = networkLayers.get(layerIdx)[neuronIdx];
                thisNeuronWeights = thisNeuron.getWeights();

                // adjust the layer neuron weights using prior level inputs' deltaError
                for (int inputIdx = 0; inputIdx < layerInputs.length; inputIdx++)
                    thisNeuronWeights[inputIdx] += learnRate * thisNeuron.getDeltaError() * layerInputs[inputIdx];

                thisNeuronWeights[thisNeuron.getWeights().length - 1] += learnRate * thisNeuron.getDeltaError();
            }
            layerInputs = new double[thisLayer.length]; //

            for (int neuronIdx = 0; neuronIdx < numLayers; neuronIdx++)
                layerInputs[neuronIdx] = thisLayer[neuronIdx].getOutput();
        }
    }

    /**
     * Train the network using training set
     * @param trainingSet - The training set with known class values
     * @param learningRate - Learning rate
     * @param numEpochs - Number of epochs (training cycles)
     */
    public void trainNetwork(List<double[]> trainingSet, double learningRate, int numEpochs) {

        Random rand = new Random(System.currentTimeMillis());

        for (int epoch = 0; epoch < numEpochs; epoch++) {
            double error = 0.0d;
            int expectedVal;

            for (double[] record : trainingSet) {

                double[] outputs = forwardPropagate(record);
                double[] expected = new double[_numOutputs];

                expectedVal = (int) record[record.length - 1]; // the last record value is the expected value
                expected[expectedVal] = 1;

                for (int idx = 0; idx < expected.length; idx++)
                    error += Math.pow(expected[idx] - outputs[idx], 2);

                backPropagate(expected);
                updateWeights(record, learningRate);
            }
        }
    }

    /**
     * Forward propagate and return the prediction
     * @param record - The record to predict from
     * @return - The prediction
     */
    public int predict(double[] record){

        double[] output = forwardPropagate(record);
        int maxIdx = 0;
        double maxVal = output[0];
        for(int i = 1; i < output.length; i++){
            if(output[i] > maxVal){
                maxIdx = i;
                maxVal = output[i];
            }
        }
        return maxIdx;
    }
}
