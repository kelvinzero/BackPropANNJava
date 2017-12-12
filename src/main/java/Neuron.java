import java.util.Random;

public class Neuron {

    private double      _output;
    private double      _transferDerivative;
    private double      _deltaError;
    private double[]    _weights;
    private Random rand;

    public Neuron(int numInputs){
        rand = new Random(System.nanoTime());
        _weights = new double[numInputs];
        for(int idx = 0; idx < _weights.length; idx++)
            _weights[idx] = rand.nextDouble();
    }

    /**
     * Calculate the neuron output and transfer derivative
     * @param inputs - Layer above outputs
     * @return - Neuron output
     */
    public Double activate(double[] inputs){

        double activation = _weights[_weights.length-1]; // add bias

        for(int i = 0; i < _weights.length-1; i++)
            activation += _weights[i] * inputs[i];

        _output = 1.0 / (1.0 + Math.exp(-activation));
        _transferDerivative = _output * (1 - _output);

        return _output;
    }

    public void setDeltaError(double deltaError){
        _deltaError = deltaError;
    }

    public double getDeltaError(){
        return _deltaError;
    }

    public double getTransferDerivative(){
        return _transferDerivative;
    }

    public double getOutput(){
        return _output;
    }

    public double[] getWeights(){
        return _weights;
    }

    @Override
    public String toString(){

       StringBuilder builder = new StringBuilder();

       builder.append("Delta error: [");
        builder.append(_deltaError);
        builder.append("] Output: [");
        builder.append(_output);
        builder.append("]");
        builder.append(" 'Weights' {");

        for(int idx = 0; idx < _weights.length; idx++){
            builder.append(_weights[idx]);
            if(idx < _weights.length-1)
                builder.append(", ");
        }
        builder.append("}");
        return builder.toString();
    }

}
