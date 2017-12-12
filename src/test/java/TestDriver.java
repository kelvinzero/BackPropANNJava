import org.junit.Test;

import java.util.List;

public class TestDriver {

    @Test
    public void trainTest() throws Exception{

        List<double[]> dataSet = DatasetLoader.loadSet("./seeds_dataset.txt");
        int numInputs = dataSet.get(0).length-1;
        int numHiddens = 3;
        int numOutputs = 4;
        double error = 0.0d;
        NeuralNet neuralNet = new NeuralNet(numInputs, numHiddens, numOutputs);


        DatasetLoader.normalizeMinMax(0, 1, 7, dataSet);
        neuralNet.trainNetwork(dataSet, 0.03, 460);

        for(double[] row : dataSet){
            double prediction = neuralNet.predict(row);
            if(prediction - row[row.length-1] != 0)
                error += 1;
        }
        error /= dataSet.size();
        System.out.printf("Error: %f%%\n", error*100);
    }
}
