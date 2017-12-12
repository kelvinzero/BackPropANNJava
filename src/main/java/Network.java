import java.util.ArrayList;

public class Network {

    private ArrayList<Neuron[]> _layers;

    public Network(){
            _layers = new ArrayList<>();
    }

    public void append(Neuron[] layer){
        _layers.add(layer);
    }

    public ArrayList<Neuron[]> getLayers() {
        return _layers;
    }

    @Override
    public String toString(){

        StringBuilder sb = new StringBuilder();
        int layerNum = 0;

        for(Neuron[] layer : _layers){
            sb.append("Layer :[")
            .append(layerNum+1)
            .append("]\n");
            int neuronNum = 0;

            for(Neuron neuron : layer){
                sb.append("Neuron :[")
                .append(neuronNum+1)
                .append("]  ")
                .append(neuron.toString());
                if(neuronNum == layer.length-1)
                    sb.append("\n");
                neuronNum++;
            }
            layerNum++;
        }
        return sb.toString();
    }
}
