use rand::{Rng, RngCore};

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

#[derive(Debug)]
pub struct LayerTopology {
    pub neurons: usize,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {
        // Network with just one layer is technically doable, but doesn't
        // make much sense:
        assert!(layers.len() > 1);

        let layers = layers
            .windows(2)
            .map(|layers| Layer::random(rng, layers[0].neurons, layers[1].neurons))
            .collect();

        Self { layers }
    }

    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }
}

#[derive(Debug, Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(neurons: Vec<Neuron>) -> Self {
        assert!(!neurons.is_empty());

        assert!(neurons
            .iter()
            .all(|neuron| neuron.weights.len() == neurons[0].weights.len()));

        Self { neurons }
    }

    fn random(rng: &mut dyn RngCore, input_size: usize, output_size: usize) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::random(rng, input_size))
            .collect();

        Self { neurons }
    }

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Neuron {
    pub fn new(bias: f32, weights: Vec<f32>) -> Self {
        assert!(!weights.is_empty());

        Self { bias, weights }
    }

    fn random(rng: &mut dyn RngCore, input_size: usize) -> Self {
        let bias = rng.gen_range(-1.0..=1.0);
        let weights = (0..input_size).map(|_| rng.gen_range(-1.0..=1.0)).collect();

        Self { bias, weights }
    }

    fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        (self.bias + output).max(0.0)
    }
}

#[cfg(test)]
mod network_tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn random() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let network = Network::random(
            &mut rng,
            &[
                LayerTopology { neurons: 3 },
                LayerTopology { neurons: 2 },
                LayerTopology { neurons: 1 },
            ],
        );

        assert_eq!(network.layers.len(), 2);
        assert_eq!(network.layers[0].neurons.len(), 2);

        assert_relative_eq!(network.layers[0].neurons[0].bias, -0.6255188);

        assert_relative_eq!(
            network.layers[0].neurons[0].weights.as_slice(),
            &[0.67383957, 0.8181262, 0.26284897].as_slice()
        );

        assert_relative_eq!(network.layers[0].neurons[1].bias, 0.5238807);

        assert_relative_eq!(
            network.layers[0].neurons[1].weights.as_slice(),
            &[-0.5351684, 0.069369555, -0.7648182].as_slice()
        );

        assert_eq!(network.layers[1].neurons.len(), 1);

        assert_relative_eq!(
            network.layers[1].neurons[0].weights.as_slice(),
            &[-0.48879623, -0.19277143].as_slice()
        );
    }

    #[test]
    fn propagate() {
        let layers = (
            Layer::new(vec![
                Neuron::new(0.0, vec![-0.5, -0.4, -0.3]),
                Neuron::new(0.0, vec![-0.2, -0.1, 0.0]),
            ]),
            Layer::new(vec![Neuron::new(0.0, vec![-0.5, 0.5])]),
        );
        let network = Network::new(vec![layers.0.clone(), layers.1.clone()]);

        let actual = network.propagate(vec![0.5, 0.6, 0.7]);
        let expected = layers.1.propagate(layers.0.propagate(vec![0.5, 0.6, 0.7]));

        assert_relative_eq!(actual.as_slice(), expected.as_slice());
    }
}

#[cfg(test)]
mod layer_tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn random() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let layer = Layer::random(&mut rng, 3, 2);

        let actual_biases: Vec<_> = layer.neurons.iter().map(|neuron| neuron.bias).collect();
        let expected_biases = vec![-0.6255188, 0.5238807];

        let actual_weights: Vec<_> = layer
            .neurons
            .iter()
            .map(|neuron| neuron.weights.as_slice())
            .collect();

        let expected_weights: Vec<&[f32]> = vec![
            &[0.67383957, 0.8181262, 0.26284897],
            &[-0.53516835, 0.069369674, -0.7648182],
        ];

        assert_relative_eq!(actual_biases.as_slice(), expected_biases.as_slice());
        assert_relative_eq!(actual_weights.as_slice(), expected_weights.as_slice());
    }

    #[test]
    fn propagate() {
        let neurons = (
            Neuron::new(0.0, vec![0.1, 0.2, 0.3]),
            Neuron::new(0.0, vec![0.4, 0.5, 0.6]),
        );

        let layer = Layer::new(vec![neurons.0.clone(), neurons.1.clone()]);
        let inputs = &[-0.5, 0.0, 0.5];

        let actual = layer.propagate(inputs.to_vec());
        let expected = vec![neurons.0.propagate(inputs), neurons.1.propagate(inputs)];

        assert_relative_eq!(actual.as_slice(), expected.as_slice());
    }
}

#[cfg(test)]
mod neuron_tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn random() {
        // Because we always use the same seed, our `rng` in here will
        // always return the same set of values
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let neuron = Neuron::random(&mut rng, 4);

        assert_relative_eq!(neuron.bias, -0.6255188);
        assert_relative_eq!(
            neuron.weights.as_slice(),
            &[0.67383957, 0.8181262, 0.26284897, 0.5238807].as_ref()
        );
    }

    #[test]
    fn propagate() {
        let neuron = Neuron {
            bias: 0.5,
            weights: vec![-0.3, 0.8],
        };

        // Ensures `.max()` (our ReLU) works:
        assert_relative_eq!(neuron.propagate(&[-10.0, -10.0]), 0.0,);

        // `0.5` and `1.0` chosen by a fair dice roll:
        assert_relative_eq!(
            neuron.propagate(&[0.5, 1.0]),
            (-0.3 * 0.5) + (0.8 * 1.0) + 0.5,
        );

        // We could've written `1.15` right away, but showing the entire
        // formula makes our intentions clearer
    }
}
