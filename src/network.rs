use rand::seq::SliceRandom;
use rand::thread_rng;

use ndarray::prelude::*;

use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

pub struct Network {
    pub sizes: Vec<u32>,
    pub num_layers: u32,
    pub biases: Vec<Array1<f32>>,
    pub weights: Vec<Array2<f32>>
}

impl Network {
    pub fn new(sizes: Vec<u32>) -> Self {
        let num_layers = sizes.len() as u32;

        let mut biases: Vec<Array1<f32>> = Vec::new();
        let mut weights: Vec<Array2<f32>> = Vec::new();

        

        for i in 1..sizes.len() {
            let random_bias = Array::random(sizes[i] as usize, StandardNormal);
            biases.push(random_bias);
        }

        for i in 0..sizes.len()-1 {
            let random_weight = Array::random((sizes[i+1] as usize, sizes[i] as usize), StandardNormal);
            weights.push(random_weight);
        }

        Self {
            sizes: sizes,
            num_layers: num_layers,
            biases: biases,
            weights: weights
        }
    }   

    pub fn feed_forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let mut result = input.clone();
        for i in 0..self.num_layers-1 {
            result = (&self.weights[i as usize]).dot(&result) + &self.biases[i as usize];
            result = sigmoid(&result);
        }
        return result;
    }

    pub fn sgd(&mut self, training_data: &Vec<(Array1<f32>, Array1<f32>)>, epochs: u32, mini_batch_size: u32, eta: f32, test_data: &(Vec<Array1<f32>>, Vec<u8>)) {
        let n_test = test_data.0.len();
        // let n = training_data.len();

        let mut training_data = training_data.clone();
        // let test_data = test_data.clone();

        for i in 0..epochs {
            let mut rng = thread_rng();
            training_data.shuffle(&mut rng);

            for j in (0..training_data.len()-(mini_batch_size as usize)).step_by(mini_batch_size as usize) {
                // let batch_train = training_data[j..j+(mini_batch_size as usize)];
                let mut batch_train = Vec::new();
                for k in 0..mini_batch_size{
                    batch_train.push(training_data[(j as usize) + (k as usize)].clone());
                }
                self.update_mini_batch(&batch_train, eta);
            }

            println!("Epoch {}, {} / {}", i, self.evaluate(&test_data), n_test);
            
        }
    }

    pub fn update_mini_batch(&mut self, mini_batch: &Vec<(Array1<f32>, Array1<f32>)>, eta: f32) {
        let mut nabla_b: Vec<Array1<f32>> = Vec::new();
        for i in 0..self.biases.len() {
            let nabla_b_v = Array1::zeros((&self.biases[i]).dim());
            nabla_b.push(nabla_b_v);
        }

        let mut nabla_w: Vec<Array2<f32>> = Vec::new();
        for i in 0..self.weights.len() {
            let nabla_w_v = Array2::zeros((&self.weights[i]).dim());
            nabla_w.push(nabla_w_v);
        }

        for i in mini_batch {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(&i.0, &i.1);
            for j in 0..self.num_layers-1 {
                nabla_b[j as usize] = &nabla_b[j as usize] + &delta_nabla_b[j as usize];
                // println!("{}x{}", nabla_w[j as usize].shape()[0], nabla_w[j as usize].shape()[1]);
                // println!("{}x{}", delta_nabla_w[j as usize].shape()[0], delta_nabla_w[j as usize].shape()[1]);
                nabla_w[j as usize] = &nabla_w[j as usize] + &delta_nabla_w[j as usize];

            }
        }

        for i in 0..self.num_layers-1 {
            self.weights[i as usize] = &self.weights[i as usize] - (eta/(mini_batch.len() as f32)) * &nabla_w[i as usize];
            self.biases[i as usize] = &self.biases[i as usize] - (eta/(mini_batch.len() as f32)) * &nabla_b[i as usize];
        }

    }

    // I have no idea if I implemented this correctly
    // Update from future: Apparently I did
    pub fn backprop(&mut self, x: &Array1<f32>, y: &Array1<f32>) -> (Vec<Array1<f32>>, Vec<Array2<f32>>) {
        let mut nabla_b = Vec::new();
        for i in 0..self.biases.len() {
            let nabla_b_v = Array1::zeros((&self.biases[i]).dim());
            nabla_b.push(nabla_b_v);
        }

        let mut nabla_w = Vec::new();
        for i in 0..self.weights.len() {
            let nabla_w_v = Array2::zeros((&self.weights[i]).dim());
            nabla_w.push(nabla_w_v);
        }

        let mut current_activation = x.clone();
        let mut activations = Vec::new();
        activations.push(current_activation.clone());    
        let mut zs = Vec::new();

        for i in 0..self.num_layers-1 {
            let z = (&self.weights[i as usize]).dot(&current_activation) + &self.biases[i as usize];
            zs.push(z);
            current_activation = sigmoid(&zs[i as usize]);
            activations.push(current_activation.clone());            
        }

        let mut delta = self.cost_derivative(&activations[activations.len() - 1], y) * sigmoid_prime(&zs[zs.len() - 1]);
        let mut b_index = nabla_b.len() - 1;
        nabla_b[b_index] = delta.clone();
        let mut w_index = nabla_w.len() - 1;
        nabla_w[w_index] = get_vector_mul(&delta, &activations[activations.len() - 2]);


        for layer in 2..self.num_layers {
            let sp = sigmoid_prime(&zs[zs.len() - (layer) as usize]);
            
            let weight_transpose = (&self.weights[ self.weights.len() + 1 - layer as usize]).clone();
            let weight_transpose = weight_transpose.reversed_axes();
            delta = weight_transpose.dot(&delta) * &sp;

            b_index = nabla_b.len() - layer as usize;
            nabla_b[b_index] = delta.clone();
            w_index = nabla_w.len() - layer as usize;

            nabla_w[w_index] = get_vector_mul(&delta, &activations[activations.len() - 1 - layer as usize]);

        }

        return (nabla_b, nabla_w);
    }

    pub fn evaluate(&mut self, test_data: &(Vec<Array1<f32>>, Vec<u8>)) -> u32 {
        let mut correct_count: u32 = 0;
        let test_images = &test_data.0;
        let test_labels = &test_data.1;
        for i in 0..test_images.len() {
            let correct_answer = test_labels[i];

            let output = self.feed_forward(&test_images[i]);
            let mut max_index = 0;
            for j in 0..output.len() {
                if output[max_index] < output[j] {
                    max_index = j;
                }
            }

            if (max_index as u8) == correct_answer {
                correct_count += 1;
            }
        }

        return correct_count;
    }



    pub fn cost_derivative(&mut self, output_activations: &Array1<f32>, desired_output: &Array1<f32>) -> Array1<f32> {
        let mut result = output_activations.clone();
        result = &result - desired_output;
        return result;
    }
 
}


fn get_vector_mul(left_array: &Array1<f32>, right_array: &Array1<f32>) -> Array2<f32> {

    let left_array_c = left_array.clone();
    let right_array_c = right_array.clone();

    let left_array_2d = left_array_c.into_shape_with_order((left_array.shape()[0], 1))
        .expect("Could not convert array to this order");

    let right_array_2d = right_array_c.into_shape_with_order((1, right_array.shape()[0]))
        .expect("Could not convert array to this order");

    let matrix = left_array_2d.dot(&right_array_2d);
    return matrix;
}

pub fn sigmoid(array: &Array1<f32>) -> Array1<f32> {
    let mut result_array: Array1<f32>  = array.clone();
    result_array = 1.0f32 / (1.0f32 + (-&result_array).exp());
    return result_array;
}

pub fn sigmoid_prime(array: &Array1<f32>) -> Array1<f32> {
    let result_array: Array1<f32>  = sigmoid(array) * (1.0f32 - sigmoid(array));
    return result_array;
}
