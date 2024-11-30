
use crate::mnist_reader;
use crate::network;

pub fn train(epochs: u32, mini_batch_size: u32, eta: f32) {
    let mut mnist_network = network::Network::new(vec![28*28, 30, 10]);

    let training_data = mnist_reader::get_traning_data();
    let test_data = mnist_reader::get_test_data();

    mnist_network.sgd(&training_data, epochs, mini_batch_size, eta, &test_data);
}