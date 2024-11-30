use std::env;

mod mnist_reader;
mod network;
mod train;


fn main() {
    let args: Vec<String> = env::args().collect();
    let data_path = &args[1];
    let epoch = args[2].parse::<u32>().unwrap();
    let mini_batch_size = args[3].parse::<u32>().unwrap();
    let eta = args[4].parse::<f32>().unwrap();
    
    train::train(data_path, epoch, mini_batch_size, eta);
}
