
mod mnist_reader;
mod network;
mod train;


fn main() {
    train::train(100, 10, 3.0f32);
}
