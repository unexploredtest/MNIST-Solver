

use std::fs::File;
use std::io::Read;

use ndarray::prelude::*;


pub fn read_labels(data: Vec<u8>) -> Vec<u8> {
    let mut current_byte = 0;
    current_byte += 2; // The first two bytes are useless so we pass

    let _data_info = data[current_byte];
    current_byte += 1;

    let dim_number = data[current_byte];
    current_byte += 1;

    let mut data_size: u32 = 1;
    let mut dimensions: Vec<u32>= Vec::new();
    // let mut dimensions: (u32, u32, u32) = (1, 1, 1);
    for _i in 0..dim_number {
        let mut dimension_size: u32 = 0;
        for j in 0..4 {
            dimension_size += data[current_byte] as u32;
            if j != 3 {
                dimension_size = dimension_size << 8;
            }
            current_byte += 1;
        }
        // dimensions.push(dimension_size);
        dimensions.push(dimension_size);
        data_size *= dimension_size;
    }

    // println!("{}", data_size);

    let mut data_vec: Vec<u8>= Vec::new();
    for _i in 0..data_size {
        data_vec.push(data[current_byte]);
        current_byte += 1;
    }

    return data_vec;
}

pub fn read_images(data: Vec<u8>) -> Vec<Array1<f32>> {
    let mut current_byte = 0;
    current_byte += 2; // The first two bytes are useless so we pass

    let _data_info = data[current_byte];
    current_byte += 1;

    let dim_number = data[current_byte];
    current_byte += 1;

    // let mut data_size: u32 = 1;
    let mut dimensions: Vec<u32>= Vec::new();
    // let mut dimensions: (u32, u32, u32) = (1, 1, 1);
    for _i in 0..dim_number {
        let mut dimension_size: u32 = 0;
        for j in 0..4 {
            dimension_size += data[current_byte] as u32;
            if j != 3 {
                dimension_size = dimension_size << 8;
            }
            current_byte += 1;
        }
        // dimensions.push(dimension_size);
        dimensions.push(dimension_size);
        // data_size *= dimension_size;
    }

    let mut images_dat: Vec<Array1<f32>> = Vec::new();

    let each_image_size = dimensions[1] * dimensions[2];
    for _i in 0..dimensions[0] {
        let mut new_image: Vec<f32> = Vec::new();
        for _j in 0..each_image_size {
            let pixel: f32 = (data[current_byte] as f32) / 255f32;
            current_byte += 1;
            new_image.push(pixel);            
        }

        let new_image_array = Array1::from_shape_vec(each_image_size as usize, new_image)
            .expect("Should've been read");

        images_dat.push(new_image_array);

    }

    return images_dat;   
}

pub fn read_labels_file(file_path: &str) -> Vec<u8> {
    let mut file: File = File::open(file_path).expect("Failed to open file");
    let mut contents: Vec<u8> = Vec::new();
    file.read_to_end(&mut contents).expect("Failed to read file");
    return read_labels(contents);
}

pub fn read_images_file(file_path: &str) -> Vec<Array1<f32>> {
    let mut file: File = File::open(file_path).expect("Failed to open file");
    let mut contents: Vec<u8> = Vec::new();
    file.read_to_end(&mut contents).expect("Failed to read file");
    return read_images(contents);
}

pub fn get_traning_data() -> Vec<(Array1<f32>, Array1<f32>)> {
    let images = read_images_file("data/train-images-idx3-ubyte");
    let labels = read_labels_file("data/train-labels-idx1-ubyte");

    let mut traning_data: Vec<(Array1<f32>, Array1<f32>)> = Vec::new();
    for i in 0..labels.len() {
        let mut array = [0f32; 10];
        array[labels[i] as usize] = 1.0f32;
        let m_array: Array1<f32> = Array1::from_shape_vec(10, array.to_vec())
            .expect("Failed to create array");

        traning_data.push((images[i].clone(), m_array));
    }

    return traning_data;
}

pub fn get_test_data() -> (Vec<Array1<f32>>, Vec<u8>) {
    let images = read_images_file("data/t10k-images-idx3-ubyte");
    let labels = read_labels_file("data/t10k-labels-idx1-ubyte");
    
    return (images, labels);
}


