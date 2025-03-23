import handwritten_digits_loader
import network

training_data, test_data = handwritten_digits_loader.load_data_wrapper()

net = network.Network([784,30,30,30,10])
net.SGD(training_data, epochs=100, mini_batch_size=1000, eta=3.0, test_data=test_data)