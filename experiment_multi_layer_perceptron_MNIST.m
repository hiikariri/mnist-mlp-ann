function [] = experiment_multi_layer_perceptron_MNIST()
    % Define experimental parameters
    epoch_options = [100, 200, 300, 400, 500, 1000, 5000];
    layer_options = [1, 2, 3];
    neuron_options = [35, 50, 70];
    
    % Load MNIST training data
    input_data = loadMNISTImages('train-images.idx3-ubyte');
    data_labels = loadMNISTLabels('train-labels.idx1-ubyte');
    input_data = input_data(:, 1:6000);
    data_labels = data_labels(1:6000);
    
    % Transform labels into one-hot encoding for training targets
    target_values = zeros(10, size(data_labels, 1));
    for n = 1:size(data_labels, 1)
        target_values(data_labels(n) + 1, n) = 1;
    end
    
    % Load MNIST validation data
    input_data_validation = loadMNISTImages('t10k-images.idx3-ubyte');
    validation_data_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
    input_data_validation = input_data_validation(:, 1:1000);
    validation_data_labels = validation_data_labels(1:1000);
    
    % Set common parameters for training
    use_plot = false;
    batch_size = 600;
    learning_rate = 0.05;
    activation_func = @logisticSigmoid;
    derivative_activation_func = @dLogisticSigmoid;
    
    % Initialize results for plotting
    configurations = {};
    accuracy_values = [];
    training_times = [];
    prediction_times = [];
    
    % Loop through each combination of epochs, layers, and neurons
    for epochs = epoch_options
        for num_layers = layer_options
            for neurons = neuron_options
                % Define the structure of the hidden layers
                hidden_units_num = repmat(neurons, 1, num_layers);
                
                % Display configuration details
                fprintf('\nTraining with %d epochs, %d layers, %d neurons per layer.\n', epochs, num_layers, neurons);
                
                % Start timing training
                tic;
                [hidden_weights, output_weights, ~] = trainMultiLayerPerceptron(activation_func, derivative_activation_func, hidden_units_num, input_data, target_values, epochs, batch_size, learning_rate, use_plot);
                training_time = toc;
                
                % Validate the model
                [true_predictions, false_predictions, prediction_time, ~] = validateMultiLayerPerceptron(activation_func, hidden_weights, output_weights, input_data_validation, validation_data_labels);
                
                % Calculate accuracy
                accuracy = (true_predictions / (true_predictions + false_predictions)) * 100;
                
                % Store results for plotting
                configurations{end+1} = sprintf('%d epochs, %d layers, %d neurons', epochs, num_layers, neurons);
                accuracy_values(end+1) = accuracy;
                training_times(end+1) = training_time;
                prediction_times(end+1) = prediction_time;
            end
        end
    end
    
    % Plot Accuracy vs Configuration
    figure;
    bar(categorical(configurations), accuracy_values);
    title('Accuracy vs Configuration');
    xlabel('Configuration');
    ylabel('Accuracy (%)');
    xtickangle(45);
    grid on;
    
    % Plot Training Time vs Configuration
    figure;
    bar(categorical(configurations), training_times);
    title('Training Time vs Configuration');
    xlabel('Configuration');
    ylabel('Training Time (seconds)');
    xtickangle(45);
    grid on;
    
    % Plot Prediction Time vs Configuration
    figure;
    bar(categorical(configurations), prediction_times);
    title('Prediction Time vs Configuration');
    xlabel('Configuration');
    ylabel('Prediction Time (seconds)');
    xtickangle(45);
    grid on;
end
