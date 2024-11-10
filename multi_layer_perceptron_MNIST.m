function [] = multi_layer_perceptron_MNIST()
% multi layer perceptron using the MNIST dataset.
    
    % Load MNIST
    input_data = loadMNISTImages('train-images.idx3-ubyte');
    data_labels = loadMNISTLabels('train-labels.idx1-ubyte');

    % Visualization of the data
    %              image 1     image 2     image 3  ...  image N
    % pixel 1        0.0         0.0         0.0           0.0            
    % pixel 2        0.0         2.0         0.0           0.0
    % pixel 3        0.0        255.0        0.0           0.0
    % pixel 4        1.0         0.0         0.0           0.0
    %   .             .           .           .             .
    %   .             .           .           .             .
    %   .             .           .           .             .
    % pixel 784     255.0        0.0         0.0           0.0

    % Data number used for training, for now use 6000 data
    input_data = input_data(:, 1:6000);
    data_labels = data_labels(1:6000);


    % Count occurrences of each class (0-9), too see the number of data
    % used in every class
    classCounts = histcounts(data_labels, 0:10);

    % Display the counts for each class
    fprintf('Number of instances for each class:\n');
    for class = 0:9
        fprintf('Class %d: %d instances\n', class, classCounts(class + 1));  % +1 because histcounts returns counts for 1-based indexing
    end

    % Transform the labels to correct target values.
    target_values = 0 .* ones(10, size(data_labels, 1));
    for n = 1:size(data_labels, 1)
        target_values(data_labels(n) + 1, n) = 1;
    end
    
    % Form of MLP [x y]
    hidden_units_num = [50];
    
    % Learning rate
    learning_rate = 0.05;
    
    % Activation function
    activation_func = @logisticSigmoid;
    derivative_activation_func = @dLogisticSigmoid;
    
    % Batch size and epochs
    batch_size = 600;
    epochs = 250;
    
    fprintf('Train perceptron with %d hidden layers and [%s] hidden units per layer.\n', length(hidden_units_num), num2str(hidden_units_num));
    fprintf('Learning rate: %d.\n', learning_rate);
    
    tic; % Starts the timer
    
    % Training
    [hidden_weights, output_weights, ~] = trainMultiLayerPerceptron(activation_func, derivative_activation_func, hidden_units_num, input_data, target_values, epochs, batch_size, learning_rate, true);
    
    elapsed_time = toc; % Stop the timer
    fprintf('Training completed in %.2f seconds.\n', elapsed_time);

    % Validation
    input_data_validation = loadMNISTImages('t10k-images.idx3-ubyte');
    validation_data_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

    % Data number used for training
    input_data_validation = input_data_validation(:, 1:1000);
    validation_data_labels = validation_data_labels(1:1000);
    
    fprintf('Validation:\n');
    
    [true_predictions, false_predictions, prediction_time, confusion_matrix] = validateMultiLayerPerceptron(activation_func, hidden_weights, output_weights, input_data_validation, validation_data_labels);
     
    fprintf('False predictions: %d\n', false_predictions);
    fprintf('True Predictions: %d\n', true_predictions);
    fprintf('Accuracy: %.2f.\n', ((true_predictions / (true_predictions + false_predictions)) * 100));
    fprintf('Prediction time: %.2f\n', prediction_time);
    
    % Visualize predictions
    visualizePredictions(input_data_validation, validation_data_labels, hidden_weights, output_weights, activation_func);
    
    % After computing the confusion matrix
    total_counts = sum(confusion_matrix, 2);  % Total number of instances for each true class
    fprintf('Data used: \n');
    for i = 1:length(total_counts)
        fprintf('Class %d: %d instances\n', i, total_counts(i));
    end

    % Create a new figure window specifically for the confusion matrix
    figure;
    confusionChart = confusionchart(confusion_matrix);
    confusionChart.Title = 'Confusion Matrix for MLP';
end
