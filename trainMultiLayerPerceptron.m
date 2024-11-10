function [hidden_weights, output_weights, error] = trainMultiLayerPerceptron(activation_func, derivative_activation_func, hidden_units_array, input_data, target_values, epochs, batch_size, learning_rate, use_plot)
    % hidden_units_array is an array containing the number of neurons for
    % each hidden layer, so the length of the array is the number of hidden
    % layer
    num_hidden_layers = length(hidden_units_array);
    
    % The number of training vectors
    training_set_size = size(input_data, 2); % number of samples based on the columns
    
    % Input and output dimensions
    input_dimensions = size(input_data, 1); % 784 input
    output_dimensions = size(target_values, 1); % 10 output
    
    % Initialize weights for all layers (including bias)
    hidden_weights = cell(num_hidden_layers, 1);
    hidden_weights{1} = rand(hidden_units_array(1), input_dimensions + 1) ./ sqrt(input_dimensions + 1);
    for i = 2:num_hidden_layers
        hidden_weights{i} = rand(hidden_units_array(i), hidden_units_array(i-1) + 1) ./ sqrt(hidden_units_array(i-1) + 1);
    end
    output_weights = rand(output_dimensions, hidden_units_array(end) + 1) ./ sqrt(hidden_units_array(end) + 1);
    
    % Initialize arrays for tracking metrics
    error_history = zeros(epochs, output_dimensions);
    weight_norm_history = zeros(epochs, num_hidden_layers + 1);
    accuracy_history = zeros(epochs, 1);
    
    % Setup plotting
    if (use_plot)
        figure;
        sgtitle('Training Progress');
        subplot(2, 2, 1); hold on; title('Training Error over Epochs'); xlabel('Epoch'); ylabel('Error');
        subplot(2, 2, 2); hold on; title('Weight Norms over Epochs'); xlabel('Epoch'); ylabel('Norm');
        subplot(2, 2, 3); hold on; title('Accuracy over Epochs'); xlabel('Epoch'); ylabel('Accuracy (%)');
    end

    % Generate random sequence for training
    randomseq = ceil(rand(1, epochs * batch_size) * training_set_size);

    for t = 1:epochs
        batch_error = 0;
        batch_true_predictions = 0;
        
        for k = 1:batch_size
            % Select input vector
            n = randomseq((t-1)*batch_size + k);
            input_vector = [1; input_data(:, n)];  % Add bias input
            
            % Forward pass
            layer_outputs = cell(num_hidden_layers + 1, 1);
            layer_inputs = cell(num_hidden_layers + 1, 1);
            
            layer_inputs{1} = hidden_weights{1} * input_vector;
            layer_outputs{1} = activation_func(layer_inputs{1});
            
            for i = 2:num_hidden_layers
                layer_inputs{i} = hidden_weights{i} * [1; layer_outputs{i-1}];
                layer_outputs{i} = activation_func(layer_inputs{i});
            end
            
            layer_inputs{end} = output_weights * [1; layer_outputs{end-1}];
            layer_outputs{end} = activation_func(layer_inputs{end});
            
            % Backpropagation
            target_vector = target_values(:, n);
            deltas = cell(num_hidden_layers + 1, 1);
            
            deltas{end} = derivative_activation_func(layer_inputs{end}) .* (layer_outputs{end} - target_vector);
            
            for i = num_hidden_layers:-1:1
                if i == num_hidden_layers
                    deltas{i} = derivative_activation_func(layer_inputs{i}) .* (output_weights(:, 2:end)' * deltas{end});
                else
                    deltas{i} = derivative_activation_func(layer_inputs{i}) .* (hidden_weights{i+1}(:, 2:end)' * deltas{i+1});
                end
            end
            
            % Update weights
            delta_output_weights = learning_rate * deltas{end} * [1; layer_outputs{end-1}]';
            output_weights = output_weights - delta_output_weights;
            
            for i = num_hidden_layers:-1:1
                if i == 1
                    delta_hidden_weights = learning_rate * deltas{i} * input_vector';
                else
                    delta_hidden_weights = learning_rate * deltas{i} * [1; layer_outputs{i-1}]';
                end
                hidden_weights{i} = hidden_weights{i} - delta_hidden_weights;
            end
            
            % Calculate error and accuracy for this sample
            batch_error = batch_error + norm(layer_outputs{end} - target_vector, 2);
            [~, predicted_label] = max(layer_outputs{end});
            [~, true_label] = max(target_vector);
            if predicted_label == true_label
                batch_true_predictions = batch_true_predictions + 1;
            end
        end
        
        % Store metrics
        error = batch_error / batch_size;
        error_history(t, :) = error;
        
        for i = 1:num_hidden_layers
            weight_norm_history(t, i) = norm(hidden_weights{i}, 'fro');
        end
        weight_norm_history(t, end) = norm(output_weights, 'fro');
        
        accuracy = (batch_true_predictions / batch_size) * 100;
        accuracy_history(t) = accuracy;
    end
    
    % Finalize plots
    if (use_plot)
        subplot(2, 2, 1); plot(1:epochs, error_history, 'b');
        subplot(2, 2, 2);
        for i = 1:num_hidden_layers+1
            plot(1:epochs, weight_norm_history(:,i), 'Color', rand(1,3));
        end
        subplot(2, 2, 3); plot(1:epochs, accuracy_history, 'k');
    end
end
