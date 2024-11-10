function visualizePredictions(input_data, data_labels, hidden_weights, output_weights, activation_func)
    images_num = 10; % Number of images to display
    cols = ceil(sqrt(images_num));  % Columns
    rows = ceil(images_num / cols); % Rows

    % Create a figure and maximize the window
    figure;
    set(gcf, 'WindowState', 'maximized'); % Set figure to fullscreen

    for i = 1:images_num
        % Get the i-th image and label
        input_vector = input_data(:, i);
        true_label = data_labels(i);

        % Make a prediction through multiple layers
        layer_output = input_vector;
        for j = 1:length(hidden_weights)
            layer_input = [1; layer_output]; % Add bias
            layer_output = activation_func(hidden_weights{j} * layer_input);
        end
        % Output layer
        output = output_weights * [1; layer_output];
        [~, predicted_label] = max(output); % Find the index of the max output
        predicted_label = predicted_label - 1; % Adjust for 0-based indexing

        % Reshape the image for display
        img = reshape(input_vector, 28, 28);

        % Display the image with both true and predicted labels in the title
        subplot(rows, cols, i);
        imshow(img, 'InitialMagnification', 'fit');
        title(sprintf('True: %d, Pred: %d', true_label, predicted_label), 'FontSize', 10);
    end
end
