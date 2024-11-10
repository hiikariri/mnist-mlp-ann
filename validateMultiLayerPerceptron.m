function [correctlyClassified, classificationErrors, predictionTime, confusionMatrix] = validateMultiLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues, labels)
    testSetSize = size(inputValues, 2);
    classificationErrors = 0;
    correctlyClassified = 0;
    numClasses = max(labels) + 1; % Assume labels are 0-based
    
    % Initialize confusion matrix
    confusionMatrix = zeros(numClasses, numClasses);
    
    % Start timing
    tic;
    for n = 1:testSetSize
        inputVector = inputValues(:, n);
        outputVector = evaluateMultiLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputVector);
        
        predictedClass = decisionRule(outputVector);
        trueClass = labels(n) + 1; % Adjust for 1-based indexing
        
        % Update correctly classified count and errors
        if predictedClass == trueClass
            correctlyClassified = correctlyClassified + 1;
        else
            classificationErrors = classificationErrors + 1;
        end
        
        % Update confusion matrix
        confusionMatrix(trueClass, predictedClass) = confusionMatrix(trueClass, predictedClass) + 1;
    end
    % End timing and store the elapsed time
    predictionTime = toc;
end

function class = decisionRule(outputVector)
    [~, class] = max(outputVector);
end

function outputVector = evaluateMultiLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputVector)
    % Forward pass through the hidden layers
    layerOutput = inputVector;
    for i = 1:length(hiddenWeights)
        layerInput = [1; layerOutput]; % Add bias
        layerOutput = activationFunction(hiddenWeights{i} * layerInput);
    end
    % Forward pass through the output layer
    outputVector = activationFunction(outputWeights * [1; layerOutput]);
end
