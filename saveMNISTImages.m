function [] = saveMNISTImages(images, n, k)
    % Check and create MNIST folder if it doesn't exist
    if ~exist('MNIST', 'dir')
        mkdir('MNIST');
    end

    % Save images every k-th image up to n images to MNIST folder
    for i = 1:n
        imwrite(reshape(images(:, i * k), 28, 28), strcat('MNIST/', num2str(i * k), '.png'));
    end
end
