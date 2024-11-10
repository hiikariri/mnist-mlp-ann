function y = softmax(x)
    % Subtract the max for numerical stability
    e_x = exp(x - max(x));
    y = e_x ./ sum(e_x);
end