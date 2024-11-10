function images = loadMNISTImages(filename)
%loads 28x28 pixels of MNIST images in matrix

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

images_num = fread(fp, 1, 'int32', 0, 'ieee-be');
rows_num = fread(fp, 1, 'int32', 0, 'ieee-be');
cols_num = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, cols_num, rows_num, images_num);
images = permute(images, [2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end
