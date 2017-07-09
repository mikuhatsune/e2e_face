function crop_and_create_h5(BBOXLIST, DATADIR, OUTDIR, name, scale)
%clear, close all
%BBOXLIST = 'data/webface_raw_bbox.txt';
%DATADIR = '../CASIA-WebFace';
%OUTDIR = 'data';
%name = 'webface';
%scale = 1.3;

%% create database
f = fopen(BBOXLIST,'r');
DATA = textscan(f, '%s%d%f%f%f%f');
fclose(f);

W0 = 128; H0 = 128;

CHUNK_SIZE = 12800; FILE_SIZE = CHUNK_SIZE * 10;

batchI = zeros(W0,H0,1, CHUNK_SIZE, 'uint8');  % image
batchId = zeros(1, CHUNK_SIZE, 'uint32');      % identity number

ih5 = 1;
totalct = Inf;
k = 0;
if DATADIR(end) ~= '/', DATADIR = [DATADIR '/']; end
if OUTDIR(end) ~= '/', OUTDIR = [OUTDIR '/']; end
FILE = fopen([OUTDIR name '.txt'], 'w');

n = length(DATA{1});
perm = randperm(n);
fprintf('%d files to process\n', n)

for p = 1:n
    i = perm(p);
    I = uint8(imread([DATADIR DATA{1}{i}]));
    W = size(I,2); H = size(I,1);
    
    b = [DATA{3}(i), DATA{4}(i), DATA{5}(i), DATA{6}(i)];
    S = max(b(3:4)) * scale;
    b(1:2) = b(1:2) + (b(3:4) - S) / 2;
    b(3:4) = S;
    I = safecrop(mean(I,3), b);
    %whos I, imwrite(uint8(I), 't.jpg'), pause
    
    I = imresize(I, [H0,W0]);
    
    % Append to buffer
    k = k+1;
%     batchI(:,:,:,k) = permute(I(:,:,[3,2,1]), [2, 1, 3]);
    batchI(:,:,:,k) = I';
    batchId(:,k) = DATA{2}(i);

    % Write batches to database
    if k == CHUNK_SIZE
        k = 0;
        if totalct >= FILE_SIZE
            filename = sprintf('%s%s%d.h5', OUTDIR, name, ih5);
            ih5 = ih5+1;
            totalct = 0;
            %fprintf(FILE, '%s\n', fullfile(pwd, filename));
            fprintf('created database %s\n', filename)
            if exist(filename, 'file')
                fprintf('Existing file is replaced\n')
                delete(filename);
            end
            h5create(filename, '/I', [W0 H0 1 Inf], 'Datatype','uint8', 'ChunkSize', [W0 H0 1 CHUNK_SIZE]); % width, height, channels, number
            h5create(filename, '/Id', [1 Inf], 'Datatype','uint32', 'ChunkSize', [1 CHUNK_SIZE]);
            [~,info] = fileattrib(filename);
            fprintf(FILE, '%s\n', info.Name);
        end
        % store to hdf5
        h5write(filename, '/I', batchI, [1,1,1,totalct+1], size(batchI));
        h5write(filename, '/Id', batchId, [1,totalct+1], size(batchId));
        info = h5info(filename);
        totalct = info.Datasets(1).Dataspace.Size(end);
    end
    if mod(k,1000) == 0
        fprintf('%d %d/%d\n', k, p, n)
    end
end

fclose(FILE);
if k > 0
    batchI = batchI(:,:,:,1:k);
    batchId = batchId(:,1:k);
    h5write(filename, '/I', batchI, [1,1,1,totalct+1], size(batchI));
    h5write(filename, '/Id', batchId, [1,totalct+1], size(batchId));
    info = h5info(filename);
    totalct = info.Datasets(1).Dataspace.Size(end);
end
