function [status, report] = convertToEnhancedGrayscale(inputFolder, outputFolder, fileExt, claheParams)

    if nargin < 3 || isempty(fileExt)
        fileExt = '*.jpg';
    end
    
    if nargin < 4
        claheParams = struct();
    end

    if ~isfield(claheParams, 'clipLimit'), claheParams.clipLimit = 0.01; end
    if ~isfield(claheParams, 'distribution'), claheParams.distribution = 'uniform'; end
    if ~isfield(claheParams, 'numTiles'), claheParams.numTiles = [4 4]; end
    if ~isfield(claheParams, 'noiseReduction'), claheParams.noiseReduction = true; end

    status = 0;
    report = struct();
    report.startTime = datetime();
    report.processedFiles = 0;
    report.failedFiles = 0;
    report.failedList = {};
    
    try
        if ~exist(inputFolder, 'dir')
            error('Folder input tidak ditemukan: %s', inputFolder);
        end

        if ~exist(outputFolder, 'dir')
            mkdir(outputFolder);
        end

        imageFiles = dir(fullfile(inputFolder, fileExt));
        numImages = length(imageFiles);
        
        if numImages == 0
            error('Tidak ada gambar ditemukan di %s dengan ekstensi %s', inputFolder, fileExt);
        end

        hWait = waitbar(0, 'Memulai proses...', 'Name', 'Grayscale Conversion');

        for i = 1:numImages
            try
                waitbar(i/numImages, hWait, sprintf('Proses: %d/%d', i, numImages));

                imgPath = fullfile(inputFolder, imageFiles(i).name);
                img = imread(imgPath);

                if ~ismember(size(img,3), [1 3])
                    error('Format gambar tidak didukung: %s', imgPath);
                end

                if size(img,3) == 3
                    grayImg = rgb2gray(img);
                else
                    grayImg = img;
                end

                if claheParams.noiseReduction
                    grayImg = medfilt2(grayImg, [3 3]);
                end

                grayImg = im2double(grayImg);

                enhancedImg = adapthisteq(grayImg,...
                    'ClipLimit', claheParams.clipLimit,...
                    'Distribution', claheParams.distribution,...
                    'NumTiles', claheParams.numTiles,...
                    'Range', 'original');

                if isa(img, 'uint8')
                    enhancedImg = im2uint8(enhancedImg);
                elseif isa(img, 'uint16')
                    enhancedImg = im2uint16(enhancedImg);
                end

                [~,name,ext] = fileparts(imageFiles(i).name);
                outputPath = fullfile(outputFolder, [name ext]);
                imwrite(enhancedImg, outputPath, 'Quality', 100);

                report.processedFiles = report.processedFiles + 1;

            catch ME
                report.failedFiles = report.failedFiles + 1;
                report.failedList{end+1} = imageFiles(i).name;
                continue;
            end
        end

        close(hWait);

        report.endTime = datetime();
        report.duration = report.endTime - report.startTime;
        report.successRate = (report.processedFiles/numImages)*100;

        status = 1;

    catch ME
        report.error = ME.message;
    end
end
