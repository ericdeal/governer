function [output] = compfun(maindirectory,lentyp,firstframe,lastframe,inum,AVGloline,AVGhiline,H,L,N,val)

numframes = lastframe - firstframe + 1;
numoutput = (numframes - mod(numframes,inum))/inum;
runlen = (numoutput-1)*inum + 1;

% initialise large arrays
init   = firstframe:lastframe;
output = zeros(H,L,numoutput);
calib  = val(ones(H*L,1),:);
input  = zeros(H,L,inum);
vidvec = zeros(H*L,numoutput);
count = 0;
for i = 1:inum:runlen
    % initialise counter
    count = count + 1;
    % read in frames to be averaged
    for j = 1:inum
        input(:,:,j) = imread(sprintf('%simages/%s/%i.tif',maindirectory,lentyp,init(i+j-1)));
    end
    % average images
    vidavg = sum(im2double(input),3)/inum; 
    % normalize
    vidnorm = vidavg/max(max(vidavg));
    % string out into vector
    vidvec(:,count) = vidnorm(:);
end
% do frame comparison
parfor i = 1:numoutput
    % assign vector
    vidtemp = vidvec(:,i);
    % and repeat vector as rows into a matrix with N rows for fast comparison with light intensity mapping
    repvid = vidtemp(:,ones(1,N));
    
    % The logical comparison of single frame with calibration matrix
    % The light intensity value of a particular pixel is repeated N times in
    % the repvid matrix. It is compared to both the AVGlo and AVGhi matrices,
    % and this will return zeros for all comparisons except for the one where
    % it is both smaller than AVGhi and greater than AVGlo. The result can
    % then be multiplied with the calib matrix to return a frame of particle
    % densities instead of light intensities.
    Compare = xor((repvid > AVGloline),(repvid >= AVGhiline));
    % multiply by specially shaped calibration matrix to get particle density
    final = calib.*Compare;
    % form array back into original H by L by N shape.
    final = reshape(final,H,L,N);
    % collapse array to 2D
    output(:,:,i) = sum(final,3);
end
