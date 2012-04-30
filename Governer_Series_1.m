%% --------------------------------------- THE GOVERNER -------------------------------------------
clc, warning off %#ok<WNOFF>
clear all, close all, %matlabpool open 4

% Initialise parameters
% set series and experiment name
series = 1;
experiment = 18;
% set the number of images for the 2 sections
number_images_short = 760;
number_images_long = 760;
% set the crop view image number
impicked = 300;
% set frame rates
framerate = [300 5000];
% set calibration to be used
caltype = 1;
% set frame choose rate for videos. Skip frames to save on time
skiprate = 1;
pblskiprate = 1;
% select the modules to use (0 don't use, 1 use)
Image_prep    = 0;
Data_analysis = 0;
Particle_conc = 1;
Visual        = [0 0 0 0 0 1 0];
% legend for Visual vector:
% 1 = normal unprocessed video
% 2 = normalised mass vs normalised time plot
% 3 = horizontally integrated plots and video
% 4 = particle concentration video
% 5 = PBL video
% 6 = PBL video with wavelength analysis
% 7 = snake plot of wavelength analysis

%% ------------------------------------------------------------------------------------------------

% settings for image contrast
% increase image contrast: (1) yes, (0) no
increase_contrast = 1;
% set darkening factor gamma
gamma = 1.8;
% set directory of original images
origimdirectory = sprintf('C:/Users/Eric/Documents/project/Unprocessed_exps/Series_%i/S%iE%i/',series,series,experiment);
% set the directory to which everything will be saved
maindirectory = sprintf('C:/Users/Eric/Documents/project/Processed_Experiments/S%iE%i/',series,experiment);
% load data from explog program
load(sprintf('%sdatapass.mat',maindirectory));
% frame data
% images to be processed for short run
firstframe(1) = 1;
lastframe(1)  = number_images_short;
% images to be processed for long run
firstframe(2) = 1;
lastframe(2)  = number_images_long;
for ii = 1:2
    numframes(ii) = lastframe(ii) - firstframe(ii) + 1; %#ok<SAGROW>
end
% set resolution of calibration
R(1) = 4;
% set frame size based on calibration: y then x
switch caltype
    case 1
        calibrationsize = [450 1300];
    case 2
        calibrationsize = [480 1300];
    case 3
        calibrationsize = [480 1300];
end
% Band pass wavelengths in mm.
cutoff = [10 400];
% set physical width of a pixel for this photo series
pixelwidth = 0.5; % in mm
% set experiment name
lentyp             = {'short' 'long'};
exp                = datapass.name;
expnameshort       = [exp 'short'];
expnamelong        = [exp 'long'];
expname            = {expnameshort expnamelong};
% set name of original images
origimname = expname;
tic

%% ------------------------------------- IMAGE PREPARATION ----------------------------------------
% The image preparation module is used to take the original images captured
% in the experiment and crop them to various sizes. The crop view image
% parameter 'impicked' set above is the number that corresponds to the
% image that will be displayed for cropping.
% The first task is to crop the image down to the size of the calibration 
% images associated with each series, essentially to crop the images down 
% to the size of the tank. This is accomplished simply by selecting the inside
% of the bottom right corner of the tank. Inside corner meaning right at the 
% edge of the tank, but still where water can reside. 
% The second task is to draw a box around the particle boundary layer (PBL)
% with the cursor provided. The box should stretch from edge to edge in the
% x direction. For the y direction it should span from just above the
% highest point where the PBL meets the cloud to just below the lowest
% visible extent of the fingers. Once the box is drawn, simply double click
% the box to move on to the next step.
% The third task of this module is to select the cloud itself in the same
% manner as the PBL was selected. The box should span again from side to
% side, and from the lowest extent of the cloud to its highest. Again,
% double click inside the box to move on. 
% After the three regions have been chosen the module will go on to apply
% these cropping parameters to all the images from the experiment. It will
% save the images from the first step simply as number.tif. Where the
% corresponding number is from the original image, the PBL selection as
% pblnumbertif, and the cloud as cloudnumber.tif. 
% Once this module has been run for an experiment, it should not be
% necessary to run it again.


% decide if image prep is necessary
if(Image_prep == 1)
    % alert tag
    sprintf('Crop Images')
    %% Step 1, Load example image and get various crop sizes
    % load image
    testim = imread(sprintf('%s%s/%g.tif',origimdirectory,char(lentyp{1}),impicked));
    [origL,origH] = size(testim);
    
    % show and crop image tank area
    % set tester to 2 so it is not necessary to manually crop while
    % testing code
    tester = 1;
    switch tester
        case 1 % usual case
            f1 = figure(1);
            imshow(testim)
            sprintf('select bottom right hand corner of tank')
            p = ginput(1);
            close(f1)
        case 2% to save time when testing code
            p = [1300 450];
    end
    
    % get the crop parameters for tank area
    xmaxtank = round(p(1));
    ymaxtank = round(p(2));
    ymintank = (ymaxtank-calibrationsize(1)+1);
    xmintank = (xmaxtank-calibrationsize(2)+1);
    
    % get crop sizes fro cloud and pbl areas
    switch tester
        case 1 % normal case
            % show and crop image for pbl
            f1 = figure(1);
            testim = testim(ymintank:ymaxtank,xmintank:xmaxtank);
            imshow(testim)
            sprintf('select the area around the PBL to be vertically averaged for fourier transformation')
            [~ , ~ , ~, rect] = imcrop(testim);
            % show and crop image for cloud area
            sprintf('select the area around the cloud to be included in the mass summation')
            [~ , ~ , ~, rect2] = imcrop(testim);
            close(f1)
        case 2 % to speed up code testing
            rect = [1 138 1300 195];
            rect2 = [1 15 1300 50];
    end
    
    % get the crop parameters for pbl
    xminpbl = round(rect(1)) + 1;
    xmaxpbl = round(xminpbl + rect(3)) - 2;
    yminpbl = round(rect(2));
    ymaxpbl = round(yminpbl + rect(4));
    
    % get the crop parameters for cloud area
    xmincloud = round(rect2(1)) + 1;
    xmaxcloud = round(xmincloud + rect2(3)) - 2;
    ymincloud = round(rect2(2));
    ymaxcloud = round(ymincloud + rect2(4));
    
    %% Step 2, Crop images
    for ii = 1:2
        % alert tag
        if(ii == 1)
            sprintf('Preparing images from first section')
        else
            sprintf('Preparing images from second section')
        end
        
        % bring in series of images and crop tank, cloud and pbl size
        % initialise arrays
        cropim = zeros(ymaxtank-ymintank+1,xmaxtank-xmintank+1,numframes(ii)); % initialise image array
        cloudim = zeros(ymaxcloud-ymincloud+1,xmaxcloud-xmincloud+1,numframes(ii)); % initialise image array
        pblim = zeros(ymaxpbl-yminpbl+1,xmaxpbl-xminpbl+1,numframes(ii)); % initialise image array
        for i = 1:numframes(ii)
            % read in images
            origim = imread(sprintf('%s%s/%g.tif',origimdirectory,lentyp{ii},i)); %#ok<*PFBNS>
            % crop to tank size
            cropimtemp = origim(ymintank:ymaxtank,xmintank:xmaxtank);
            % save tank images
            imwrite(cropimtemp,sprintf('%simages/%s/%g.tif',maindirectory,lentyp{ii},i),'tif')
            % save in array
            cropim(:,:,i) = im2double(cropimtemp);
            % crop to cloud size
            cloudimtemp = cropimtemp(ymincloud:ymaxcloud,xmincloud:xmaxcloud);
            % crop to pbl size
            pblimtemp = cropimtemp(yminpbl:ymaxpbl,xminpbl:xmaxpbl);
            % adjust contrast
            if(increase_contrast == 1)
                pblimtemp = imadjust(pblimtemp,stretchlim(pblimtemp),[],gamma);
            end
            % save cloud and pbl images
            imwrite(cloudimtemp,sprintf('%simages/%s/cloud%g.tif',maindirectory,lentyp{ii},i),'tiff')
            imwrite(pblimtemp,sprintf('%simages/%s/pbl%g.tif',maindirectory,lentyp{ii},i),'tiff')
            % crop to cloud size
            cloudim(:,:,i) = im2double(cloudimtemp);
            % crop to pbl size
            pblim(:,:,i) = im2double(pblimtemp);
        end
        
        % save crop parameters
        if( ii == 1)
            % save crop parameters for tank area
            tankcrop = [xmintank xmaxtank ymintank ymaxtank];
            % save crop parameters for cloud area
            cloudcrop = [xmincloud xmaxcloud ymincloud ymaxcloud];
            % save crop parameters for pbl
            pblcrop = [xminpbl xmaxpbl yminpbl ymaxpbl];
            % save crop parameters in file
            filename = sprintf('%s%s_crop_parameters',maindirectory,exp);
            save(filename,'tankcrop','cloudcrop','pblcrop')
        end
    end
    % alert tag
    sprintf('Finished preparing images')

end
toc
tic

%% --------------------------------------- DATA ANALYSIS ------------------------------------------

if(Data_analysis == 1)
    % alert tag
    sprintf('Analysing PBL fingering wavelengths')
    for ii = 1:2
        %% Step 1, Create space series from images
        
        % run function to extract space series from images
        % a script to read in set of pictures from my experiment, use an example image (impicked) to find the PBL,
        % delineate a box around it, and take a vertical average of the pixel
        % values in order to give a 1D space series that I can transfer into the
        % frequency domain.
        
        % create vector filled with image names
        imagenum = 1:numframes(ii);
        
        % read in images
        
        if(Image_prep == 0)
            parfor i = 1:numframes(ii)
                % read in single frame
                tempimg = imread(sprintf('%simages/%s/pbl%g.tif',maindirectory,lentyp{ii},i));
                % convert to double
                pblim(:,:,i) = im2double(tempimg);
            end
        end
        
        parfor i = 1:numframes(ii)
            % take vertical average, essentially compressing
            % images into horizontal vectors spanning the frame
            avgd_img(i,:) = sum(pblim(:,:,i));
        end
        
        % Save space series
        filename = sprintf('%s%s_Integrated_PBL_values_%s',maindirectory,exp,lentyp{ii});
        save(filename,'avgd_img')
        
        % return final matrix
        spcseries = avgd_img;
        
        % ---------------------------------------------------------------------
        % code to create artificial space series
        %
        %     %set the wavelengths of the synthesised function
        %     wavelths = [100 200];
        %     % set number of elements in data series
        %     lenfun = 1299;
        %     % set sampling spacing
        %     dxfun = 0.488;
        %     % set physical length of series
        %     Lfun = (lenfun-1)*dxfun;
        %     % set number of different frequencies in synthetic series
        %     numfreqfun = length(wavelths);
        %     % set x
        %     xfun = 0:dxfun:Lfun;
        %     % multiply x by 2*pi for sine series
        %     xpifun = 2*pi*xfun;
        %     % set wavelengths to appear in synthetic series
        %     wavefun = wavelths;
        %     % translate to wavenumbers. The official translation for now is
        %     % wavenumber = L/wavelength. A 2*pi might be added later.
        %     wavenumfun = (1./wavefun)*Lfun;
        %     % create empty array to take final series
        %     sumfun = zeros(length(xfun),1);
        %     % synthesise series
        %     for i = 1:numfreqfun
        %         fun(:,i) = sin(wavenumfun(i)*xpifun/Lfun);
        %         sumfun = sumfun + fun(:,i);
        %     end
        %
        %     % plot individual frequencies
        %     % color = rand(numfreq,3);
        %     % figure(200),hold all
        %     % for i = 1:numfreq
        %     %     h(i) = plot(x,fun(:,i));
        %     %     set(h(i), 'color', color(i,:))
        %     % end
        %
        %     % plot summmed frequencies
        %     figure(300)
        %     plot(xfun,sumfun)
        %     % output final series
        %     spcseries = sumfun';
        % ---------------------------------------------------------------------
        
        %% Step 2, Prep space series for fft.
        
        % this includes detrending, windowing and possibly filtering the data
        % feed data to function to be processed
        
        % function [output] = SSprep(input, pixelwidth, numframes, detrendflag, cutoff)
        % A basic time series analysis script.  Read in a file with two columns of
        % evenly-spaced data and no headers; x is time or a position;
        % yy is the value at that position. If your data is not evenly spaced then
        % fit a high-order polynomial to it and subsample the polynomial at
        % evenly-spaced intervals.
        % HANNING WINDOW:
        % HANNING(N) returns the N-point symmetric Hanning window in a column
        % vector.  Note that the first and last zero-weighted window samples
        % are not included.
        % w(n)=0.5*(1-cos((2*pi*n)/(N-1)))
        % clc, close all, clear all
        
        % Set up constant parameters
        [row,col]   = size(spcseries);               % get size of input matrix
        enx         = col;                           % Number of points in the poly dataset
        xlen        = 1:enx;                         % number of pixels in x
        xA          = (xlen*pixelwidth)'-pixelwidth; % physical length
        L2          = xA(end)-xA(1);                 % in mm, input this or use length() command
        Nyquist     = 0.5/pixelwidth;                % Half of the sample spacing wavelength in 1/mm
        shtwavelen  = cutoff(1);                     % in mm; remember Nyquist is 1.1 mm
        highwavenum = L2/shtwavelen;                 % high wavenumber cutoff
        lngwavelen  = cutoff(2);                     % in mm; longest wavelength i want to see
        lowwavenum  = L2/lngwavelen;                 % low wavenumber cutoff
        order       = 10;                            % Order of filter
        
        % Process space series
        for kk = 1:2
            switch kk
                case 1
                    % polynomial removal
                    % POLYFIT Fit polynomial to data.
                    % P = POLYFIT(X,Y,N) finds the coefficients of a polynomial P(X) of
                    % degree N that fits the data Y best in a least-squares sense. P is a
                    % row vector of length N+1 containing the polynomial coefficients in
                    % descending powers, P(1)*X^N + P(2)*X^(N-1) +...+ P(N)*X + P(N+1).
                    
                    % initialise array
                    spcseriespreppoly = zeros(row,col);
                    
                    % process each space series one at a time in loop
                    parfor i = 1:numframes(ii)
                        % figure(1), hold on
                        % Set space series
                        yy = spcseries(i,:)';
                        
                        % Build polynomial
                        fitCoeff = polyfit(xA,yy,2);
                        fity     = polyval(fitCoeff, xA);
                        % plot polynomial
                        % subplot(4,1,1), hold on, plot(xA,yy/max(yy),'b'), plot(xA,fity/max(fity),'r'), axis([0, xA(end), -1, 1])
                        % title('Unaltered space series with fitting line'), xlabel('distance (mm)'), ylabel('relative light intensity')
                        
                        % Remove polynomial
                        yy = yy - fity;
                        
                        % Remove the mean
                        yy = yy-mean(yy);
                        % plot detrended series
                        % subplot(4,1,3), hold on, plot(xA,0.5*yy/max(abs(yy)),'b'), plot(xA,zeros(enx,1),'k'), axis([0, xA(end), -.5, .5])
                        % title('Polynomial removed and mean removed series'), xlabel('distance (mm)'), ylabel('relative light intensity')
                        
                        % Window the data
                        window = hanning(enx);
                        yy     = (yy).*window;
                        % plot windowed series
                        % subplot(4,1,4), hold on, plot(xA,0.5*yy/max(abs(yy)),'b'), plot(xA,zeros(enx,1),'k'), axis([0, xA(end), -.5, .5])
                        % title('Tapered series'), xlabel('distance (mm)'), ylabel('relative light intensity')
                        
                        % Band pass the series
                        yy = fbpfilt(yy',pixelwidth,lowwavenum,highwavenum,order,0)';
                        % plot the badnpassed series
                        % subplot(4,1,2), hold on, plot(xA,0.5*yy/max(abs(yy)),'b'), plot(xA,zeros(enx,1),'k'), axis([0, xA(end), -.5, .5])
                        % title('Band passed series'), xlabel('distance (mm)'), ylabel('relative light intensity')
                        
                        % Store prepared space series
                        spcseriespreppoly(i,:) = yy;
                    end
                    
                case 2 % Differencing (filtering by differentiation in the frequency domain);
                    % Differencing will pump up power at high frequencies (short periods):
                    % f(t) = exp(iwt); f'(t) = iwexp(iwt);
                    % DIFF Difference and approximate derivative.
                    % DIFF(X), for a vector X, is [X(2)-X(1)  X(3)-X(2) ... X(n)-X(n-1)].
                    
                    % initialise array
                    spcseriesprepdiff = zeros(row,col-1);
                    
                    % process each space series one at a time in loop
                    parfor i = 1:numframes(ii)
                        % Select space series
                        xx = spcseries(i,:)';
                        
                        % difference series
                        diffxx     = diff(xx); % (dataset is one data pt shorter)
                        diffnx     = length(diffxx);
                        
                        % Remove the mean
                        diffxxmean = diffxx-mean(diffxx);
                        
                        % Window the differenced series
                        window     = hanning(diffnx);
                        diffxxwin  = (diffxxmean).*window;
                        
                        % band pass the series
                        xxfilt     = fbpfilt(diffxxwin',pixelwidth,lowwavenum,highwavenum,order,0)';
                        
                        % Store prepared space series
                        spcseriesprepdiff(i,:) = xxfilt;
                        
                        % Plot steps
                        % figure(2), hold on
                        % subplot(4,1,1), hold on, plot(xA,xx/max(xx),'b'), axis([0, xA(end), 0-1, 1])
                        % title('Unaltered space series\'), xlabel('distance (mm)'), ylabel('relative light intensity')
                        % subplot(4,1,2), hold on, plot(xA(1:end-1),0.5*diffxxmean/max(abs(diffxxmean)),'b'), plot(xA(1:end-1),zeros(diffnx,1),'k'), axis([0, xA(end-1), -.5, .5])
                        % title('differenced and mean removed series'), xlabel('distance (mm)'), ylabel('relative light intensity')
                        % subplot(4,1,3), hold on, plot(xA(1:end-1),0.5*diffxxwin/max(abs(diffxxwin)),'b'), plot(xA(1:end-1),zeros(diffnx,1),'k'), axis([0, xA(end-1), -.5, .5])
                        % title('Tapered series'), xlabel('distance (mm)'), ylabel('relative light intensity')
                        % subplot(4,1,4), hold on, plot(xA(1:end-1),0.5*xxfilt/max(abs(xxfilt)),'b'), plot(xA(1:end-1),zeros(diffnx,1),'k'), axis([0, xA(end-1), -.5, .5])
                        % title('Band passed series'), xlabel('distance (mm)'), ylabel('relative light intensity')
                        
                    end
            end
        end
        
        %% Step 3, Do the fft
        % Start fft function
        % function to execute fft for the two different data processing techniques
        
        % Set up constant parameter
        N       = 2^15;                % for zero padding
        sspowerpoly = zeros(numframes, N); % initialise output to save time
        sspowerdiff = zeros(numframes, N); % initialise output to save time
        
        % Process space series
        % polynomial removed
        parfor i = 1:numframes(ii)
            % Set space series
            polyyy = spcseriespreppoly(i,:)';
            % Calculate the FFT with polynomial removed, windowed and mean removed data
            polyz = real(fftshift(fft(polyyy,N)));
            polypower = abs(polyz).^2;
            % Store new space series
            sspowerpoly(i,:) = polypower';
        end
        % differenced
        parfor i = 1:numframes(ii)
            % Set space series
            diffyy = spcseriesprepdiff(i,:)';
            % Calculate the FFT with differenced, windowed and mean removed data\
            diffz = real(fftshift(fft(diffyy,N)));
            diffpower = abs(diffz).^2;
            % Store new space series
            sspowerdiff(i,:) = diffpower;
        end
        
        
        % take real portion
        sspowerpoly = sspowerpoly(:,end/2:end);
        % get maximum values for each frame
        maxpowerpoly = max(sspowerpoly,[],2);
        repmaxpoly = maxpowerpoly(:,ones(1,length(sspowerpoly)));
        % normalize results
        powernormpoly = sspowerpoly./repmaxpoly;
        
        % take real portion
        sspowerdiff = sspowerdiff(:,end/2:end);
        % get maximum values for each frame
        maxpowerdiff = max(sspowerdiff,[],2);
        repmaxdiff = maxpowerdiff(:,ones(1,length(sspowerdiff)));
        % normalize results
        powernormdiff = sspowerdiff./repmaxdiff;
        
        %% Step 4, -- DEFUNCT --  Deconvolve signal with prepared function to sharpen peaks
        %
        %     % Load prepared function
        %     % set up padding with zeros
        %     [~,nn] = size(powernorm);
        %     nn = (nn - 200)/2;
        %     nplus = mod(nn,2);
        %
        %     switch detrendflag
        %         case 1 % for polynomial detrending method
        %             filename = '/Users/eric/Documents/Code_test/S1E18/pwnm_dtf_1';
        %             load(filename);
        %             % pad decofun with zeros
        %             % decofun = [zeros(1,nn+nplus), decofun, zeros(1,nn)];
        %         case 2 % for difference method
        %             filename = '/Users/eric/Documents/Code_test/S1E18/pwnm_dtf_2';
        %             load(filename);
        %             % pad decofun with zeros
        %             decofun = [zeros(1,nn+nplus), decofun, zeros(1,nn)];
        %     end
        %
        %     % deconvolve with space series
        %     for ii = 1:numframes
        %         powernormxcorr = deconv(powernorm(ii,:),decofun);
        %     end
        
        %% Step 5, Prepare wavenumber and wavelength axis for plotting
        
        % Generate the wavenumber or frequency.  If these wavenumbers are wrong the
        % answer will be wrong and the imaginary part of the result will be large.
        % get number of points in padded space series
        nxpoly = length(sspowerpoly);
        % physical length of series
        Lpoly = (nxpoly-1)*pixelwidth;
        % calculate wavenumber
        wavenumberpoly = 1:nxpoly;
        % translate to wavelength (I must times L by two because I cut it in half
        % earlier when I cut sspower in half)
        wavelengthpoly = 2*Lpoly./wavenumberpoly';
        
        % Generate the wavenumber or frequency.  If these wavenumbers are wrong the
        % answer will be wrong and the imaginary part of the result will be large.
        % get number of points in padded space series
        nxdiff = length(sspowerdiff);
        % physical length of series
        Ldiff = (nxdiff-1)*pixelwidth;
        % calculate wavenumber
        wavenumberdiff = 1:nxdiff;
        % translate to wavelength (I must times L by two because I cut it in half
        % earlier when I cut sspower in half)
        wavelengthdiff = 2*Ldiff./wavenumberdiff';
        %     % plot
        %
        %     figure(11), hold on
        %     plot(powernormxcorr)
        %     %plot(wavelength,powernormxcorr(end/4:(end/4)*3)/max(powernormxcorr),'r')
        %     figure(12)
        %     plot(wavelength,powernorm)
        %     xlim([0 1000])
        
        %% Step 6, Conglomerate results
        if( ii == 1)
            % merge sections into one variable
            sspower.poly    = sspowerpoly;
            sspower.diff    = sspowerdiff;
            powernorm.poly  = powernormpoly;
            powernorm.diff  = powernormdiff;
            wavelength.poly = wavelengthpoly;
            wavelength.diff = wavelengthdiff;
        else
            begin = numframes(1) + 1;
            endin = numframes(1) + numframes(2);
            sspower.poly(begin:endin,:)   = sspowerpoly;
            sspower.diff(begin:endin,:)   = sspowerdiff;
            powernorm.poly(begin:endin,:) = powernormpoly;
            powernorm.diff(begin:endin,:) = powernormdiff;
            wavelength.poly(:,2)          = wavelengthpoly;
            wavelength.diff(:,2)          = wavelengthdiff;
        end
    end
        %% Step 7, Save results
    % set filename
    filename = sprintf('%s%s_interp%g_Data_analysis_results.mat',maindirectory,exp,R(1));
    % save it
    save(filename, 'sspower','powernorm','wavelength')
end
toc
tic

%% ------------------------------ PARTICLE CONCENTRATION COMPARISON -------------------------------

% Seperates my calibration images into groups of x by y pixels which have
% the same light intensity to particle density mapping. It then compares
% all the photos in the directory to the calibration images in order to
% figure out the particles concentration at any point in space and time as
% a function of the light intensity and pixel location
% additionally it sums up the mass in the top cloud and produces a graph of
% mass over time.

if(Particle_conc == 1)
    % alert tag
    sprintf('Comparing calibration images with experiment images')
    %% Step 1, Initialise constant parameters, counters and large arrays
    
    % Only set R(2) to zero if you need to build a new interpolated calibration matrix
    R(2) = R(1);
    % R(2) = 0;
    
    % set up calibration parameters
    switch caltype
        case 1 % calibration 1
            % number of pixels in H height and L length in every frame
            H = 450; % full size (450)
            L = 1300; % full size (1300)
            
            % Construct cells of pixels with identical mapping from light intensity to particle concentration
            r = 90; % number of rows; do 90 for a 5 x 5 cell
            c = 260; % number of columns; do 260 for a 5 x 5 cell
            
            % set number of calibration photos total
            Norig = 18;
            N = R(1)*Norig;
            
            % string of calibrated image names
            valstr = {'0.0','0.2','0.4','0.6','0.8','1.0','1.2','1.4','1.6',...
                '1.8','2.0','2.2','2.4','2.6','2.8','3.3','3.8','4.3'};
            
            % double array for calibrated image names
            valorig = [0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.3,3.8,4.3];
            
            % interpolated array of calibrated image names
            val = interp(valorig,R(1));
            
            % set propery file path
            AVGdirectory = '/Users/Eric/Documents/Project/Processed_Experiments/Session_1_Calibration/AVG/';
            %             AVGdirectory = '/Users/eric/Documents/Code_test/Calibration_1/';
            
            
        case 2 % calibration 2
            % number of pixels in H height and L length in every frame
            H = 480; % full size (450)
            L = 1300; % full size (1300)
            
            % Construct cells of pixels with identical mapping from light intensity to particle concentration
            r = 96; % number of rows; do 90 for a 5 x 5 cell
            c = 260; % number of columns; do 260 for a 5 x 5 cell
            
            % set number of calibration photos total
            Norig = 33;
            N = R(1)*Norig;
            
            % string of calibrated image names
            valstr = {'0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0',...
                '1.1','1.2','1.3','1.5','1.6','1.7','1.8','1.9','2.0','2.1','2.2',...
                '2.3','2.4','2.5','3.0','3.5','4.0','5.0','6.0','7.0','10.0','15.0'};
            
            % double array for calibrated image names
            valorig = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5,...
                1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,3.0,3.5,4.0,5.0,6.0,7.0,10.0,15.0];
            
            % interpolated array of calibrated image names
            val = interp(valorig,R(1));
            
            % set propery file path
            AVGdirectory = '/Users/Eric/Documents/Project/Processed_Experiments/Session_2_Calibration/AVG/';
            %             AVGdirectory = '/Users/eric/Documents/Code_test/Calibration_2/';
        case 3 % calibration 3
            % number of pixels in H height and L length in every frame
            H = 480; % full size (450)
            L = 1300; % full size (1300)
            
            % Construct cells of pixels with identical mapping from light intensity to particle concentration
            r = 96; % number of rows; do 90 for a 5 x 5 cell
            c = 260; % number of columns; do 260 for a 5 x 5 cell
            
            % set number of calibration photos total
            Norig = 20;
            N = R(1)*Norig;
            
            % string of calibrated image names
            valstr = {'0.0','0.2','0.4','0.6','0.8','1.0',...
                '1.2','1.4','1.6','1.8','2.0','2.5','3.0','3.5','4.0','4.5','5.0','6.0','8.0','10.0'};
            
            % double array for calibrated image names
            valorig = [0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,...
                1.6,1.8,2.0,2.5,3.0,3.5,4.0,4.5,5.0,6.0,8.0,10.0];
            
            % interpolated array of calibrated image names
            val = interp(valorig,R(1));
            
            % set propery file path
            AVGdirectory = '/Users/Eric/Documents/Project/Processed_Experiments/Session_3_Calibration/AVG/';
            %             AVGdirectory =
            %             '/Users/eric/Documents/Code_test/Calibration_2/';
    end
    
    % construct cells
    Hsub = H/r; % number of vertical pixels in each cell
    Lsub = L/c; % number of horizontal pixels in each cell
    
    % initialise large arrays that don't depend on section
    AVGorig   = zeros(H,L,Norig);
    AVG       = zeros(H,L,N);
    AVGhi     = AVG;
    AVGlo     = AVG;
    AVGloline = zeros(H*L,N);
    AVGhiline = AVGloline;
    AVGmid    = zeros(H,L,N-1);
    avgcell   = zeros(r,c,N);
    
    % Initialise physical parameters
    % density of fresh water
    rowater = 996.157; % kg/m^3
    
    % radius of particles (I assuming they are spherical)
    radparticle = 0.00015; % in m
    
    % density of particles
    roparticle = 2600; % kg/m^3
    
    % multiplying constant for particle volume density
    PVD = (3*rowater/(4*pi*(radparticle^3)*roparticle));
    
    % length and width of a single pixel
    Hpix = pixelwidth/1000; % in m
    Lpix = pixelwidth/1000; % in m
    
    % average width of tank
    width = .023; % in m
    
    % multiplying constant for mass of cloud
    MPC = Hpix*Lpix*width*rowater; % in units of kg
    
    %% Step 2, Build average particle concentration images to be used for calibrating
    % alert tag
    sprintf('Building average particle concentration images for calibration')
    
    % choose to reload old interpolated matrix or reinterpolate
    switch R(2)
        case 0
            % bring in calibrated images and store them in 3D array AVG as a stack of
            % frames H by L pixels
            parfor i = 1:Norig
                tempimg = imread(sprintf('%sAVG%s%%.tif',AVGdirectory,valstr{i}));
                AVGorig(:,:,i) = im2double(tempimg(1:H,1:L));
            end
            % Interpolate the array AVG to fabricate new calibration steps
            numb = R(1);
            for i = 1:H
                parfor j = 1:L
                    AVG(i,j,:) = interp(AVGorig(i,j,:),numb);
                end
            end
            % save new matrix for later use
            % command to save matrix:
            filename = sprintf('%sinterp%g',AVGdirectory,R(1));
            save(filename, 'AVG')
            
        case 1
            % bring in calibrated images and store them in 3D array AVG as a stack of
            % frames H by L pixels
            parfor i = 1:Norig
                tempimg = imread(sprintf('%sAVG%s%%.tif',AVGdirectory,valstr{i}));
                AVG(:,:,i) = im2double(tempimg(1:H,1:L));
            end
            
        case 2
            % reload previously 2-interpolated AVG matrix
            eval(['load ' [AVGdirectory 'AVGinterp2.mat']  ' -mat']);
            
        case 3
            % reload previously 3-interpolated AVG matrix
            eval(['load ' [AVGdirectory 'AVGinterp3.mat']  ' -mat']);
            
        case 4
            % reload previously 4-interpolated AVG matrix
            eval(['load ' [AVGdirectory 'AVGinterp4.mat']  ' -mat']);
            
    end
    
    % find average light intensity values for cells of identical map that are size Hsub by Lsub
    % first calibration 0.0% step i = 1
    for j = 1:r;
        for k = 1:c;
            % get light intensity values for each pixel in cell
            getcell = AVG((j-1)*Hsub+1:j*Hsub,(k-1)*Lsub+1:k*Lsub,1);
            % average the values over the cell
            avgcell(j,k,1) = sum(getcell(:))/(Hsub*Lsub);
            % replace pixel values in cell with averaged value
            AVG((j-1)*Hsub+1:j*Hsub,(k-1)*Lsub+1:k*Lsub,1) = avgcell(j,k,1);
        end
    end
    
    % for the final N-1 calibration steps ranging up to 3.3% particles density
    for i = 2:N;
        for j = 1:r;
            for k = 1:c;
                % get light intensity values for each pixel in cell
                getcell = AVG((j-1)*Hsub+1:j*Hsub,(k-1)*Lsub+1:k*Lsub,i);
                % average the values over the cell
                avgcell(j,k,i) = sum(getcell(:))/(Hsub*Lsub);
                % replace pixel values in cell with averaged value
                AVG((j-1)*Hsub+1:j*Hsub,(k-1)*Lsub+1:k*Lsub,i) = avgcell(j,k,i);
                % record halfstep light intensity value between the current cell and the same
                % spatially oriented cell one step below in particle density
                AVGmid((j-1)*Hsub+1:j*Hsub,(k-1)*Lsub+1:k*Lsub,i-1) = (avgcell(j,k,i) + avgcell(j,k,i-1))/2;
            end
        end
    end
    
    % create AVGlo and AVGhi, which are the halfsteps between the steps of AVG
    AVGlo(:,:,1) = AVG(:,:,1) + 0.1;
    AVGlo(:,:,2:N) = AVGmid;
    
    AVGhi(:,:,1:N-1) = AVGmid;
    AVGhi(:,:,N) = AVG(:,:,N);
    
    % reduce 3d AVGlo and AVGhi to 2d arrays that are H*L x N
    parfor i = 1:N
        tempavglo      = AVGlo(:,:,i);
        AVGloline(:,i) = tempavglo(:);
        tempavghi      = AVGhi(:,:,i);
        AVGhiline(:,i) = tempavghi(:);
    end
    
    %% Step 3, Compare frames of movie to calibration images to calculate particle concentration
    % Create calibration matrix, which is a matrix the size and shape of AVG,
    % but every frame consists of just a single value of particle density.
    % when multiplied with a compared sparse matrix of ones, and then collapsed
    % to a single frame, it will leave a frame filled with particles densities
    % that correspond to particle densities.
    
    for ii = 1:2
        % alert tag
        if(ii == 1)
            sprintf('Comparing images for particle concentration in first section')
        else
            sprintf('Comparing images for particle concentration in second section')
        end
       
        
        for kk = 1:numframes(ii)
            
            % initialise large arrays
            Fval = zeros(H,L,numframes(ii));
            Mval = zeros(numframes(ii),1);
            
            % build calib array
            calib = val(ones(H*L,1),:);
            
            parfor i = 1:numframes(ii)
                % read in single frame
                tempimg = imread(sprintf('%simages/%s/%i.tif',maindirectory,lentyp{ii},i));
                
                % convert to double
                vid = im2double(tempimg(1:H,1:L));
                
                % string out into vector
                vid2 = vid(:);
                
                % and repeat vector as rows into a matrix with N rows for fast comparison with light intensity mapping
                repvid = vid2(:,ones(1,N));
                
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
                Fval(:,:,i) = sum(final,3);
            end
            
            % save Fval for later
            if(ii == 1)
                Fvalshort = Fval;
            else
                Fvallong = Fval;
            end
            
            if(kk == 1)
                % create Mdata structure
                massdata{ii} = {MPC; PVD}; %#ok<SAGROW>
            end
        end
    end
          
    %% Step 4, Save results
    % save Fval as a .mat file
    filename = sprintf('%s%s_interp%g_wt_percent_array.mat',maindirectory,exp,R(1));
    save(filename, 'Fvalshort','Fvallong')
end
toc

%% ---------------------------------------- VISUALISATION -----------------------------------------

    %% Step 1, Set up common parameters
if( sum(Visual) > 0)
    % load parameters from previous modules
    % module one
    if(Image_prep == 0)
        % alert tag
        sprintf('Loading crop parameters')
        filename = sprintf('%s%s_crop_parameters',maindirectory,exp);
        load(filename);
    end
    % module 2
    if(Data_analysis == 0)
        % alert tag
        sprintf('Loading fingering wavelength analysis results')
        % load results from data analysis
        filename = sprintf('%s%s_interp%g_Data_analysis_results.mat',maindirectory,exp,R(1));
        load(filename);
    end
    % module 3
    if(Particle_conc == 0)
        % alert tag
        sprintf('Loading Fval')
        % load Fval array for visualision
        filename = sprintf('%s%s_interp%g_wt_percent_array.mat',maindirectory,exp,R(1));
        load(filename);
    end
        
    % time vector
    Tstartshort = 0; % start time in seconds
    Tendshort = framerate(1)*numframes(1)/1000; % end time of first portion in seconds
    Tstartlong = Tendshort + framerate(2)/1000; % start time of second portion in seconds
    Tendlong = framerate(2)*numframes(2)/1000; % end time of exp
    % create time vector for plotting
    time = [linspace(Tstartshort,Tendshort,numframes(1)), linspace(Tstartlong,Tendlong,numframes(2))];
    
    % total number of frames
    vidlen = numframes(1) + numframes(2);
  %%  
    % merge Fvals into one.
    Fvalcombi = Fvalshort;
    Fvalcombi(:,:,numframes(1)+1:vidlen) = Fvallong;
    
    % load pbl images
    parfor i = 1:numframes(1)
        pblimtemp(:,:,i) = imread(sprintf('%simages/%s/pbl%g.tif',maindirectory,lentyp{1},i),'tif');
    end
    for i = 1:numframes(2)
        pblimtemp(:,:,i+numframes(1)) = imread(sprintf('%simages/%s/pbl%g.tif',maindirectory,lentyp{2},i),'tif');
    end
    
end

    %% Step 2, Normal unprocessed video
if( Visual(1) == 1)
    % load original images
    parfor i = 1:numframes(1)
        cropim(:,:,i) = imread(sprintf('%simages/%s/%g.tif',maindirectory,lentyp{1},i),'tif');
    end
    for i = 1:numframes(2)
        cropim(:,:,i+numframes(1)) = imread(sprintf('%simages/%s/%g.tif',maindirectory,lentyp{2},i),'tif');
    end
    
    % timestamp images
    parfor i = 1:vidlen
        % stamp image
        cropim(:,:,i) = labelimg(cropim(:,:,i), sprintf('%06.1f s',time(i)), 2, 1, 0, 0, 1, 3, 0.5, 1, [0, 0, 0], [1, 1, 1]);
    end
    % normalise video
    cropim = im2double(cropim);
    cropim = cropim/max(max(max(cropim)));
    
    % build video
    % name video
    filename = sprintf('%s/%s_Original_Video.avi',maindirectory,exp);
    % get handle
    vid = VideoWriter(filename);
    % set frame rate
    vid.FrameRate = 7;
    % open video
    open(vid);
    % feed open video the frames
    for i = 1:skiprate:vidlen
        frame = cropim(:,:,i);
        writeVideo(vid, frame);
    end
    % close video
    close(vid);
end

    %% Step 3, Integrate total mass of cloud for each frame
if( Visual(2) == 1)
    % crop to cloud size
    cloudval = Fvalcombi(cloudcrop(3):cloudcrop(4),:,:);
    
    % sum cloud into one value in each frame
    parfor i = 1:vidlen
        Mval(i) = sum(sum(cloudval(:,:,i)));
    end
    
    % normalise the array
    Mval = Mval/max(Mval);
    
    % normalise time
    time = time/max(time);
    
    % Plot result
    fig = figure(10);
    plot(time,Mval)
    title('Normalised mass of cloud over normalised time')
    xlabel('time')
    ylabel('Mass')
    
    % save figure
    filename = sprintf('%s%s_interp%g_Norm_Mass_Norm_Time.tif',maindirectory,exp,R(1));
    saveas(fig,filename,'tif')
end

    %% Step 4, Horizontal mass integration plots
if( Visual(3) == 1)
    % horizontally integrate Fval
    FvalhintS = sum(Fvalshort,2);
    FvalhintL = sum(Fvallong,2);
    % lose singleton dimension
    Fvalhintshort = squeeze(FvalhintS);
    Fvalhintlong = squeeze(FvalhintL);
    % replicate frames in the second section as needed to correct for the
    % time factor
    % set original time vector
    origtime = linspace(Tstartlong,Tendlong,numframes(2));
    % set time vector to interpolate onto
    interptime = Tstartlong:framerate(1)/1000:Tendlong;
    % initialise array for interp
    Fvalhintlonginterp = zeros(size(Fvalhintlong,1),size(interptime,2));
    for i = 1:size(Fvalhintlong,1)
    Fvalhintlonginterp(i,:) = interp1(origtime,Fvalhintlong(i,:),interptime);
    end
    % combine into one array
    finalhint = [Fvalhintshort, Fvalhintlonginterp];
    % normalisethe arrays
    finalhint = finalhint/(max(max(finalhint)));
    Fvalhintshort = Fvalhintshort/max(max(Fvalhintshort));
    
    % plot as 2D image
    fig = figure(20);
    imagesc([time(1) time(end)],[0 30],finalhint);
    % fix range of color bar
    caxis([0 1]),colorbar
    % set interpolating shading
    shading interp
    % annotate plot
    title('Horizontally integrated particle mass - Color bar represents normalized mass - full range plot')
    xlabel('time (s)')
    ylabel('Side of tank (cm)')
    % save figure
    filename = sprintf('%s%s_interp%g_H_Integrated_Fig_FullRange.tif',maindirectory,exp,R(1));
    
    saveas(fig,filename,'tif')
     % plot as 2D image
    fig = figure(21);
    imagesc([time(1) time(end)],[0 30],finalhint);
    % fix range of color bar
    caxis([0 0.4]),colorbar
    % set interpolating shading
    shading interp
    % annotate plot
    title('Horizontally integrated particle mass - Color bar represents normalized mass - low range plot')
    xlabel('time (s)')
    ylabel('Side of tank (cm)')
    % save figure
    filename = sprintf('%s%s_interp%g_H_Integrated_Fig_LowRange.tif',maindirectory,exp,R(1));
    saveas(fig,filename,'tif')
    
     % plot as 2D image
    fig = figure(22);
    imagesc([time(1) time(numframes(1))],[0 30],Fvalhintshort);
    % fix range of color bar
    caxis([0 1]),colorbar
    % set interpolating shading
    shading interp
    % annotate plot
    title('Horizontally integrated particle mass - Color bar represents normalized mass - full range plot')
    xlabel('time (s)')
    ylabel('Side of tank (cm)')
    % save figure
    filename = sprintf('%s%s_interp%g_H_Integrated_Fig_FullRangeShort.tif',maindirectory,exp,R(1));
    saveas(fig,filename,'tif')
    
     % plot as surface
    fig = figure(23);
    x = [time(1) time(end)];
    xdif = (x(2) - x(1))/(size(finalhint,2)-1);
    y = [0 30];
    ydif = (y(2) - y(1))/(size(finalhint,1)-1);
    contour(x(1):xdif:x(2),y(1):ydif:y(2),flipud(finalhint),12);
    % fix range of color bar
    caxis([0 1]),colorbar
    % set interpolating shading
    shading interp
    % annotate plot
    title('Horizontally integrated particle mass contour plot')
    xlabel('Time (s)')
    ylabel('Side of tank (cm)')
    % save figure
    filename = sprintf('%s%s_interp%g_H_Integrated_Fig_Contour.tif',maindirectory,exp,R(1));
    saveas(fig,filename,'tif')
   
    % plot as surface
    fig = figure(24);
    x = [time(1) time(end)];
    xdif = (x(2) - x(1))/(size(finalhint,2)-1);
    y = [0 30];
    ydif = (y(2) - y(1))/(size(finalhint,1)-1);
    surf(x(1):xdif:x(2),y(1):ydif:y(2),flipud(finalhint));
    % fix range of color bar
    caxis([0 .5]),colorbar
    % set interpolating shading
    shading interp
    % annotate plot
    title('Horizontally integrated particle mass')
    xlabel('Time (s)')
    ylabel('Side of tank (cm)')
    zlabel('Normalized mass')
    % save figure
    filename = sprintf('%s%s_interp%g_H_Integrated_Fig_Surface.tif',maindirectory,exp,R(1));
    saveas(fig,filename,'tif')
    
    % take every 10th frame
    slice = zeros(size(finalhint,1),round(size(finalhint,2)/10)+1);
    count = 0;
    for i = 1 : 20 : length(finalhint)
        count = count + 1;
        slice(:,count) = finalhint(:,i);
    end
    [slicedeep slicelen] = size(slice);
    posit = linspace(1,300,slicedeep);
    sliceint = zeros(slicedeep,slicelen*4);
    for i = 1:slicedeep
    % interpolate between frames
    sliceint(i,:) = interp1(1:slicelen,slice(i,:),1:.25:slicelen);
    end
    % build video
    % name video
    filename = sprintf('%s/%s_Interp_%g_H_Integrated_Particle_Conc_Video.avi',maindirectory,exp,R(1));
    % get handle
    vid = VideoWriter(filename);
    % set frame rate
    vid.FrameRate = 7;
    % set compression quality
    vid.Quality = 50;
    % open video
    open(vid);
    % create color images from Fvalcombi
    % and feed to open video
    for i = 1:vidlen
        % select image
        I = sliceint(:,i);
        % get handle and set figure to invisible
        fig = figure('visible','off');
        % plot image
        plot(posit,I);
        % annotate plot
        title(sprintf('Horizontally integrated particle mass -- Time: %06.1f s',time(i)));
        ylabel('Normalised Mass')
        xlabel('Position on side of tank in mm - Origin is top of tank')
        % get image from fig
        frame = getFrameForFigure(fig);
        % feed open video the frames
        writeVideo(vid, frame);
    end
    % close video
    close(vid);
end

    %% Step 5, Particle concentration video
if( Visual(4) == 1)

    % build video
    % name video
    filename = sprintf('%s/%s_Interp_%g_Particle_Conc_Video.avi',maindirectory,exp,R(1));
    % get handle
    vid = VideoWriter(filename);
    % set frame rate
    vid.FrameRate = 7;
    % set compression quality
    vid.Quality = 80;
    % open video
    open(vid);
    % create color images from Fvalcombi
    % and feed to open video
    for i = 1:skiprate:vidlen
        % select image
        I = Fvalcombi(:,:,i);
        % get handle and set figure to invisible
        fig = figure('visible','off');
        % plot image
        imshow(I,[],'border','tight');
        % set image size
        set(fig, 'Position', [1 1 1445 500])
        % false color image
        colormap('jet')
        % set color bar and color bar range
        caxis([0 2])
        colorbar
        % get image from fig
        frame = getFrameForFigure(fig);
        % feed open video the frames
        writeVideo(vid, frame);
    end
    % close video
    close(vid);
end

    %% Step 6, PBL video
if( Visual(5) == 1)

    % cut PBL image in half and restitch in a stacked sense to give a closer view
    pblim = [pblimtemp(:,1:end/2,:); zeros(3,size(pblimtemp,2)/2,size(pblimtemp,3)); pblimtemp(:,end/2+1:end,:)];
    
    % timestamp images
    parfor i = 1:vidlen
        % stamp image
        pblim(:,:,i) = labelimg(pblim(:,:,i), sprintf('%06.1f s',time(i)), 1, 1, 0, 0, 1, 3, 0.5, 1, [0, 0, 0], [1, 1, 1]);
    end
    % normalise video
    pblim = im2double(pblim);
    pblim = pblim/max(max(max(pblim)));
    
    % build video
    % name video
    filename = sprintf('%s/%s_PBL_video.avi',maindirectory,exp);
    % get handle
    vid = VideoWriter(filename);
    % set frame rate
    vid.FrameRate = 7;
    % open video
    open(vid);
    % feed open video the frames
    for i = 1:pblskiprate:vidlen
        frame = pblim(:,:,i);
        writeVideo(vid, frame);
    end
    % close video
    close(vid);
end

    %% Step 7, PBL video with wavelength analysis
if( Visual(6) == 1)
    % timestamp images
    parfor i = 1:vidlen
        % stamp image
        D(:,:,i) = labelimg(pblim(:,:,i), sprintf('%06.1f s',time(i)), 2, 1, 0, 0, 1, 3, 0.5, 1, [0, 0, 0], [1, 1, 1]);
    end
    % set up wavelength and wavenumber vectors
    WvPowP = powernorm.poly;
    WvLenP = wavelength.poly;
    WvPowD = powernorm.diff;
    WvLenD = wavelength.diff;
    
    % name video
    filename = sprintf('%s%s_gamma%g_PBL_with_Wavelength_Analysis',maindirectory,exp,gamma);
    % get handle
    vid = VideoWriter(filename);
    % set frame rate
    vid.FrameRate = 7;
    % open video
    open(vid);
    % create figure
    fig = figure('visible','off');
    % plot figure and feed to videowriter
    for i = 1:pblskiprate:vidlen
        subplot(3,1,1)
        % create figure to be saved for movie
        semilogx(WvLenP(i,:),WvPowP(i,:),'b'), xlim([1 350])
        % annotate plot
        title('Power spectrum in wavelengths using polynomial removal method')
        xlabel('Wavelengths (in mm)')
        ylabel('Normalised power')
        
        subplot(3,1,2)
        % create figure to be saved for movie
        semilogx(WvLenD(i,:),WvPowD(i,:),'b'), xlim([1 350])
        % annotate plot
        title('Power spectrum in wavelengths using difference method')
        xlabel('Wavelengths (in mm)')
        ylabel('Normalised power')
        
        subplot(3,1,3)
        % create figure to be saved for movie
        imshow(D(:,:,i),'border','tight')
        % annotate plot
        title('Particle Boundary Layer')
        xlabel('Bottom of tank - total length 650 mm')
        ylabel('Side of tank - 300 mm')
        % save figure
        frame = getFrameForFig(fig);
        writeVideo(vid, frame);
    end
    % close video
    close(vid);
end

    %% Step 8, Snake plot of wavelength analysis

%% ------------------------------------------- CLEAN UP -------------------------------------------
%matlabpool close
%{}
