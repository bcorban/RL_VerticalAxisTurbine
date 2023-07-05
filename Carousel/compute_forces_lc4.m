function forces = compute_forces_lc4(volts, R4, param, filter, remove_outliers)

%% Function to compute the loads
% INPUTS:
%  - volts
%  - v_offset
%  - R4: calibration matrix
%  - filter: boolean, to filter or not the forces
    %% Remove outliers if necessary
    if remove_outliers
        volts = filloutliers(volts, "pchip", "movmedian", 5);
    end    

    %% Compute the forces in [N]

    % multiply volts matrix with calibration matrix to obtain forces
    forces = volts*R4;   % convert from volt to newtons
    
    % Get forces
    forces(:,1) = forces(:,1) - param.Ftmodel; % include drag offset created from taking zero measurement with flow (signal should physically be at drag of SP and naca)


    %% Filter if necessary
    if filter

        for i =1:size(forces, 2)
            
            % with filtfilt
            forces(:,i) = dfilter(double(forces(:,i)), 1000, 30, 3);  % low-pass filter at 30Hz

            % with filterAR
            % COFreq = 1/30; % 1/cut-off frequency for filter
            % fs = 1000; % acquisition frequency of loads
            % forces(:,i) = filterAR(forces(:,i),COFreq,fs);
        end
    end
end