function res=process_ni(ms,mpt)
    
    folder='C:\Users\PIVUSER\Desktop\RL_VerticalAxisTurbine\Carousel\2023_BC\bc001\raw\20230711\';
    
    load(append(folder,sprintf('ms%03dmpt%03d_1.mat',ms,mpt)))
    load(append(folder,sprintf('ms%03dmpt%03d_2.mat',ms,mpt)))
    load(append(folder,sprintf('ms%03dmpt%03d_3.mat',ms,mpt)))
   
    % figure;
    % plot(t_g,-volts_raw_g(:,2));
    % hold on
    % plot(t_ni,volts_ni(:,2));
    
    % Resample galil volts at 1kHz
    volts_g_resync=single(interp1(t_g,volts_raw_g,t_ni,'linear',0));
    % galil starts after the NI, so find the range where both are recording
    range=find(t_ni>t_g(1));
    
    % find the delay between the start of the loadcell and the reference time
    % recorded in python to resynchronize NI and galil volts
    delay=finddelay(-volts_g_resync(range,1),volts_ni(range,1));
    
    plot(t_ni(range(delay:end)),-volts_g_resync(range(1:end-delay+1),2));
    hold on
    %reinterpolate all galil quantities + correct delay
    pitch_is_r=single(interp1(t_g,pitch_is,t_ni,'linear',0));
    pitch_is_r=pitch_is_r(range(delay:end));
    
    pitch_should_r=single(interp1(t_g,pitch_should,t_ni,'linear',0));
    pitch_should_r=pitch_should_r(range(delay:end));
    
    pitch_should_r=single(interp1(t_g,pitch_should,t_ni,'linear',0));
    pitch_should_r=pitch_should_r(range(delay:end));
    
    dpitch_filtered_r=single(interp1(t_g,dpitch_filtered,t_ni,'linear',0));
    dpitch_filtered_r=dpitch_filtered_r(range(delay:end));
    
    dpitch_noisy_r=single(interp1(t_g,dpitch_noisy,t_ni,'linear',0));
    dpitch_noisy_r=dpitch_noisy_r(range(delay:end));
    
    phase_cont_r=single(interp1(t_g,phase_cont,t_ni,'linear',0));
    phase_cont_r=phase_cont_r(range(delay:end));
    
    forces_g_r=single(interp1(t_g,forces_g,t_ni,'linear',0));
    forces_g_r=forces_g_r(range(delay:end),:);
    
    forces_noisy_g_r=single(interp1(t_g,forces_noisy_g,t_ni,'linear',0));
    forces_noisy_g_r=forces_noisy_g_r(range(delay:end));
    
    forces_butter_g_r=single(interp1(t_g,forces_butter_g,t_ni,'linear',0));
    forces_butter_g_r=forces_butter_g_r(range(delay:end),:);
    
    force_coeff_g_r=single(interp1(t_g,force_coeff_g,t_ni,'linear',0));
    force_coeff_g_r=force_coeff_g_r(range(delay:end),:);
    
    phase_r=mod(phase_cont_r, 360);
    
    volts=volts_ni(range(1:end-delay+1),:);
    t=t_ni(range(1:end-delay+1));
    
    % add missing parameters
    param_definition;
    param.Ftmodel=param.F0;
    param.Ueff = param.Uinf* sqrt(1 + 2*param.lambda*cosd(phase_r) + param.lambda^2);
    param.Fsp=param.Csp .* 0.5 .* param.rho .* (param.Ueff).^2 .* param.spr^2*pi*2;
    param.denom=param.f_denom;
    % number of rotation for postprocessing
    param.n_rot_trans = 0;
    param.n_rot = 5;
    
    % Remove offset
    volts=volts+v_offset_g;
    
    % compute forces
    forces = compute_forces_lc4(volts, param.R4, param, true, true);
    
    pitch=deg2rad(pitch_is_r);
    
    % NOTE THAT raw.pitch is ALREADY IN RADIAN!
    pitch(isnan(pitch)) = 0;  % Set nans to 0, otherwise filterAR will output only nans.
    pitch = dfilter(pitch, 1000, 30, 3);
    dpitch = gradient2(pitch, t);
    dpitch =  dfilter(dpitch, 1000, 30, 3);
    ddpitch = gradient2(dpitch, t);
    ddpitch =  dfilter(ddpitch, 1000, 30, 3);
    
    % Compute the loads in the carousel reference
    projected_forces = project_forces_b(forces, rad2deg(pitch), param);
    %% STORE RES DATA
    res_range = int64(find(...
    phase_cont_r >= (param.n_rot_trans) * 360 &  ...
    phase_cont_r < (param.n_rot+param.n_rot_trans) * 360));
    
    % Pitch angle
    res.pitch = pitch;
    res.dpitch = dpitch;
    res.ddpitch = ddpitch;
    
    res.phase = mod(phase_cont_r, 360);
    res.phase_cont = phase_cont_r;
    t_start = t(1);
    res.t = t - t_start;
    res.t_T = res.t / param.rotT;
    res.tc = res.t * param.Ub / param.c;
    res.N = length(res.t);
    res.alpha = atand(sind(res.phase)./(param.lambda + cosd(res.phase)));
    res.Ueff = param.Ueff;
    dt = diff(res.t(1:2));
    res.tc_Ueff = cumsum(res.Ueff*dt/param.c);
    %     res.pitch = out.pitch;
    
    
    % Forces
    res.Fx   = forces(:,1);
    res.Fy   = forces(:,2);
    % res.Mx   = forces(:,3);
    % res.My   = forces(:,4);
    res.Mz   = forces(:,3);
    res.Fr   = projected_forces.Fr  ;
    res.Ft   = projected_forces.Ft  ;
    res.Ftot = projected_forces.Ftot;
    res.Ctot = projected_forces.Ctot;
    res.Cr   = projected_forces.Cr  ;
    res.Ct   = projected_forces.Ct  ;
    res.Cm   = res.Mz ./ (param.denom * param.c);
    % res.Cmx  = forces(:, 3) ./ (param.denom * param.c);
    % res.Cmy  = forces(:, 4) ./ (param.denom * param.c);
    
    % Power and power coefficient
    % Moment of inertia. Compute angular acceleration (filtered)
    % NOTE: res.Mz = res.Mzfull, the intertial moment was not removed.
    res.Cp = Cp_vawt_inst(res.Ft, res.Mz, res.dpitch, param);
    
    % Add Cp to the file
    param.Cp = mean(res.Cp(res_range));
    
    % figure;
    % plot(phase_cont/360,pitch_is);
    % 
    figure;
    % 
    plot(reward);
    % 
    % t=phase;
    % y=pitch_is;
    % [pks,locsp] = findpeaks(t);                                 % Find Maxima & Indices
    % [vls,locsv] = findpeaks(-t);                                % Find Minima & Indices
    % pkidx = sort([locsv locsp(locsp > locsv(1))]);              % Edit & Sort Indices
    % for k = 1:2:numel(pkidx)-1
    %     idxrng = pkidx(k):pkidx(k+1);                           % Index Range For Each Segment
    %     tv{k} = t(idxrng);                                      % Corrseponding Time Vector
    %     Asc{k} = y(idxrng);                                     % Corresponding Signal Segment Vector
    % end
    % 
    % figure;
    % c=gray(numel(tv));
    % c = flipud(c);
    % % plot(t, y)                                                  % Plot Original Signal
    % hold on
    % for k = 1:numel(tv)
    %     plot(tv{k}, Asc{k},'color',c(k,:))                % plot Results
    % end
    % hold off
    % grid
    
end
