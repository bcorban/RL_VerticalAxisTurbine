% clear
close all

figure
plot(t_g,forces_noisy_g(:,1)-0.4938)
hold on
% plot(t_g, foo)
plot(t_g,forces_g(:,1))

% 
% figure 
% plot(t_g,pitch_is)
% hold on
% plot(t_g,pitch_should)
% 
figure
v_raw_offset=(-(volts_raw_g-v_offset_g));
plot(t_g,v_raw_offset(:,3))
hold on
plot(t_g,volts_g(:,3))

fs = 550;
fc = 30;
ws = 100;
forder = 2;
[bfilt, afilt]    = butter(forder,fc/(fs/2));
zfilt = zeros(1, forder);
forces_noisy_fo = filloutliers(forces_noisy_g, "pchip", "movmedian", 10);


figure
plot(forces_noisy_g(:,1))
hold on
plot(foo(:,1))


