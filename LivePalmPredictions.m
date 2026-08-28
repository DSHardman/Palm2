% PalmMesh.m runs first to set up mesh

% Connect to board
clear device
device = serialport("COM16",115200);
device.Timeout = 25;
device.write("y", "string");

% Read empty frame
baseline = readamodo(device, 1);

accumulateddata = [];

n = 600; % Number of data frames to read
for i = 1:n
    i
    data = readamodo(device, 1);
    accumulateddata = [accumulateddata; data];

    clf;
    rec_img= inv_solve(inv2d, baseline.', data.');

    rec_img.calc_colours.ref_level = -0.1;
    rec_img.calc_colours.clim = 0.1;

    subplot(1,2,1);
    plot(data(find(data~=0)), 'linewidth', 2, 'color', 'b');
    ylim([0 2.5e-3]);
    xlim([0 208]);
    set(gca, 'linewidth', 2);
    box off;

    subplot(1,2,2);
    f = show_fem(rec_img, [0, 1]);
    set(gcf, 'color', 'w');
    drawnow();
    % pause();
end

clear device

function data = readamodo(device, n)
    if nargin == 1
        n = 1;
    end
    flush(device);
    readline(device);
    data = zeros([1 208]);
    for i = 1:n
        temp = split(readline(device), ",");
        for j = 1:208
            if contains(char(temp(j)), 'C')
                fprintf("Clipped\n");
                unclipped = char(temp(j));
                unclipped = unclipped(1:end-1);
                data(j) = data(j) + str2num(unclipped);
            else
                data(j) = data(j) + str2num(temp(j));
            end
        end
    end
    data = data./n;
end