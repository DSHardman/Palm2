%% Create mesh
shape = [0 0; 46 0; 42.5 27; 66 33; 73 33; 79.5 38.5; 79.5 66; 58.5 74.5; 44 74.5; 0 56];
% a counter- clockwise list of points in 2D defining the outer contour

ep = [3 57.4; 20.8 65.1; 37.5 72.1; 56 74.7; 74.8 67.8; 79.3 52.4; 76.8 36.2; 60.9 31.8;...
    43.4 27.2; 44.6 13; 41.5 0; 26.6 0; 10.4 0; 0 12; 0 27; 0 42.9];
% a list of points in 2D (will be projected on closest edge of the first contour specified in shape)


ng_write_opt('very_fine'); % very_coarse, coarse, moderate, fine, very_fine 
% Changes only seem to apply if matlab is closed & reopened
mdl = ng_mk_2d_model(shape, ep);

[stim, meas_sel] = mk_stim_patterns(16, 1, '{ad}', '{ad}');
mdl.stimulation = stim; mdl.meas_sel = meas_sel;


%% Solve forward homogeneous model
sim_img= mk_image(mdl,1);
homo_c = 0.3;
for j = 1:length(sim_img.elem_data)
    sim_img.elem_data(j) = homo_c;
end
sim_img.fwd_solve.get_all_meas = 1;
vh = fwd_solve(sim_img);

% Inclusions
inclusion_c = 0.7;
location1 = [50 50]; location2 = [11 21]; radius = 8;
for i = 1:length(mdl.elems)
    corners = [mdl.nodes(mdl.elems(i, 1), :);...
        mdl.nodes(mdl.elems(i, 2), :);...
        mdl.nodes(mdl.elems(i, 3), :)];
    centroid = mean(corners);
    if rssq(centroid-location1) <= radius || rssq(centroid-location2) <= radius
        sim_img.elem_data(i) = inclusion_c;
    end
end
sim_img.fwd_solve.get_all_meas = 1;
vc = fwd_solve(sim_img);

%% Inverse
inv2d= eidors_obj('inv_model', 'EIT inverse');
inv2d.reconst_type= 'difference';
inv2d.jacobian_bkgnd.value = 1;
inv2d.fwd_model= mdl;
inv2d.hyperparameter.value = 0.00005;
inv2d.solve=       'inv_solve_diff_GN_one_step';
inv2d.RtR_prior=   'prior_laplace';
inv2d.stimulation = mdl.stimulation;
rec_img= inv_solve(inv2d, vh.meas, vc.meas);

%% Plot ground truth and reconstruction
subplot(1,3,1);
show_fem(sim_img, [0, 1]); axis off;
title("Ground Truth");

subplot(1,3,2);
plot(vh.meas, 'color', 'b');
hold on
plot(vc.meas, 'color', 'r');
box off
xlim([1 length(vh.meas)]);
title("Signals");

subplot(1,3,3);
f = show_fem(rec_img, [0, 1]); axis off;
colorbar off;
title("Reconstruction");

set(gcf, 'color', 'w', 'position', [77 351 1387 343]);
