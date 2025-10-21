clear; close all; clc;
sigma = 0.3; sigma_x = 0.25; sigma_y = 0.25;
K_values = [1, 2, 3, 4];

% Choose the random location
angle_true = 2*pi*rand();
radius_true = sqrt(rand());  % sqrt for uniform distribution in circle
x_T = radius_true * cos(angle_true);
y_T = radius_true * sin(angle_true);

%Lines that the contours will be drawn at
contour_levels = [0.5, 1, 2, 4, 8, 16, 32, 64, 128];

figure('Position', [100, 100, 1400, 900]);

for idx = 1:length(K_values)
    K = K_values(idx);
    
    % Place the landmarks
    angles = linspace(0, 2*pi, K+1);
    angles = angles(1:K);  % Remove duplicate point at 2*pi
    landmarks = [cos(angles); sin(angles)]';  % K x 2 matrix
    
    % Calculate all of the distances
    d_true = zeros(K, 1);
    for i = 1:K
        d_true(i) = norm([x_T; y_T] - landmarks(i, :)');
    end
    
    % Add the noise to measurements and ensure they aren't negative
    r_measurements = zeros(K, 1);
    for i = 1:K
        valid_measurement = false;
        while ~valid_measurement
            noise = sigma * randn();
            r_measurements(i) = d_true(i) + noise;
            if r_measurements(i) >= 0
                valid_measurement = true;
            end
        end
    end
    
    % Map function call so it is simpler to call
    map_objective = @(pos) compute_map_objective(pos, landmarks, ...
                           r_measurements, sigma_x, sigma_y, sigma);
    
    x_range = linspace(-2, 2, 200);
    y_range = linspace(-2, 2, 200);
    [X, Y] = meshgrid(x_range, y_range);
    Z = zeros(size(X));
    
    % Get the MAP at each grid point
    for i = 1:size(X, 1)
        for j = 1:size(X, 2)
            Z(i, j) = map_objective([X(i, j); Y(i, j)]);
        end
    end
    
    %Find the minimum value
    options = optimoptions('fminunc', 'Display', 'off', ...
                          'Algorithm', 'quasi-newton');
    [pos_MAP, obj_MAP] = fminunc(map_objective, [0; 0], options);
    
    % Calculate error
    error_distance = norm(pos_MAP - [x_T; y_T]);
    
    % Plot the contours
    subplot(2, 2, idx);
    contourf(X, Y, Z, contour_levels, 'LineWidth', 1.5);
    hold on;
    contour(X, Y, Z, contour_levels, 'k', 'LineWidth', 0.5);
    plot(x_T, y_T, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
    plot(landmarks(:, 1), landmarks(:, 2), 'ko', 'MarkerSize', 10, ...
         'LineWidth', 2, 'MarkerFaceColor', 'w');
    
    % Plot estimated location
    plot(pos_MAP(1), pos_MAP(2), 'mx', 'MarkerSize', 15, 'LineWidth', 3);
    
    % Plot unit circle (landmark boundary)
    theta_circle = linspace(0, 2*pi, 100);
    plot(cos(theta_circle), sin(theta_circle), 'k--', 'LineWidth', 1);
    
    hold off;
    
    colorbar;
    caxis([min(contour_levels), max(contour_levels)]);
    
    axis equal;
    xlim([-2, 2]);
    ylim([-2, 2]);
    grid on;
    xlabel('x coordinate');
    ylabel('y coordinate');
    
    title(sprintf(['K = %d | True: (%.3f, %.3f) | MAP: (%.3f, %.3f)\n' ...
                   'Error: %.3f | Min Obj: %.3f'], ...
                   K, x_T, y_T, pos_MAP(1), pos_MAP(2), ...
                   error_distance, obj_MAP), 'FontSize', 9);
    
    legend('MAP contours', '', 'True position', 'Landmarks', ...
           'MAP estimate', 'Unit circle', 'Location', 'best');
end

sgtitle(sprintf(['Vehicle Localization MAP Estimation\n' ...
                 'True position: (%.3f, %.3f), ' ...
                 '\\sigma = %.2f, \\sigma_x = \\sigma_y = %.2f'], ...
                 x_T, y_T, sigma, sigma_x));

% Function to compute the objective MAP
function obj = compute_map_objective(pos, landmarks, measurements, ...
                                     sigma_x, sigma_y, sigma)
    x = pos(1);
    y = pos(2);
    K = length(measurements);
    prior_term = (x^2 / sigma_x^2) + (y^2 / sigma_y^2);
    likelihood_term = 0;
    for i = 1:K
        d_i = norm([x; y] - landmarks(i, :)');
        likelihood_term = likelihood_term + (measurements(i) - d_i)^2;
    end
    likelihood_term = likelihood_term / (sigma^2);
    obj = prior_term + likelihood_term;
end