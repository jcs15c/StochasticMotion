%% Step 0: Load Vector Field Data
% Read velocity file
fileName = 'circles17EulerVelocities.bin';
[nx,ny,eulerX,eulerY,u,v] = ...
    loadEulerVelocities(fileName);

% Read trap file
Xcenters_file = matfile('Xcenters.mat');
Ycenters_file = matfile('Ycenters.mat');
radii_file = matfile('radii.mat');

% Load data into arrays
Xcenters = Xcenters_file.Xcenters;
Ycenters = Ycenters_file.Ycenters;

radii = radii_file.radii;
centers = Xcenters + 1i*Ycenters;
max_rad = max(radii);

% Create interpolating objects that can be queried at given points
Fx = griddedInterpolant( eulerX', eulerY', u', 'linear' );
Fy = griddedInterpolant( eulerX', eulerY', v', 'linear' );

%% Step 1: Define Bounding Geometry
xmin = 0;
xmax = 28;
ymin = 0.01;
ymax = 5.19;

%% Step 2: Define Absorbing Geometries
sL = real(centers) > xmin - 0.3;
sR = real(centers) < xmax + 0.3;
sU = imag(centers) > ymin - 0.3;
sD = imag(centers) < ymax + 0.3;

centers = centers( sL & sR & sU & sD );
radii = radii( sL & sR & sU & sD );

N = numel(centers);

%% Step 3: Initialize Parameters   
p1 = 1; % Diffusion
p2 = 1; % Advection
dt = 0.001;% our step size for the random walker
nruns = 100;% number of runs per initial location

%% Step 4: Initialize Random Walkers

% Initialize walkers in a line
z = xmin + 1i*linspace(ymin, ymax, nruns)';

nruns = numel( z );

fpt = zeros(nruns,1); % vector of first passage times
count_absorbed = 0;   % Number of tracers that hit other edge

z_paths = zeros(nruns, 1000);
z_paths(:, 1) = z(:);
j = 1;
zold = z;
znew = zold;

% Check which walkers start inside circles
inside = zeros(size(z));
for i=1:nruns
    inside(i) = any( abs( znew(i) - centers ) < radii );
end

%% Step 5: For all walkers, 
%%         apply Euler Maruyama while any points have not yet collided
while any(~inside) 
    s = find(~inside);
    disp( sum( inside ) );
    
    fpt(s) = fpt(s) + dt;

    %% Step 5.a: Compute Diffusive Term
    omega = 2*pi*rand(nruns,1);
    alpha = rand(nruns, 1);
    
    % This code would sample from the normal distribution
    %gamma = normrnd(0, sqrt(dt), nruns, 2);
    %znew(s) = zold(s) + p1*dt*(gamma(:,1) + 1i*gamma(:,2)) + ...
    %                    p2*dt*(xinterp + 1i*yinterp);

    
    %% Step 5.b: Compute Advective term
    xinterp = Fx( real(zold(s)), imag(zold(s)) );
    yinterp = Fy( real(zold(s)), imag(zold(s)) );
    
    %% Step 5.c: Scale the two terms, sum them with old position
    znew(s) = zold(s) + p1*exp(1i*omega(s)) .* sqrt(-dt*2*log(alpha(s))) + ...
                        p2*dt*(xinterp + 1i*yinterp);

    j = j + 1;
    z_paths(:,j) = znew;
    
    if mod( j, 1000 ) == 0
        z_paths = [z_paths zeros(nruns, 1000)];
    end
    
    %% Step 5.d: Check for collisions
    inside = zeros(size(z));
    for i=1:nruns
        inside(i) = any( abs( znew(i) - centers ) < radii );
    end
    
    % Check top bound
    if any((imag(znew) > ymax))
        s = find((imag(znew) > ymax));    
        znew(s) = real(znew(s)) + 1i*ymax; % Reflecting
        %inside(s) = 1; % Absorbing
    end

    % Check bottom bound
    if any((imag(znew) < ymin))
        s = find((imag(znew) < ymin));    
        znew(s) = real(znew(s)) + 1i*ymin; % Reflecting
        %inside(s) = 1; % Absorbing 
    end

    % Check left bound
    if any((real(znew) < xmin))
        s = find((real(znew) < xmin));    
        znew(s) = xmin + 1i*imag(znew(s)); % Reflecting
        %inside(s) = 1; % Absorbing
    end

    % Check right bound
    if any((real(znew) > xmax))
        s = find((real(znew) > xmax));    
        %znew(s) = xmax + 1i*imag(znew(s)); % Reflecting
        inside(s) = 1; % Absorbing
        count_absorbed = count_absorbed + 1;
    end


    zold = znew;
end

%% Step 6: Plot the paths of each walker

figure(2);
clf;
axis equal; hold on

% Plot Paths
z_paths = z_paths(:, 1:j);
for i=1:nruns 
    % Check if particle collides with a surface
    if( real(z_paths(i,end)) >= xmax || ...
        real(z_paths(i,end)) <= xmin || ...
        imag(z_paths(i,end)) >= ymax || ...
        imag(z_paths(i,end)) <= xmin)
        p1 = plot( real(z_paths(i,:)), imag(z_paths(i,:)), 'b-' );        
    else
        p1 = plot( real(z_paths(i,:)), imag(z_paths(i,:)), 'b-' );
    end
    p1.Color(4) = 1;
end

% Plot initial walker positions
scatter( real(z), imag(z), 'r.' );

% Plot circles
for i=1:N
    th = 0:pi/50:2*pi;
    xunit = radii(i) * cos(th) + real(centers(i));
    yunit = radii(i) * sin(th) + imag(centers(i));
    h = fill(xunit, yunit, 'k');
    set(h,'edgecolor',[.5 .5 .5])
    set(h,'facecolor',[.75 .75 .75])
end

xlim([xmin xmax])
ylim([ymin ymax])

hold off

% Function to read in velocity data
function [nx,ny,eulerX,eulerY,u,v] = loadEulerVelocities(fileName)

fid = fopen(fileName,'r');
val = fread(fid,'double');
fclose(fid);

nx = val(1);
ny = val(2);
val = val(3:end);

eulerX = zeros(nx,ny);
eulerY = zeros(nx,ny);
u = zeros(nx,ny);
v = zeros(nx,ny);

istart = 1;
% start of a pointer to where everything is stored in val


for k = 1:ny
  iend = istart + nx - 1;
  eulerX(1:nx,k) = val(istart:iend);
  istart = iend + 1;
end

for k = 1:ny
  iend = istart + nx - 1;
  eulerY(1:nx,k) = val(istart:iend);
  istart = iend + 1;
end

for k = 1:ny
  iend = istart + nx - 1;
  u(1:nx,k) = val(istart:iend);
  istart = iend + 1;
end

for k = 1:ny
  iend = istart + nx - 1;
  v(1:nx,k) = val(istart:iend);
  istart = iend + 1;
end

end % loadEulerVelocities

    
    