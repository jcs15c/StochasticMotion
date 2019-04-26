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
xmin = 1;
xmax = 2;
ymin = 1;
ymax = 2;

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
dt = 0.1;% our step size for the random walker
nruns = 100;% number of runs per initial location

%% Step 4: Definte initial walker positions
nx = 500; %number of points in the x direction
ny = 500; %number of points in the y direction

% Create grid of walkers
[x,y] = meshgrid(linspace(xmin,xmax,nx),linspace(ymin,ymax,ny));
init_pts = x(:) + 1i*y(:);% initial location of the random walks
npts = numel(init_pts);% number of initial points

mfpt = zeros(npts,1);% initialize our mean first passage time vector

%% Step 5: For all walkers that start at the same initial position,
%%         apply Euler-Maruyama while any points have not yet collided
for k = 1:numel(init_pts)
    disp(init_pts(k))
      
    % Initializearray of walkers
    z = init_pts(k)*ones(nruns,1); 
    zold = z;
    znew = zold;
     
    inside = zeros(size(z));
    for i=1:nruns
        inside(i) = any( abs( znew(i) - centers ) < radii );
    end      


    fpt = zeros(nruns,1);
    % min_dist.m
    while any(~inside) 
        s = find(~inside);
        
        % Store MFPT data
        fpt(s) = fpt(s) + dt;

        %% Step 5.a: Compute Diffusive Term
        omega = 2*pi*rand(nruns,1);
        alpha = rand(nruns, 1);

        %% Step 5.b: Compute Advective Term
        xinterp = Fx( real(zold(s)), imag(zold(s)) );
        yinterp = Fy( real(zold(s)), imag(zold(s)) );

        %% Step 5.c: Scale the two terms, sum them with old position
        znew(s) = zold(s) + p1*exp(1i*omega(s)) .* sqrt(-dt*2*log(alpha(s))) + ...
                            p2*dt*(xinterp + 1i*yinterp);

             
        %% Step 5.d: Check for collisions
        inside = zeros(size(z));
        for i=1:nruns
            inside(i) = any( abs( znew(i) - centers ) < radii );
        end       

        % Check top bound
        if any((imag(znew) > ymax))
            s = find((imag(znew) > ymax));    
            %znew(s) = real(znew(s)) + 1i*ymax; % Reflecting
            inside(s) = 1; % Absorbing
        end

        % Check bottom bound
        if any((imag(znew) < ymin))
            s = find((imag(znew) < ymin));    
            %znew(s) = real(znew(s)) + 1i*ymin; % Reflecting
            inside(s) = 1; % Absorbing 
        end

        % Check left bound
        if any((real(znew) < xmin))
            s = find((real(znew) < xmin));    
            %znew(s) = xmin + 1i*imag(znew(s)); % Reflecting
            inside(s) = 1; % Absorbing
        end

        % Check right bound
        if any((real(znew) > xmax))
            s = find((real(znew) > xmax));    
            %znew(s) = xmax + 1i*imag(znew(s)); % Reflecting
            inside(s) = 1; % Absorbing
        end

        zold = znew;
    end
    
    mfpt(k) = mean(fpt); % stores the mean first passage of the random walker
end

%% Step 6: Record and plot MFPT for each initial position
figure(1);
clf;
mfpt = reshape(mfpt,ny,nx);
ss = find(mfpt < 0.01*dt);
mfpt(ss) = 1/0;

surf(x,y,mfpt);
colorbar
view(2); shading interp;

axis equal; hold on

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

% Function to read velocity data
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

    
    